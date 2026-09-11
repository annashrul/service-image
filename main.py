# FastAPI embedding service untuk Search-by-Image MiniPOS.
#
# Kontrak:
#   GET  /health          -> { status, model, dim, preprocess }
#   POST /embed           -> { vectors, model, dim, failed }
#   POST /embed-text      -> { vectors, model, dim, failed }
#
# Input gambar boleh berupa data URL ("data:image/jpeg;base64,...") atau URL
# http(s) publik (mis. Cloudinary). Vektor dikembalikan SUDAH L2-normalized
# supaya cosine distance pgvector (<=>) langsung sebanding.
#
# /embed-text memakai TEXT TOWER SigLIP (ruang vektor yang sama dengan gambar),
# jadi hasilnya bisa dibandingkan langsung ke products.embedding. Dipakai sebagai
# jaring terakhir Search-by-Image: deskripsi yang dibaca LLM vision dari foto
# ("air mineral botol plastik tanpa label") dicocokkan ke FOTO produk, bukan ke
# nama produk. Skala jaraknya BEDA dari gambar↔gambar — lihat image-search.gate.ts.
#
# `model` yang dilaporkan = "<AI_MODEL_NAME>+<PREPROCESS_TAG>". Preprocessing
# ikut menentukan nilai vektor, jadi tag-nya bagian dari identitas embedding:
# backend memakai string ini untuk mendeteksi baris yang perlu di-embed ulang.
#
# KEAMANAN: kalau env AI_SERVICE_TOKEN di-set, SEMUA endpoint kecuali /health
# wajib mengirim header `Authorization: Bearer <token>`. Service ini dirancang
# untuk dideploy sebagai private service (tidak terekspos internet).
from __future__ import annotations

import base64
import binascii
import io
import logging
import os
from typing import List, Optional

import httpx
import numpy as np
import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from PIL import Image, ImageChops, ImageStat
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-service")

MODEL_NAME = os.getenv("AI_MODEL_NAME", "google/siglip-base-patch16-224")
AI_SERVICE_TOKEN = os.getenv("AI_SERVICE_TOKEN", "")
MAX_BATCH = int(os.getenv("AI_MAX_BATCH", "32"))
FETCH_TIMEOUT = float(os.getenv("AI_FETCH_TIMEOUT", "15"))
MAX_IMAGE_BYTES = int(float(os.getenv("IMAGE_MAX_SIZE_MB", "5")) * 1024 * 1024)

# ─── Versi preprocessing ─────────────────────────────────────────────
# Vektor hanya sebanding kalau DIBUAT dengan preprocessing yang sama, jadi tag
# ini ikut dilaporkan sebagai bagian dari `model` ("<nama>+<tag>"). Backend
# menyimpannya di products."embeddingModel" dan menganggap baris dengan tag
# berbeda sebagai stale → otomatis di-embed ulang oleh sweeper/backfill.
#
# tp3 = trim border → pad ke persegi → 3 view TTA (full + 0.85 + 0.68),
#       weighted average lalu L2-normalize ulang.
# NAIKKAN TAG INI setiap kali langkah/parameter di bawah berubah.
PREPROCESS_TAG = "tp3"
MODEL_TAG = f"{MODEL_NAME}+{PREPROCESS_TAG}"
# Toleransi selisih warna (0..255) saat mendeteksi border seragam.
TRIM_TOL = 12
# Fraksi center-crop untuk view TTA kedua.
TTA_CROP = 0.85
# Fraksi crop fokus objek untuk view TTA ketiga.
TTA_TIGHT_CROP = 0.68

app = FastAPI(title="MiniPOS Embedding Service", version="1.0.0")

_model = None
_processor = None
_dim = 0


def _load_model() -> None:
    """Muat model sekali saat startup. Weights sudah di-bake ke image."""
    global _model, _processor, _dim
    if _model is not None:
        return
    logger.info("memuat model %s ...", MODEL_NAME)
    _processor = AutoProcessor.from_pretrained(MODEL_NAME)
    _model = AutoModel.from_pretrained(MODEL_NAME)
    _model.eval()
    # Dimensi TIDAK di-hardcode — dibaca dari forward pass nyata supaya
    # /health.dim selalu benar walau MODEL_NAME diganti. Sengaja lewat
    # _embed_images() (bukan _encode) supaya seluruh pipeline preprocessing
    # ikut tereksekusi sekali saat startup, bukan pertama kali dipakai user.
    dummy = Image.new("RGB", (224, 224), (127, 127, 127))
    vec = _embed_images([dummy])
    _dim = int(vec.shape[1])
    logger.info("model siap: %s, dim=%d (preprocess=%s)", MODEL_NAME, _dim, PREPROCESS_TAG)


# ─── Preprocessing ───────────────────────────────────────────────────
# Foto katalog = studio, objek kecil di tengah latar putih. Foto dari user =
# framing/rasio berbeda. Tanpa normalisasi, jarak antara dua foto produk yang
# SAMA bisa 0.30 sementara pasangan acak di katalog ~0.46 — dua sebaran itu
# tumpang tindih sehingga tidak ada ambang absolut yang memisahkannya.
# Terukur pada 2 foto uji nyata (76 produk katalog): jarak ke produk benar
# turun 0.3033 → 0.2602 dan 0.2111 → 0.1207, sementara pesaing terdekat TIDAK
# turun (0.4948 → 0.4530, 0.5623 → 0.5100), jadi marginnya melebar.
def _bg_color(img: Image.Image) -> tuple[int, int, int]:
    """Warna latar = median 4 sudut. Foto produk studio hampir selalu putih."""
    w, h = img.size
    k = max(1, min(w, h) // 40)
    boxes = [(0, 0, k, k), (w - k, 0, w, k), (0, h - k, k, h), (w - k, h - k, w, h)]
    px = [tuple(int(v) for v in ImageStat.Stat(img.crop(b)).median[:3]) for b in boxes]
    return tuple(int(np.median([p[i] for p in px])) for i in range(3))  # type: ignore[return-value]


def _trim_border(img: Image.Image, tol: int = TRIM_TOL) -> Image.Image:
    """Buang border yang hampir seragam supaya objek mengisi frame.

    Konservatif: kalau bbox hasil deteksi < 15% sisi (kemungkinan salah deteksi,
    mis. foto gelap merata) atau tidak mengecilkan apa pun, gambar dikembalikan
    apa adanya. Sisa margin 2% ditahan supaya objek tidak terpotong pas di tepi.
    """
    bg = Image.new("RGB", img.size, _bg_color(img))
    diff = ImageChops.difference(img, bg).convert("L")
    box = diff.point(lambda p: 255 if p > tol else 0).getbbox()
    if box is None:
        return img
    w, h = img.size
    nw, nh = box[2] - box[0], box[3] - box[1]
    if nw < w * 0.15 or nh < h * 0.15 or (nw >= w and nh >= h):
        return img
    mx, my = int(nw * 0.02), int(nh * 0.02)
    return img.crop(
        (
            max(0, box[0] - mx),
            max(0, box[1] - my),
            min(w, box[2] + mx),
            min(h, box[3] + my),
        )
    )


def _pad_square(img: Image.Image) -> Image.Image:
    """Letterbox ke persegi memakai warna latar.

    Processor SigLIP me-resize ke 224x224 tanpa menjaga rasio; tanpa padding,
    foto potret akan tergencet berbeda dari foto katalog yang persegi.
    """
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), _bg_color(img))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def _center_crop(img: Image.Image, frac: float) -> Image.Image:
    w, h = img.size
    nw, nh = int(w * frac), int(h * frac)
    left, top = (w - nw) // 2, (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh))


def _views(img: Image.Image) -> List[Image.Image]:
    """View TTA: full, crop sedang, dan crop fokus objek."""
    trimmed = _trim_border(img)
    return [
        _pad_square(trimmed),
        _pad_square(_center_crop(trimmed, TTA_CROP)),
        _pad_square(_center_crop(trimmed, TTA_TIGHT_CROP)),
    ]


# ─── Encoder ─────────────────────────────────────────────────────────
def _encode(images: List[Image.Image]) -> np.ndarray:
    """Jalankan image encoder → matriks (n, dim) yang sudah L2-normalized."""
    inputs = _processor(images=images, return_tensors="pt")
    with torch.inference_mode():
        feats = _model.get_image_features(**inputs)
    arr = feats.detach().cpu().numpy().astype("float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Hindari pembagian nol untuk gambar degenerate.
    norms[norms == 0] = 1.0
    return arr / norms


def _embed_images(images: List[Image.Image]) -> np.ndarray:
    """Preprocess + encode + gabung view → matriks (n, dim) unit-norm.

    Gabungan antar-view = rata-rata vektor yang sudah dinormalisasi, lalu
    dinormalisasi ULANG supaya hasilnya tetap unit-norm (syarat agar cosine
    distance pgvector `<=>` sebanding). Forward pass dipecah per MAX_BATCH view
    supaya puncak memori tidak naik walau tiap gambar jadi beberapa view.
    """
    views: List[Image.Image] = []
    weights: List[float] = []
    spans: List[tuple[int, int]] = []
    for img in images:
        v = _views(img)
        spans.append((len(views), len(views) + len(v)))
        views.extend(v)
        weights.extend((0.5, 0.3, 0.2))

    chunks = [_encode(views[i : i + MAX_BATCH]) for i in range(0, len(views), MAX_BATCH)]
    encoded = np.vstack(chunks)

    merged = np.stack([
        (encoded[a:b] * np.asarray(weights[a:b], dtype=np.float32)[:, None]).sum(axis=0)
        for a, b in spans
    ])
    norms = np.linalg.norm(merged, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return merged / norms


def _embed_texts(texts: List[str]) -> np.ndarray:
    """Encode teks lewat TEXT TOWER → matriks (n, dim) unit-norm.

    Tidak ada preprocessing gambar di sini (PREPROCESS_TAG tidak berlaku untuk
    teks). `padding="max_length"` WAJIB untuk SigLIP: tokenizer-nya dilatih pada
    panjang tetap 64 token, dan padding dinamis menghasilkan vektor yang berbeda.
    """
    inputs = _processor(text=texts, padding="max_length", return_tensors="pt")
    with torch.inference_mode():
        feats = _model.get_text_features(**inputs)
    arr = feats.detach().cpu().numpy().astype("float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


@app.on_event("startup")
def on_startup() -> None:
    _load_model()



# ─── Auth ────────────────────────────────────────────────────────────
def require_token(authorization: Optional[str] = Header(default=None)) -> None:
    """Bearer token wajib kalau AI_SERVICE_TOKEN di-set.

    Kalau env-nya kosong, service dianggap berjalan di jaringan privat
    (docker-compose lokal / Render private service) dan auth dilewati —
    tapi WARNING dicetak supaya tidak diam-diam terbuka.
    """
    if not AI_SERVICE_TOKEN:
        return
    expected = f"Bearer {AI_SERVICE_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Token tidak valid")


# ─── Schemas ─────────────────────────────────────────────────────────
class EmbedRequest(BaseModel):
    images: List[str] = Field(..., min_length=1)


class EmbedTextRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)


class FailedItem(BaseModel):
    index: int
    reason: str


class EmbedResponse(BaseModel):
    vectors: List[List[float]]
    model: str
    dim: int
    failed: List[FailedItem]


class HealthResponse(BaseModel):
    status: str
    model: str
    dim: int
    # Tag preprocessing aktif — supaya operator bisa memastikan embedding di DB
    # dibuat dengan pipeline yang sama (lihat PREPROCESS_TAG).
    preprocess: str


# ─── Pemuatan gambar ─────────────────────────────────────────────────
def _decode_data_url(src: str) -> bytes:
    head, _, payload = src.partition(",")
    if not payload:
        raise ValueError("data URL tidak punya payload base64")
    if "base64" not in head:
        raise ValueError("hanya data URL base64 yang didukung")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"base64 tidak valid: {exc}") from exc


def _fetch_url(client: httpx.Client, url: str) -> bytes:
    res = client.get(url, timeout=FETCH_TIMEOUT, follow_redirects=True)
    res.raise_for_status()
    return res.content


def _load_image(client: httpx.Client, src: str) -> Image.Image:
    src = src.strip()
    if src.startswith("data:"):
        raw = _decode_data_url(src)
    elif src.startswith("http://") or src.startswith("https://"):
        raw = _fetch_url(client, src)
    else:
        raise ValueError("sumber gambar harus data URL atau URL http(s)")

    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"ukuran gambar {len(raw)} byte melebihi batas {MAX_IMAGE_BYTES}"
        )
    img = Image.open(io.BytesIO(raw))
    # convert() memaksa dekode penuh sekaligus menyeragamkan mode (RGBA/P/L →
    # RGB) supaya processor tidak menerima jumlah channel yang tidak konsisten.
    return img.convert("RGB")


# ─── Endpoints ───────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Tanpa auth — dipakai health check platform & introspeksi dimensi."""
    _load_model()
    return HealthResponse(
        status="ok", model=MODEL_TAG, dim=_dim, preprocess=PREPROCESS_TAG
    )


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest, _: None = Depends(require_token)) -> EmbedResponse:
    _load_model()
    if len(req.images) > MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"maksimum {MAX_BATCH} gambar per request, dikirim {len(req.images)}",
        )

    images: List[Image.Image] = []
    failed: List[FailedItem] = []
    with httpx.Client() as client:
        for i, src in enumerate(req.images):
            try:
                images.append(_load_image(client, src))
            except Exception as exc:  # noqa: BLE001 — semua kegagalan per-item dilaporkan
                failed.append(FailedItem(index=i, reason=str(exc)))

    if not images:
        return EmbedResponse(vectors=[], model=MODEL_TAG, dim=_dim, failed=failed)

    try:
        matrix = _embed_images(images)
    except Exception as exc:  # noqa: BLE001
        logger.exception("inference gagal")
        raise HTTPException(status_code=500, detail=f"inference gagal: {exc}") from exc

    return EmbedResponse(
        vectors=[row.tolist() for row in matrix],
        model=MODEL_TAG,
        dim=_dim,
        failed=failed,
    )


@app.post("/embed-text", response_model=EmbedResponse)
def embed_text(req: EmbedTextRequest, _: None = Depends(require_token)) -> EmbedResponse:
    """Embed TEKS ke ruang vektor yang sama dengan gambar (text tower SigLIP).

    Tidak ada kegagalan per-item seperti /embed (tidak ada I/O gambar), jadi
    `failed` selalu kosong — bentuk response disamakan supaya klien bisa memakai
    satu tipe untuk kedua endpoint.
    """
    _load_model()
    if len(req.texts) > MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"maksimum {MAX_BATCH} teks per request, dikirim {len(req.texts)}",
        )

    texts = [t.strip() for t in req.texts if t and t.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="teks tidak boleh kosong")

    try:
        matrix = _embed_texts(texts)
    except Exception as exc:  # noqa: BLE001
        logger.exception("inference teks gagal")
        raise HTTPException(
            status_code=500, detail=f"inference teks gagal: {exc}"
        ) from exc

    return EmbedResponse(
        vectors=[row.tolist() for row in matrix],
        model=MODEL_TAG,
        dim=_dim,
        failed=[],
    )

