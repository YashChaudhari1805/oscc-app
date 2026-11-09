"""Internal inference service. NOT exposed to the internet: only the gateway calls it,
authenticated with a shared secret. Input: raw JPEG/PNG bytes + magnification."""
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import zlib
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from explain import explain_occlusion
from preprocess import InvalidImage, decode_image, preprocess_image, tissue_mask

log = logging.getLogger("inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "../models"))
INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN", "")
MAX_BYTES = int(os.environ.get("MAX_BYTES", 25 * 1024 * 1024))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", 40_000_000))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", 2))
BACKEND = os.environ.get("BACKEND", "auto")  # auto | onnx | keras
LOW_TISSUE = float(os.environ.get("LOW_TISSUE_FRACTION", 0.15))
LOW_RES_SHORT_SIDE = int(os.environ.get("LOW_RES_SHORT_SIDE", 480))
EXPLAIN_ROWS = int(os.environ.get("EXPLAIN_ROWS", 9))
EXPLAIN_COLS = int(os.environ.get("EXPLAIN_COLS", 12))
EXPLAIN_WINDOW = int(os.environ.get("EXPLAIN_WINDOW", 3))
EXPLAIN_STRIDE = int(os.environ.get("EXPLAIN_STRIDE", 2))
EXPLAIN_RANDOM = int(os.environ.get("EXPLAIN_RANDOM", 4))
EXPLAIN_SLOTS = threading.Semaphore(int(os.environ.get("EXPLAIN_MAX_CONCURRENCY", 1)))  # explanations are ~75 forward passes each
MAGS = ("100x", "400x")

if len(INTERNAL_TOKEN) < 16:
    raise SystemExit("INTERNAL_TOKEN must be set (>=16 chars)")


class Model:
    def __init__(self, mag: str, folder: Path):
        self.mag = mag
        cfg = json.loads((folder / "serving_config.json").read_text())
        self.img_hw = tuple(cfg["img_hw"])
        self.cfg = cfg
        onnx_path, keras_path = folder / "oscc_calibrated.onnx", folder / "oscc_calibrated.keras"
        use_onnx = BACKEND == "onnx" or (BACKEND == "auto" and onnx_path.exists())
        if use_onnx:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", 2))
            self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name
            self.backend, self.file = "onnx", onnx_path
        else:
            import keras
            self.km = keras.saving.load_model(keras_path, compile=False)
            self.backend, self.file = "keras", keras_path
        h = hashlib.sha256(self.file.read_bytes()).hexdigest()[:12]
        self.version = f"{mag}-{h}"
        self.lock = threading.Semaphore(MAX_CONCURRENCY)
        # warm-up so the first user request is not slow
        self.predict(np.zeros((1, *self.img_hw, 3), np.uint8))

    def predict_batch(self, batch_uint8: np.ndarray) -> np.ndarray:
        with self.lock:
            if self.backend == "onnx":
                out = self.sess.run(None, {self.input_name: batch_uint8})[0]
            else:
                out = self.km.predict(batch_uint8, verbose=0)[0]
        return np.asarray(out, dtype=np.float64).reshape(len(batch_uint8), -1)[:, 0]

    def predict(self, batch_uint8: np.ndarray) -> float:
        return float(self.predict_batch(batch_uint8)[0])


MODELS: dict[str, Model] = {}
for _m in MAGS:
    _f = MODEL_DIR / _m
    if (_f / "serving_config.json").exists():
        MODELS[_m] = Model(_m, _f)
        log.info("loaded %s (%s, %s)", _m, MODELS[_m].backend, MODELS[_m].version)
if not MODELS:
    raise SystemExit(f"No models found under {MODEL_DIR} (expected {MODEL_DIR}/100x and/or /400x)")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def _auth(request: Request):
    tok = request.headers.get("x-internal-token", "")
    if not secrets.compare_digest(tok.encode(), INTERNAL_TOKEN.encode()):
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/healthz")
def healthz(request: Request):
    _auth(request)
    return {"ok": True, "models": {k: {"backend": v.backend, "version": v.version} for k, v in MODELS.items()}}


@app.post("/v1/predict")
async def predict(request: Request, mag: str = Query(...)):
    _auth(request)
    if mag not in MODELS:
        raise HTTPException(status_code=400, detail="magnification_unavailable")
    cl = request.headers.get("content-length")
    if cl and int(cl) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="too_large")
    data = await request.body()
    if not data or len(data) > MAX_BYTES:
        raise HTTPException(status_code=413 if data else 400, detail="too_large" if data else "empty")
    return _run(MODELS[mag], data)


def _run(model: Model, data: bytes):
    t0 = time.perf_counter()
    try:
        arr, (ow, oh) = decode_image(data, model.img_hw, MAX_PIXELS)
    except InvalidImage as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    img, tissue = preprocess_image(arr, model.img_hw)
    p = model.predict(img[None])
    qc = []
    if tissue < LOW_TISSUE:
        qc.append("LOW_TISSUE")
    if min(ow, oh) < LOW_RES_SHORT_SIDE:
        qc.append("LOW_RESOLUTION")
    return {
        "p_oscc": round(p, 6),
        "mag": model.mag,
        "tissue_fraction": round(tissue, 4),
        "original_size": [ow, oh],
        "qc": qc,
        "model": {"version": model.version, "backend": model.backend, "img_hw": list(model.img_hw)},
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    }


@app.post("/v1/explain")
async def explain(request: Request, mag: str = Query(...)):
    _auth(request)
    if mag not in MODELS:
        raise HTTPException(status_code=400, detail="magnification_unavailable")
    data = await request.body()
    if not data or len(data) > MAX_BYTES:
        raise HTTPException(status_code=413 if data else 400, detail="too_large" if data else "empty")
    if not EXPLAIN_SLOTS.acquire(blocking=False):  # never queue heavy work behind heavy work
        raise HTTPException(status_code=503, detail="busy")
    try:
        return _explain(MODELS[mag], data)
    finally:
        EXPLAIN_SLOTS.release()


def _explain(model: Model, data: bytes):
    t0 = time.perf_counter()
    try:
        arr, (ow, oh) = decode_image(data, model.img_hw, MAX_PIXELS)
    except InvalidImage as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    img, tissue = preprocess_image(arr, model.img_hw)
    glass = img[~tissue_mask(img)]  # hide patches with this image's own blank-glass colour
    fill = np.median(glass, axis=0) if len(glass) > 200 else (240, 240, 240)
    res = explain_occlusion(model.predict_batch, img, rows=EXPLAIN_ROWS, cols=EXPLAIN_COLS, win=EXPLAIN_WINDOW, stride=EXPLAIN_STRIDE,
                            fill=fill, n_random=EXPLAIN_RANDOM, seed=zlib.crc32(data) & 0x7FFFFFFF)
    res.update({"mag": model.mag, "tissue_fraction": round(tissue, 4), "model": {"version": model.version, "backend": model.backend},
                "latency_ms": int((time.perf_counter() - t0) * 1000)})
    return res
