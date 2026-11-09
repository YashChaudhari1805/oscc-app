"""Tests for the occlusion explanation: algorithm correctness on models with KNOWN behaviour, plus the API."""
import io, os
from pathlib import Path

import cv2, numpy as np, pytest
from PIL import Image

os.environ.setdefault("INTERNAL_TOKEN", "t" * 32)
os.environ.setdefault("MODEL_DIR", str(Path(__file__).resolve().parents[2] / "models"))
from explain import _positions, explain_occlusion  # noqa: E402

H, W = 288, 384
sig = lambda z: 1 / (1 + np.exp(-z))


def quadrant_model(x):          # score depends ONLY on how dark the top-left quadrant is
    return sig((200 - x[:, : H // 2, : W // 2, :].astype(float).mean(axis=(1, 2, 3))) / 10)


def global_model(x):            # score depends on the whole image equally: no region is special
    return sig((200 - x.astype(float).mean(axis=(1, 2, 3))) / 10)


def test_highlights_the_region_the_model_actually_uses():
    img = np.full((H, W, 3), 200, np.uint8); img[: H // 2, : W // 2] = 60
    r = explain_occlusion(quadrant_model, img, seed=1)
    rel = np.array(r["relative"])
    inside, outside = rel[:4, :6].mean(), np.r_[rel[4:, :].ravel(), rel[:, 6:].ravel()].mean()
    assert inside > 3 and abs(outside) < 1, (inside, outside)
    assert r["check"]["supported"] and r["check"]["direction"] == "oscc"
    assert r["check"]["p_top_hidden"] < r["check"]["p_random_hidden"]


def test_negative_control_does_not_claim_support():
    r = explain_occlusion(global_model, np.full((H, W, 3), 90, np.uint8), seed=1)
    assert not r["check"]["supported"]          # hiding "top" cells is no better than hiding random ones


def test_constant_model_gives_flat_map_and_no_support():
    r = explain_occlusion(lambda x: np.full(len(x), 0.97), np.full((H, W, 3), 90, np.uint8), seed=1)
    assert r["max_abs"] == 0 and not r["check"]["supported"]


def test_normal_prediction_explains_evidence_against_oscc():
    img = np.full((H, W, 3), 200, np.uint8); img[: H // 2, : W // 2] = 235     # bright quadrant => low score
    r = explain_occlusion(lambda x: sig((x[:, : H // 2, : W // 2].astype(float).mean(axis=(1, 2, 3)) * -1 + 220) / 10 - 4), img, seed=1)
    assert r["p_oscc"] < 0.5 and r["check"]["direction"] == "normal"


def test_deterministic_and_complete_coverage():
    img = np.full((H, W, 3), 200, np.uint8); img[: H // 2, : W // 2] = 60
    assert explain_occlusion(quadrant_model, img, seed=3) == explain_occlusion(quadrant_model, img, seed=3)
    for rows, cols, win, stride in [(9, 12, 3, 2), (9, 12, 3, 1), (6, 8, 2, 3), (4, 4, 4, 2)]:
        cov = np.zeros((rows, cols))
        for a in _positions(rows, win, stride):
            for b in _positions(cols, win, stride):
                cov[a:a + win, b:b + win] += 1
        assert (cov > 0).all(), (rows, cols, win, stride)


def test_rejects_window_larger_than_grid():
    with pytest.raises(ValueError):
        explain_occlusion(global_model, np.zeros((H, W, 3), np.uint8), rows=2, cols=2, win=3)


# ---------------- API (uses the models found in MODEL_DIR) ----------------
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)
HDR = {"x-internal-token": "t" * 32}


def field_jpeg(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((480, 640, 3), (240, 236, 242), np.uint8)
    for _ in range(40):
        cv2.ellipse(img, (int(rng.integers(0, 640)), int(rng.integers(0, 480))), (int(rng.integers(15, 60)), int(rng.integers(10, 40))),
                    int(rng.integers(0, 180)), 0, 360, (int(rng.integers(150, 215)), int(rng.integers(90, 150)), int(rng.integers(150, 205))), -1)
    b = io.BytesIO(); Image.fromarray(cv2.GaussianBlur(img, (0, 0), 3)).save(b, "JPEG", quality=92); return b.getvalue()


def test_api_requires_token_and_validates_input():
    assert client.post("/v1/explain?mag=100x", content=field_jpeg()).status_code == 401
    assert client.post("/v1/explain?mag=40x", content=field_jpeg(), headers=HDR).status_code == 400
    assert client.post("/v1/explain?mag=100x", content=b"", headers=HDR).status_code == 400
    r = client.post("/v1/explain?mag=100x", content=b"not an image", headers=HDR)
    assert r.status_code == 422 and r.json()["error"] == "undecodable_image"


def test_api_contract_and_determinism():
    data = field_jpeg(1)
    a = client.post("/v1/explain?mag=100x", content=data, headers=HDR)
    assert a.status_code == 200, a.text
    j = a.json()
    rows, cols = j["grid"]
    assert len(j["importance"]) == len(j["relative"]) == rows and all(len(x) == cols for x in j["importance"])
    assert np.isfinite(np.array(j["importance"])).all() and 0 <= j["p_oscc"] <= 1
    assert set(j["check"]) >= {"cells_hidden", "direction", "p_before", "p_top_hidden", "p_random_hidden", "delta_top", "delta_random", "supported"}
    b = client.post("/v1/explain?mag=100x", content=data, headers=HDR).json()
    assert b["importance"] == j["importance"] and b["check"] == j["check"]          # same image => same explanation
    # the explained score must equal what /v1/predict says for the same image
    p = client.post("/v1/predict?mag=100x", content=data, headers=HDR).json()["p_oscc"]
    assert abs(p - j["p_oscc"]) < 1e-5


def test_api_busy_when_slot_taken():
    assert server.EXPLAIN_SLOTS.acquire(blocking=False)
    try:
        assert client.post("/v1/explain?mag=100x", content=field_jpeg(), headers=HDR).status_code == 503
    finally:
        server.EXPLAIN_SLOTS.release()
    assert client.post("/v1/explain?mag=100x", content=field_jpeg(), headers=HDR).status_code == 200
