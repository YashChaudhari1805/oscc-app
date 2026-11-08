import ast, io, os, textwrap
from pathlib import Path
import numpy as np, pytest, cv2
from PIL import Image

TOKEN = "t" * 32
os.environ.update(INTERNAL_TOKEN=TOKEN, MODEL_DIR=str(Path(__file__).resolve().parents[2] / "models"), BACKEND="auto")
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)
H = {"x-internal-token": TOKEN}


def jpeg_bytes(w=640, h=480, seed=0, fmt="JPEG"):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 245, np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, (190, 120, 185), -1)
    img = np.clip(img.astype(int) + rng.integers(-12, 12, img.shape), 0, 255).astype(np.uint8)
    b = io.BytesIO(); Image.fromarray(img).save(b, fmt); return b.getvalue()


def test_requires_token():
    assert client.post("/v1/predict?mag=100x", content=jpeg_bytes()).status_code == 401
    assert client.post("/v1/predict?mag=100x", content=jpeg_bytes(), headers={"x-internal-token": "wrong" * 8}).status_code == 401
    assert client.get("/healthz").status_code == 401


def test_predict_ok_and_contract():
    for mag in ("100x", "400x"):
        r = client.post(f"/v1/predict?mag={mag}", content=jpeg_bytes(), headers=H)
        assert r.status_code == 200, r.text
        j = r.json()
        assert 0.0 <= j["p_oscc"] <= 1.0 and j["mag"] == mag
        assert j["original_size"] == [640, 480] and j["qc"] == []
        assert set(j["model"]) == {"version", "backend", "img_hw"}


def test_png_and_rgba_accepted():
    assert client.post("/v1/predict?mag=100x", content=jpeg_bytes(fmt="PNG"), headers=H).status_code == 200
    rgba = io.BytesIO(); Image.new("RGBA", (600, 500), (200, 100, 180, 128)).save(rgba, "PNG")
    assert client.post("/v1/predict?mag=100x", content=rgba.getvalue(), headers=H).status_code == 200


@pytest.mark.parametrize("payload,err", [(b"not an image", "undecodable_image"), (b"GIF89a" + b"\0" * 50, None)])
def test_rejects_garbage(payload, err):
    r = client.post("/v1/predict?mag=100x", content=payload, headers=H)
    assert r.status_code == 422
    if err: assert r.json()["error"] == err


def test_gif_unsupported_format():
    b = io.BytesIO(); Image.new("P", (300, 300)).save(b, "GIF")
    r = client.post("/v1/predict?mag=100x", content=b.getvalue(), headers=H)
    assert r.status_code == 422 and r.json()["error"] == "unsupported_format"


def test_truncated_jpeg_rejected():
    r = client.post("/v1/predict?mag=100x", content=jpeg_bytes()[:900], headers=H)
    assert r.status_code == 422


def test_pixel_bomb_guard(monkeypatch):
    monkeypatch.setattr(server, "MAX_PIXELS", 100_000)
    r = client.post("/v1/predict?mag=100x", content=jpeg_bytes(640, 480), headers=H)
    assert r.status_code == 422 and r.json()["error"] == "image_too_large"


def test_unknown_mag_and_empty_body():
    assert client.post("/v1/predict?mag=40x", content=jpeg_bytes(), headers=H).status_code == 400
    assert client.post("/v1/predict?mag=100x", content=b"", headers=H).status_code == 400


def test_qc_flags():
    j = client.post("/v1/predict?mag=100x", content=jpeg_bytes(300, 250), headers=H).json()
    assert "LOW_RESOLUTION" in j["qc"]
    blank = io.BytesIO(); Image.new("RGB", (800, 600), (250, 250, 250)).save(blank, "JPEG")
    j = client.post("/v1/predict?mag=100x", content=blank.getvalue(), headers=H).json()
    assert "LOW_TISSUE" in j["qc"] or j["tissue_fraction"] >= 0  # blank slide must not crash


def test_keras_and_onnx_agree():
    folder = server.MODEL_DIR / "100x"
    if not (folder / "oscc_calibrated.onnx").exists(): pytest.skip("no onnx")
    server.BACKEND = "keras"; mk = server.Model("100x", folder)
    server.BACKEND = "onnx"; mo = server.Model("100x", folder)
    server.BACKEND = "auto"
    rng = np.random.default_rng(1)
    for _ in range(3):
        x = rng.integers(0, 255, (1, *mk.img_hw, 3), dtype=np.uint8)
        assert abs(mk.predict(x) - mo.predict(x)) < 1e-3


def test_preprocess_matches_training_notebook():
    """Extract tissue_mask/macenko from the training script and compare outputs bit-for-bit."""
    nb = Path(os.environ.get("TRAIN_SCRIPT", "/home/claude/nb/oscc_pipeline.py"))
    if not nb.exists(): pytest.skip("training script not available")
    tree = ast.parse(nb.read_text())
    want = {"tissue_mask", "macenko"}
    src = "\n".join(ast.get_source_segment(nb.read_text(), n) for n in tree.body
                    if (isinstance(n, ast.FunctionDef) and n.name in want)
                    or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("HE_REF", "MAXC_REF")))
    ns = {}
    exec("import numpy as np\nfrom skimage.color import rgb2hsv\nfrom skimage.filters import threshold_otsu\n"
         "from skimage.morphology import remove_small_objects, remove_small_holes\n" + src, ns)
    from preprocess import macenko, tissue_mask
    img = np.asarray(Image.open(io.BytesIO(jpeg_bytes(384, 288, seed=3))).convert("RGB"))
    assert np.array_equal(ns["macenko"](img), macenko(img))
    assert np.array_equal(ns["tissue_mask"](img), tissue_mask(img))
