"""Tests for the XAI toolkit on models with KNOWN behaviour, plus the real exported models where the claim is exact.

Run from the inference/ folder:  python -m pytest tests/test_xai.py -q
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("INTERNAL_TOKEN", "t" * 32)
ROOT = Path(__file__).resolve().parents[2]

from xai.adapter import Head, ModelAdapter, OnnxAdapter  # noqa: E402
from xai.alignment import alignment_metrics, annotation_ablation, cell_labels, interrater  # noqa: E402
from xai.explainers import BaselineBank, Attribution, explain_cam, explain_eg, explain_occlusion, run_control, to_cells, upsample  # noqa: E402
from xai.faithfulness import faithfulness_image  # noqa: E402
from xai.imputers import ConstantImputer, make_imputers  # noqa: E402
from xai.serve import explain_for_ui  # noqa: E402
from xai.stability import default_perturbations, map_agreement  # noqa: E402
from xai.stats import cluster_bootstrap_ci, holm  # noqa: E402

H, W = 288, 384


# ------------------------------------------------------------------ toy models with known ground truth
class RegionModel(ModelAdapter):
    """logit = b + mean_ij F_ij, F_ij = (200 - mean pixel of cell ij)/10 ONLY for cells in the 'tumour' block (rows<4, cols<6).
    Linear in pixels, exactly decomposable, and the truth about where the evidence is is known."""
    has_cam, has_grad = True, True
    img_hw = (H, W)

    def __init__(self, b=-1.0):
        self.head = Head(np.ones(1), b, 1.0)
        self.b = b

    def _F(self, x):
        c = x.astype(np.float64).reshape(len(x), 9, 32, 12, 32, 3).mean((2, 4, 5))
        f = (200 - c) / 10
        f[:, 4:, :] = 0
        f[:, :, 6:] = 0
        return f[..., None]

    def features(self, x):
        F = self._F(x)
        return self.b + F.mean((1, 2, 3)), F

    def logit(self, x):
        return self.features(x)[0]

    def grad_logit(self, x):
        g = np.zeros(x.shape, np.float32)
        g[:, : 4 * 32, : 6 * 32, :] = -1.0 / (10 * 32 * 32 * 3 * 108)
        return self.logit(x), g


class EdgeTrapModel(ModelAdapter):
    """A model that is fooled by the masking procedure: its logit falls with the number of exactly-glass-coloured pixels
    that touch non-glass pixels (i.e. the border of a flat-painted hole). It knows nothing about tissue."""
    img_hw = (H, W)
    GLASS = np.array([240, 240, 240], np.uint8)

    def logit(self, x):
        out = []
        for im in x:
            g = (im == self.GLASS).all(-1)
            n = np.zeros_like(g)
            n[1:] |= ~g[:-1]; n[:-1] |= ~g[1:]; n[:, 1:] |= ~g[:, :-1]; n[:, :-1] |= ~g[:, 1:]
            out.append(3.0 - 60.0 * float((g & n).mean()))
        return np.array(out)


class TextureModel(RegionModel):
    """Evidence = local texture (std of grey level) inside the tumour block, like nuclear chromatin: smooth interpolation
    cannot recreate it, so every sensible way of removing the block must lower the score."""

    def _F(self, x):
        g = x.astype(np.float64).mean(-1).reshape(len(x), 9, 32, 12, 32)
        f = g.std(axis=(2, 4)) / 10
        f[:, 4:, :] = 0
        f[:, :, 6:] = 0
        return f[..., None]


def textured_image(seed=0):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal((150, 90, 160), 5, (H, W, 3)), 0, 255).astype(np.uint8)
    img[: 4 * 32, : 6 * 32] = np.clip(rng.normal((150, 90, 160), 45, (4 * 32, 6 * 32, 3)), 0, 255).astype(np.uint8)
    return img


def tissue_image(seed=0, tumour=True):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal((150, 90, 160), 12, (H, W, 3)), 0, 255).astype(np.uint8)
    if tumour:
        img[: 4 * 32, : 6 * 32] = np.clip(rng.normal((60, 40, 110), 10, (4 * 32, 6 * 32, 3)), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 1.5)


# ------------------------------------------------------------------ imputers
@pytest.mark.parametrize("name", ["constant", "blur", "shuffle", "road"])
def test_imputers_leave_the_outside_untouched_and_are_deterministic(name):
    img = tissue_image(1)
    mask = np.zeros((H, W), bool); mask[50:150, 80:200] = True
    imp = make_imputers((name,))[name]
    a, b = imp(img, mask, np.random.default_rng(5)), imp(img, mask, np.random.default_rng(5))
    assert (a[~mask] == img[~mask]).all() and (a == b).all() and a.dtype == np.uint8


def test_road_has_no_visible_edge_but_constant_does():
    img = tissue_image(2, tumour=False)
    mask = np.zeros((H, W), bool); mask[64:192, 96:288] = True
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    def edge(o):
        g = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY).astype(float)
        return np.hypot(*np.gradient(g))[ring].mean()
    road = edge(make_imputers(("road",))["road"](img, mask, np.random.default_rng(0)))
    const = edge(ConstantImputer((240, 240, 240))(img, mask))
    assert road < 0.25 * const, (road, const)


# ------------------------------------------------------------------ exactness and completeness
def test_cam_is_an_exact_decomposition_of_the_logit_on_a_known_model():
    m, img = RegionModel(), tissue_image(3)
    at = explain_cam(m, img)
    assert at.meta["completeness_abs_err"] < 1e-9
    c = at.map
    assert c[:4, :6].mean() > 10 * abs(c[4:, :].mean() + 1e-12) and np.allclose(c[4:, :], 0) and np.allclose(c[:, 6:], 0)


@pytest.mark.parametrize("mag", ["100x", "400x"])
def test_cam_is_exact_on_the_real_exported_models(mag):
    d = ROOT / "models" / mag
    if not (d / "oscc_calibrated.onnx").exists():
        pytest.skip("no model")
    a = OnnxAdapter(d / "oscc_calibrated.onnx", (H, W))
    if not a.has_cam:
        pytest.skip(f"onnx package missing or head not decomposable: {a.cam_error}")
    rng = np.random.default_rng(0)
    for k in range(3):
        img = np.clip(rng.normal(180, 50, (H, W, 3)), 0, 255).astype(np.uint8)
        at = explain_cam(a, img)
        assert at.meta["completeness_abs_err"] < 1e-4, at.meta
    # the explained logit is the model's own output
    p = a.prob(img[None])[0]
    assert abs(p - 1 / (1 + np.exp(-at.logit))) < 1e-5


def test_expected_gradients_completeness_and_localisation_on_a_known_model():
    m, img = RegionModel(), tissue_image(4)
    bank = BaselineBank(np.stack([tissue_image(100 + i, tumour=False) for i in range(8)]))
    at = explain_eg(m, img, bank, n_baselines=4, n_steps=8, seed=0)
    assert at.meta["completeness_rel_err"] < 1e-3 and at.meta["converged"], at.meta   # exact for a linear model
    cells = to_cells(at.map, (H, W))
    assert cells[:8, :12].sum() > 0.95 * cells.sum()


# ------------------------------------------------------------------ faithfulness: the harness must expose a masking artefact
def _scatter_map():
    r, c = np.mgrid[:18, :24]
    return ((r + c) % 2 == 0).astype(np.float32)


def _cluster_map():
    m = np.zeros((18, 24), np.float32); m[:9, :] = 1
    return m


def test_harness_exposes_an_explanation_that_only_looks_faithful_under_one_imputer():
    model = EdgeTrapModel()
    imputers = {"constant": ConstantImputer((240, 240, 240)), **make_imputers(("blur", "shuffle", "road"))}
    rows = []
    for sd in range(2):
        img = tissue_image(sd, tumour=False)
        attrs = {"scatter": Attribution(_scatter_map(), "scatter", 3.0), "cluster": Attribution(_cluster_map(), "cluster", 3.0)}
        rows += faithfulness_image(model, img, attrs, imputers, fracs=(0.1, 0.25, 0.5), n_random=2, seed=sd)
    g = {(r["explainer"], r["imputer"]): [] for r in rows}
    for r in rows:
        g[(r["explainer"], r["imputer"])].append(r["gain"])
    gain = {k: float(np.mean(v)) for k, v in g.items()}
    assert gain[("scatter", "constant")] > 0.5            # 'faithful' only because the hole boundary fools the model
    for imp in ("blur", "shuffle", "road"):
        assert abs(gain[("scatter", imp)]) < 0.15, (imp, gain)   # evaporates when the hole is not a flat patch
    shock = {r["imputer"]: r["shock"] for r in rows}
    assert shock["constant"] > 5 * max(shock["road"], shock["blur"], shock["shuffle"], 1e-3)   # artefact index flags the imputer


def test_true_evidence_is_faithful_under_every_imputer_and_beats_controls():
    model, img = RegionModel(), tissue_image(7)
    cam = explain_cam(model, img)
    ctrl = {"ctrl_random": run_control("ctrl_random", model, img, 0), "ctrl_tissue": run_control("ctrl_tissue", model, img)}
    imputers = make_imputers(("constant", "blur", "shuffle", "road"))
    rows = faithfulness_image(model, img, {"cam": cam, **ctrl}, imputers, fracs=(0.05, 0.1, 0.2), n_random=3, seed=1)
    for r in rows:
        if r["explainer"] == "cam":
            assert r["gain"] > 0 and r["gap"] > 0, r
    best_ctrl = max(r["gain"] for r in rows if r["explainer"].startswith("ctrl_") and r["imputer"] == "road")
    assert next(r["gain"] for r in rows if r["explainer"] == "cam" and r["imputer"] == "road") > best_ctrl


def test_model_free_controls_cannot_be_faithful_to_a_localised_model():
    model, img = RegionModel(), tissue_image(8)
    rows = faithfulness_image(model, img, {"ctrl_random": run_control("ctrl_random", model, img, 3)},
                              make_imputers(("road",)), fracs=(0.1, 0.2, 0.3), n_random=4, seed=2)
    assert abs(rows[0]["gain"]) < 1.0


# ------------------------------------------------------------------ stability
def test_map_agreement_basics_and_symmetry_inverses():
    a = np.random.default_rng(0).standard_normal((9, 12)).astype(np.float32)
    assert map_agreement(a, a, (H, W))["spearman"] == pytest.approx(1.0)
    assert map_agreement(a, -a, (H, W))["spearman"] == pytest.approx(-1.0)
    up = upsample(a, (H, W))
    for p in default_perturbations():
        if p.kind == "symmetry":
            # flipping the image flips an equivariant map; applying the inverse must restore it exactly
            flipped = p.apply(np.dstack([up] * 3).astype(np.float32), None)[..., 0]
            assert np.allclose(p.inverse(flipped), up)


def test_cam_is_equivariant_under_flips_on_a_symmetric_model():
    class Sym(RegionModel):
        def _F(self, x):                               # evidence on BOTH left and right edge: symmetric under hflip
            c = x.astype(np.float64).reshape(len(x), 9, 32, 12, 32, 3).mean((2, 4, 5))
            f = (200 - c) / 10
            f[:, :, 3:9] = 0
            return f[..., None]
    m, img = Sym(), tissue_image(9)
    img[:, :64] = 40; img[:, -64:] = 40
    a, b = explain_cam(m, img).map, explain_cam(m, img[:, ::-1].copy()).map
    assert np.allclose(a, b[:, ::-1], atol=1e-6)


# ------------------------------------------------------------------ alignment
def test_alignment_perfect_random_and_degenerate():
    mask = np.zeros((H, W), bool); mask[64:192, 96:288] = True
    good = np.zeros((18, 24), np.float32); good[4:12, 6:18] = 1
    r = alignment_metrics(good, mask, (H, W))
    assert r["auroc"] == 1.0 and r["rra"] == 1.0 and r["rma_excess"] > 0.5
    rnd = [alignment_metrics(np.random.default_rng(s).standard_normal((18, 24)), mask, (H, W))["auroc"] for s in range(40)]
    assert abs(np.mean(rnd) - 0.5) < 0.06
    assert alignment_metrics(good, np.zeros((H, W), bool), (H, W)) == {}
    assert interrater(mask, mask)["dice"] == 1.0


def test_annotation_ablation_detects_that_the_model_uses_the_marked_region():
    m, img = RegionModel(), tissue_image(10)
    mask = np.zeros((H, W), bool); mask[: 4 * 32, : 6 * 32] = True
    tissue = np.ones((H, W), bool)
    rows = annotation_ablation(m, img, mask, make_imputers(("road", "blur")), tissue, seed=0)
    assert rows and all(r["necessity_gain"] > 0.1 and abs(r["remove_control"]) < 0.1 for r in rows)   # equal-area tissue elsewhere does nothing
    assert next(r for r in rows if r["imputer"] == "road")["necessity_gain"] > 0.5                    # weaker imputers (blur) only partly remove it


# ------------------------------------------------------------------ statistics
def test_cluster_bootstrap_uses_patients_not_images():
    g = np.repeat(np.arange(8), 10)
    v = np.repeat(np.random.default_rng(0).normal(0, 1, 8), 10)         # 80 images but only 8 independent values
    est, lo, hi, n = cluster_bootstrap_ci(v, g, n_boot=1000)
    assert n == 8 and (hi - lo) > 0.5
    assert np.allclose(holm([0.01, 0.02, 0.5]), [0.03, 0.04, 0.5])


# ------------------------------------------------------------------ serving contract (what gateway/inference.go validates)
def test_ui_payload_matches_the_gateway_schema_and_is_deterministic():
    m, img = TextureModel(b=0.5), textured_image(11)
    a, b = explain_for_ui(m, img, seed=3, min_gain=0.2), explain_for_ui(m, img, seed=3, min_gain=0.2)
    assert a == b
    rows, cols = a["grid"]
    assert (rows, cols) == (9, 12) and len(a["relative"]) == rows and all(len(r) == cols for r in a["relative"])
    c = a["check"]
    assert 1 <= c["cells_hidden"] <= rows * cols and c["direction"] in ("oscc", "normal") and 1 <= a["passes"] <= 1000
    for v in (a["p_oscc"], c["p_before"], c["p_top_hidden"], c["p_random_hidden"]):
        assert 0 <= v <= 1
    assert a["completeness"]["rel_err"] < 1e-6 and a["method"] == "cam_exact"
    assert c["supported"] and len(c["by_imputer"]) == 2            # real evidence passes under BOTH masking procedures


def test_ui_check_fails_for_a_model_that_ignores_location():
    class Flat(TextureModel):
        def _F(self, x):                                # same contribution from every cell -> no region is special
            c = x.astype(np.float64).mean((1, 2, 3))
            return np.broadcast_to(((200 - c) / 10)[:, None, None, None], (len(x), 9, 12, 1)).copy()
    r = explain_for_ui(Flat(), textured_image(12), min_gain=0.2)
    assert not r["check"]["supported"]
