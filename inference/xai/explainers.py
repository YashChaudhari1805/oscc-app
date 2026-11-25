"""Explainers. Convention everywhere: a map is in LOGIT units, positive = evidence FOR OSCC (raises the logit).

  cam         exact decomposition of the logit over the last feature map. Faithful by construction (completeness holds
              to float precision, verified at load), deterministic, no masking, no baseline, 1 forward pass.
              Coarse (9x12 here) and each cell has a large receptive field: so it is exact about the DECISION,
              approximate about the PIXELS. Both claims are measured in the evaluation, not assumed.
  eg          Expected Gradients / Integrated Gradients averaged over a distribution of REAL baselines (normal tissue).
              Pixel resolution; satisfies completeness in expectation (gap reported); needs gradients (Keras backend).
  occlusion   the original method, kept as a comparison and now parameterised by imputer and using true logits.
  controls    model-free maps (random, tissue/hematoxylin density, texture, distance-from-glass). A faithfulness or
              alignment score that a control also achieves is not evidence about the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.color import rgb2hed

from .imputers import Imputer, glass_colour


@dataclass
class Attribution:
    map: np.ndarray                    # [h,w] native resolution, logit units, + supports OSCC
    name: str
    logit: float                       # model logit of the explained image
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- geometry helpers
def upsample(m: np.ndarray, hw) -> np.ndarray:
    H, W = hw
    return m.astype(np.float32) if m.shape == (H, W) else cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)


def to_cells(m: np.ndarray, hw, cell=16) -> np.ndarray:
    """Common evaluation resolution: area-pool a map onto a grid of cell x cell pixel tiles."""
    H, W = hw
    full = upsample(m, hw)
    return cv2.resize(full, (W // cell, H // cell), interpolation=cv2.INTER_AREA)


def cell_mask(cells_bool: np.ndarray, hw, cell=16) -> np.ndarray:
    H, W = hw
    return np.kron(cells_bool.astype(np.uint8), np.ones((cell, cell), np.uint8)).astype(bool)[:H, :W]


# ----------------------------------------------------------------------------- CAM (exact)
def explain_cam(adapter, img):
    if not adapter.has_cam:
        raise RuntimeError("CAM needs a verified head decomposition")
    lg, F = adapter.features(img[None])
    h = adapter.head
    contrib = h.scale * (F[0].astype(np.float64) @ h.w) / (F.shape[1] * F.shape[2])   # [h,w]; sums to logit - scale*b
    bias = h.scale * h.b
    gap = float(abs(contrib.sum() + bias - lg[0]))
    return Attribution(contrib.astype(np.float32), "cam", float(lg[0]),
                       {"bias": bias, "completeness_abs_err": gap, "completeness_rel_err": gap / max(abs(lg[0]), 1e-6)})


# ----------------------------------------------------------------------------- Expected Gradients
class BaselineBank:
    """Reference images that mean 'no OSCC evidence'. Best source: preprocessed NORMAL-class TRAINING images.
    Fallback (no data available): glass / heavy-blur / noise, which are less in-distribution and say so in the metadata."""

    def __init__(self, images: np.ndarray | None = None):
        self.images = images           # uint8 [M,H,W,3] or None
        self.kind = "normal_tissue_bank" if images is not None else "synthetic_fallback"

    @classmethod
    def from_npz(cls, path):
        return cls(np.load(path)["images"])

    @classmethod
    def from_folder(cls, folder, preprocess, hw, n=64, seed=0):
        import glob, os
        paths = sorted(p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True) if p.lower().endswith((".jpg", ".jpeg", ".png")))
        rng = np.random.default_rng(seed)
        rng.shuffle(paths)
        ims = []
        for p in paths[:n]:
            arr = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            ims.append(preprocess(arr, hw))
        if not ims:
            raise FileNotFoundError(f"no images under {folder}")
        return cls(np.stack(ims))

    def sample(self, img, rng):
        if self.images is not None:
            return self.images[rng.integers(len(self.images))].astype(np.float32)
        k = rng.integers(3)
        if k == 0:
            return np.broadcast_to(glass_colour(img).astype(np.float32), img.shape).copy()
        if k == 1:
            return cv2.GaussianBlur(img, (0, 0), 32).astype(np.float32)
        return np.clip(rng.normal(img.reshape(-1, 3).mean(0), 25.0, img.shape), 0, 255).astype(np.float32)


def explain_eg(adapter, img, bank: BaselineBank, n_baselines=4, n_steps=12, power=3, seed=0, batch=1, converge_tol=0.10):
    """phi_i = E_{b~bank} int_0^1 (x_i - b_i) * d logit/d x_i (b + a (x - b)) da, channels summed.

    Along each path the integral uses Gauss-Legendre nodes on u with a = u**power: on this network the logit changes
    violently in the first ~20% of the path (blank/normal -> tissue), where uniform steps are badly wrong
    (measured: errors of 0.1-2.5 nats between 6 and 24 uniform steps). Completeness is NOT assumed: the gap between
    sum(phi) and logit(x) - E logit(b) is returned, with converged = gap <= converge_tol * |target|. Report it."""
    if not adapter.has_grad:
        raise RuntimeError("Expected Gradients needs the Keras backend (true gradients)")
    rng = np.random.default_rng(seed)
    x = img.astype(np.float32)
    u, w = np.polynomial.legendre.leggauss(n_steps)
    u, w = (u + 1) / 2, w / 2
    alphas, weights = u ** power, w * power * u ** (power - 1)
    acc = np.zeros(img.shape, np.float64)
    base_logits = []
    for _ in range(n_baselines):
        b = bank.sample(img, rng)
        for i in range(0, n_steps, batch):
            pts = np.stack([b + a * (x - b) for a in alphas[i:i + batch]])
            _, g = adapter.grad_logit(pts)
            for gi, wt in zip(g, weights[i:i + batch]):
                acc += wt * (x - b) * gi
        base_logits.append(float(adapter.logit(np.clip(b, 0, 255).astype(np.uint8)[None])[0]))
    phi = acc / n_baselines
    lg = float(adapter.logit(img[None])[0])
    target = lg - float(np.mean(base_logits))
    gap = abs(phi.sum() - target)
    rel = float(gap / max(abs(target), 1e-6))
    return Attribution(phi.sum(-1).astype(np.float32), "eg", lg,
                       {"baseline": bank.kind, "n_baselines": n_baselines, "n_steps": n_steps, "completeness_target": target,
                        "completeness_abs_err": float(gap), "completeness_rel_err": rel, "converged": bool(rel <= converge_tol), "seed": seed})


# ----------------------------------------------------------------------------- Occlusion (comparison / legacy)
def _positions(n, win, stride):
    stride = max(1, min(stride, win))
    pos = list(range(0, n - win + 1, stride))
    if pos[-1] != n - win:
        pos.append(n - win)
    return pos


def explain_occlusion(adapter, img, imputer: Imputer, rows=9, cols=12, win=3, stride=2, seed=0, batch=8):
    H, W = img.shape[:2]
    ys, xs = np.linspace(0, H, rows + 1).astype(int), np.linspace(0, W, cols + 1).astype(int)
    rng = np.random.default_rng(seed)
    l0 = float(adapter.logit(img[None])[0])
    wins = [(r, c) for r in _positions(rows, win, stride) for c in _positions(cols, win, stride)]
    batch_imgs = []
    for r, c in wins:
        m = np.zeros((H, W), bool)
        m[ys[r]:ys[r + win], xs[c]:xs[c + win]] = True
        batch_imgs.append(imputer(img, m, rng))
    drop = l0 - adapter.logit(np.stack(batch_imgs))          # >0: removing it LOWERS the OSCC logit => supports OSCC
    tot, cnt = np.zeros((rows, cols)), np.zeros((rows, cols))
    for (r, c), d in zip(wins, drop):
        tot[r:r + win, c:c + win] += d
        cnt[r:r + win, c:c + win] += 1
    return Attribution((tot / np.maximum(cnt, 1)).astype(np.float32), f"occlusion[{imputer.name}]", l0,
                       {"imputer": imputer.name, "passes": len(wins) + 1})


# ----------------------------------------------------------------------------- model-free controls
def _tissue(img):
    mx, mn = img.max(-1).astype(np.float32), img.min(-1).astype(np.float32)
    return ((mx - mn) / np.maximum(mx, 1) > 0.12).astype(np.float32)


def control_random(img, seed=0):
    return np.random.default_rng(seed).standard_normal((img.shape[0] // 16, img.shape[1] // 16)).astype(np.float32)


def control_tissue(img, seed=0):
    return cv2.GaussianBlur(_tissue(img), (0, 0), 8)


def control_hematoxylin(img, seed=0):
    with np.errstate(all="ignore"):
        h = rgb2hed(img)[..., 0]
    return cv2.GaussianBlur(np.nan_to_num(h).astype(np.float32), (0, 0), 8)


def control_texture(img, seed=0):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.GaussianBlur(g, (0, 0), 6)
    return np.sqrt(np.maximum(cv2.GaussianBlur(g * g, (0, 0), 6) - mu * mu, 0))


def control_glass_distance(img, seed=0):
    d = np.linalg.norm(img.astype(np.float32) - glass_colour(img).astype(np.float32), axis=-1)
    return cv2.GaussianBlur(d, (0, 0), 8)


CONTROLS = {"ctrl_random": control_random, "ctrl_tissue": control_tissue, "ctrl_hematoxylin": control_hematoxylin,
            "ctrl_texture": control_texture, "ctrl_glass_distance": control_glass_distance}


def run_control(name, adapter, img, seed=0):
    return Attribution(CONTROLS[name](img, seed), name, float(adapter.logit(img[None])[0]), {"model_free": True})
