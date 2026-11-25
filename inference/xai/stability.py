"""Stability: do explanations stay put when the input changes in ways that should not matter?

Perturbations act on the RAW image and go through the same preprocessing as serving (resize + Macenko), so we test
the system a user actually meets, not an idealised one.

  nuisance   gaussian noise, JPEG re-compression, brightness/contrast, H&E stain jitter, slight defocus
  symmetry   hflip, vflip, rot180: histology has no preferred orientation, so the explanation must be EQUIVARIANT
             (explain(flip(x)) == flip(explain(x))). This is a non-negotiable, label-free correctness test.
  shift      small translation, compared on the overlap

Reading the numbers correctly (this is what reviewers check):
  * explanation change is only blamed on the explainer when the MODEL's output stayed put: we report results
    conditional on |delta logit| <= tau, and the correlation between explanation drift and prediction drift;
  * every stability metric has a CHANCE level (map of a perturbed image vs the map of a DIFFERENT image) and a
    model-free control, so "rho = 0.8" is interpretable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from scipy.stats import spearmanr
from skimage.color import hed2rgb, rgb2hed

from .explainers import to_cells, upsample


@dataclass
class Perturbation:
    name: str
    kind: str                                   # nuisance | symmetry | shift
    apply: Callable                             # (raw, rng) -> raw'
    inverse: Callable = lambda m: m             # map explanation(raw') back to the frame of explanation(raw)
    crop: int = 0                               # model-input pixels to drop from each border before comparing


def _noise(sigma):
    return lambda raw, rng: np.clip(raw.astype(np.float32) + rng.normal(0, sigma, raw.shape), 0, 255).astype(np.uint8)


def _jpeg(q):
    def f(raw, rng):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.cvtColor(cv2.imdecode(buf, 1), cv2.COLOR_BGR2RGB)
    return f


def _bc(raw, rng):
    a, b = rng.uniform(0.94, 1.06), rng.uniform(-8, 8)
    return np.clip(raw.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


def _hed(sigma):
    def f(raw, rng):
        hed = rgb2hed(raw)
        hed = hed * rng.normal(1.0, sigma, 3) + rng.normal(0, sigma * 0.1, 3)
        return (np.clip(hed2rgb(hed), 0, 1) * 255).astype(np.uint8)
    return f


def _blur(raw, rng):
    return cv2.GaussianBlur(raw, (0, 0), 0.8)


def _shift(px_model, hw_model):
    def f(raw, rng):
        h, w = raw.shape[:2]
        dx, dy = int(round(px_model * w / hw_model[1])), int(round(px_model * h / hw_model[0]))
        return cv2.warpAffine(raw, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderMode=cv2.BORDER_REFLECT)
    return f


def _unshift(px):
    return lambda m: np.roll(np.roll(m, -px, axis=0), -px, axis=1)


def default_perturbations(hw=(288, 384), shift_px=8):
    return [
        Perturbation("noise_s3", "nuisance", _noise(3)),
        Perturbation("noise_s6", "nuisance", _noise(6)),
        Perturbation("jpeg_q70", "nuisance", _jpeg(70)),
        Perturbation("brightness_contrast", "nuisance", _bc),
        Perturbation("stain_jitter", "nuisance", _hed(0.04)),
        Perturbation("defocus", "nuisance", _blur),
        Perturbation("hflip", "symmetry", lambda r, g: r[:, ::-1].copy(), lambda m: m[:, ::-1]),
        Perturbation("vflip", "symmetry", lambda r, g: r[::-1].copy(), lambda m: m[::-1]),
        Perturbation("rot180", "symmetry", lambda r, g: r[::-1, ::-1].copy(), lambda m: m[::-1, ::-1]),
        Perturbation(f"shift{shift_px}px", "shift", _shift(shift_px, hw), _unshift(shift_px), crop=16),
    ]


def map_agreement(a, b, hw, cell=16, s=1, crop=0, k_frac=0.2):
    """a, b: native-resolution maps in the SAME frame. Returns spearman, pearson, top-k Jaccard of the predicted-class evidence."""
    A, B = s * to_cells(a, hw, cell), s * to_cells(b, hw, cell)
    if crop:
        c = crop // cell
        A, B = A[c:-c, c:-c], B[c:-c, c:-c]
    a1, b1 = A.ravel(), B.ravel()
    if a1.std() < 1e-12 or b1.std() < 1e-12:
        return {"spearman": np.nan, "pearson": np.nan, "jaccard_top20": np.nan}
    k = max(1, int(round(k_frac * a1.size)))
    ta, tb = set(np.argsort(-a1, kind="stable")[:k]), set(np.argsort(-b1, kind="stable")[:k])
    return {"spearman": float(spearmanr(a1, b1).statistic), "pearson": float(np.corrcoef(a1, b1)[0, 1]),
            "jaccard_top20": len(ta & tb) / len(ta | tb)}


def stability_rows(raw, preprocess, adapter, explainers: dict, perturbations, seed=0, n_repeats=2, tau=0.25, other_maps: dict | None = None):
    """explainers: name -> fn(img_uint8, seed) -> Attribution.  preprocess: raw -> uint8 model input.
    other_maps: name -> native map of a DIFFERENT image (chance reference).
    All maps are brought to pixel resolution BEFORE any inverse geometry, so coarse (CAM) and pixel-level (EG) methods
    are treated identically."""
    rng = np.random.default_rng(seed)
    img0 = preprocess(raw)
    hw = img0.shape[:2]
    base = {n: fn(img0, seed) for n, fn in explainers.items()}
    l0 = next(iter(base.values())).logit
    s = 1 if l0 >= 0 else -1
    rows = []
    for p in perturbations:
        for rep in range(n_repeats if p.kind == "nuisance" else 1):
            img2 = preprocess(p.apply(raw, rng))
            for n, fn in explainers.items():
                at2 = fn(img2, seed + 1000 * (rep + 1))
                back = p.inverse(upsample(at2.map, hw))
                r = {"explainer": n, "perturbation": p.name, "kind": p.kind, "rep": rep, "logit0": l0,
                     "dlogit": at2.logit - l0, "pred_stable": abs(at2.logit - l0) <= tau}
                r.update(map_agreement(base[n].map, back, hw, s=s, crop=p.crop))
                if other_maps and n in other_maps:
                    r.update({"chance_" + k: v for k, v in map_agreement(other_maps[n], back, hw, s=s, crop=p.crop).items()})
                rows.append(r)
    return rows


def seed_stability_rows(img, explainer_fn, seeds=(0, 1, 2), s=1):
    maps = [explainer_fn(img, sd) for sd in seeds]
    hw = img.shape[:2]
    out = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            out.append(map_agreement(maps[i].map, maps[j].map, hw, s=s))
    return out
