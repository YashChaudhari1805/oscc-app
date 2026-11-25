"""Imputers: the ways we make a region of an image "absent".

Every perturbation-based explanation or faithfulness test must answer "absent compared with what?". The
answer is the main hidden confound in XAI for CNNs: a patch painted with one flat colour is an image no
microscope ever produced, so the score can move because the network reacts to the *hole*, not because the
hidden tissue mattered. We therefore never rely on one imputer. Four with different failure modes are provided,
and every claim in the evaluation must hold across all of them (see faithfulness.py / run_eval.py).

    constant  flat blank-glass colour                    (what the first version used; colour- and edge-OOD)
    blur      heavy Gaussian blur of the same image      (stain colours kept, texture/nuclei destroyed)
    shuffle   pixels re-drawn from the image's own tissue (colour distribution kept, spatial structure destroyed)
    road      Noisy Linear Imputation (Rong et al. 2022) (smooth, boundary-consistent, has no sharp edge;
                                                          designed so the model cannot "see" the mask shape)

Contract: imputer(img uint8 [H,W,3], mask bool [H,W] (True = remove), rng) -> uint8 [H,W,3]
Pixels outside the mask are returned bit-identical.
"""
from __future__ import annotations

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg

# 3x3 neighbourhood weights from the ROAD paper (sum = 1)
_ROAD_OFFSETS = [(-1, -1, 0.1036), (-1, 0, 0.1464), (-1, 1, 0.1036), (0, -1, 0.1464),
                 (0, 1, 0.1464), (1, -1, 0.1036), (1, 0, 0.1464), (1, 1, 0.1036)]


def glass_colour(img: np.ndarray, tissue: np.ndarray | None = None) -> np.ndarray:
    """Median colour of blank glass (low-saturation pixels). Falls back to near-white."""
    if tissue is None:
        mx, mn = img.max(-1).astype(np.float32), img.min(-1).astype(np.float32)
        sat = (mx - mn) / np.maximum(mx, 1)
        glass = img[sat < 0.08]
    else:
        glass = img[~tissue]
    return np.median(glass, axis=0).astype(np.uint8) if len(glass) > 200 else np.array([240, 240, 240], np.uint8)


class Imputer:
    name = "base"

    def __call__(self, img, mask, rng=None):
        raise NotImplementedError


class ConstantImputer(Imputer):
    name = "constant"

    def __init__(self, fill=None):
        self.fill = None if fill is None else np.asarray(fill, np.uint8)

    def __call__(self, img, mask, rng=None):
        out = img.copy()
        out[mask] = self.fill if self.fill is not None else glass_colour(img)
        return out


class BlurImputer(Imputer):
    name = "blur"

    def __init__(self, sigma=24.0):
        self.sigma = sigma

    def __call__(self, img, mask, rng=None):
        blurred = cv2.GaussianBlur(img, (0, 0), self.sigma, borderType=cv2.BORDER_REFLECT)
        out = img.copy()
        out[mask] = blurred[mask]
        return out


class ShuffleImputer(Imputer):
    """Fill with pixels sampled (with replacement) from the *unmasked* part of the same image."""
    name = "shuffle"

    def __call__(self, img, mask, rng=None):
        rng = rng or np.random.default_rng(0)
        keep = img[~mask]
        out = img.copy()
        n = int(mask.sum())
        if n and len(keep):
            out[mask] = keep[rng.integers(0, len(keep), n)]
        elif n:
            out[mask] = glass_colour(img)
        return out


class NoisyLinearImputer(Imputer):
    """ROAD (Rong, Leemann, Borisov, Kasneci, 2022). Each removed pixel becomes the weighted mean of its 8
    neighbours (solved jointly over the whole hole, so large holes are filled by smooth interpolation of the
    boundary) plus small Gaussian noise. Symmetric positive-definite system -> conjugate gradient."""
    name = "road"

    def __init__(self, noise=0.01, rtol=1e-4, maxiter=600):
        self.noise, self.rtol, self.maxiter = noise, rtol, maxiter

    def __call__(self, img, mask, rng=None):
        rng = rng or np.random.default_rng(0)
        n = int(mask.sum())
        if n == 0:
            return img.copy()
        if n == mask.size:
            return np.full_like(img, glass_colour(img))
        H, W = mask.shape
        idx = -np.ones((H, W), np.int64)
        idx[mask] = np.arange(n)
        ys, xs = np.nonzero(mask)
        diag = np.zeros(n)
        rhs = np.zeros((n, 3))
        rows, cols, vals = [], [], []
        f = img.astype(np.float64)
        for dy, dx, w in _ROAD_OFFSETS:
            ny, nx = ys + dy, xs + dx
            ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            diag[ok] += w
            nyo, nxo, io = ny[ok], nx[ok], np.nonzero(ok)[0]
            inside = mask[nyo, nxo]
            rows.append(io[inside]); cols.append(idx[nyo[inside], nxo[inside]]); vals.append(np.full(inside.sum(), -w))
            known = ~inside
            rhs[io[known]] += w * f[nyo[known], nxo[known]]
        rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
        A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        M = sp.diags(1.0 / diag)
        sol = np.empty((n, 3))
        x0 = np.full(3, 0.0)
        for ch in range(3):
            x0[ch] = f[~mask][:, ch].mean() if (~mask).any() else 240.0
            sol[:, ch], _ = cg(A, rhs[:, ch], x0=np.full(n, x0[ch]), rtol=self.rtol, maxiter=self.maxiter, M=M)
        sol += rng.normal(0.0, self.noise * 255.0, sol.shape)
        out = img.copy()
        out[mask] = np.clip(np.rint(sol), 0, 255).astype(np.uint8)
        return out


REGISTRY = {"constant": ConstantImputer, "blur": BlurImputer, "shuffle": ShuffleImputer, "road": NoisyLinearImputer}


def make_imputers(names=("constant", "blur", "shuffle", "road")):
    return {n: REGISTRY[n]() for n in names}
