"""Serving-time explanation: exact CAM decomposition + a masking self-check that must pass under TWO imputers.

Response schema is a superset of the original occlusion response, so the gateway's validateExplain and the UI keep working.
"""
from __future__ import annotations

import numpy as np

from .adapter import sigmoid
from .explainers import cell_mask, explain_cam
from .imputers import make_imputers


def _check_one(adapter, img, imp, top, rand_sets, rows, cols, l0, s, seed, min_gain, ratio):
    H, W = img.shape[:2]
    ys, xs = np.linspace(0, H, rows + 1).astype(int), np.linspace(0, W, cols + 1).astype(int)

    def hide(cells):
        m = np.zeros((H, W), bool)
        for r, c in cells:
            m[ys[r]:ys[r + 1], xs[c]:xs[c + 1]] = True
        return imp(img, m, np.random.default_rng(seed))

    lg = adapter.logit(np.stack([hide(top)] + [hide(rs) for rs in rand_sets]))
    delta = s * (l0 - lg)
    d_top, d_rand = float(delta[0]), float(delta[1:].mean())
    ok = bool(d_top >= min_gain and d_top >= ratio * max(d_rand, 0.0) + min_gain / 2 and d_top > float(delta[1:].max()))
    return {"imputer": imp.name, "delta_top": d_top, "delta_random": d_rand, "p_top_hidden": float(sigmoid(lg[0])),
            "p_random_hidden": float(sigmoid(lg[1:]).mean()), "supported": ok}


def explain_for_ui(adapter, img, *, k_frac=0.15, n_random=6, seed=0, min_gain=0.5, ratio=2.0, imputers=("road", "blur")):
    at = explain_cam(adapter, img)
    imp_t = at.map.astype(np.float64)
    rows, cols = imp_t.shape
    rel = imp_t - np.median(imp_t)
    l0 = at.logit
    s = 1 if l0 >= 0 else -1
    k = max(1, int(round(k_frac * rows * cols)))
    order = np.argsort(-(s * imp_t).ravel(), kind="stable")
    top = [(int(i // cols), int(i % cols)) for i in order[:k]]
    rng = np.random.default_rng(seed)
    rand_sets = [[(int(i // cols), int(i % cols)) for i in rng.choice(rows * cols, k, replace=False)] for _ in range(n_random)]
    checks = [_check_one(adapter, img, imp, top, rand_sets, rows, cols, l0, s, seed, min_gain, ratio) for imp in make_imputers(imputers).values()]
    worst = min(checks, key=lambda c: c["delta_top"] - c["delta_random"])        # report the conservative imputer
    p0 = float(sigmoid(l0))
    return {
        "p_oscc": round(p0, 6), "logit": round(float(l0), 4),
        "grid": [rows, cols], "window": [1, 1], "stride": 1,
        "importance": np.round(imp_t, 4).tolist(), "relative": np.round(rel, 4).tolist(),
        "max_abs": round(float(np.abs(imp_t).max()), 4), "max_rel": round(float(np.abs(rel).max()), 4),
        "bias": round(float(at.meta["bias"]), 4),
        "completeness": {"abs_err": float(at.meta["completeness_abs_err"]), "rel_err": float(at.meta["completeness_rel_err"])},
        "check": {"cells_hidden": k, "direction": "oscc" if s == 1 else "normal", "p_before": round(p0, 6),
                  "p_top_hidden": round(worst["p_top_hidden"], 6), "p_random_hidden": round(worst["p_random_hidden"], 6),
                  "delta_top": round(worst["delta_top"], 4), "delta_random": round(worst["delta_random"], 4),
                  "supported": bool(all(c["supported"] for c in checks)),
                  "by_imputer": [{k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in c.items()} for c in checks]},
        "method": "cam_exact", "passes": 1 + len(checks) * (1 + n_random),
    }
