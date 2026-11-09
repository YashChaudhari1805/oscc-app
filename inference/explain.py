"""Occlusion-sensitivity explanation.

How it works (and why this method): the model is a black box that returns a score. We hide one small
patch of the image at a time (paint it with blank glass colour), re-score, and record how much the
log-odds of OSCC changed. A patch whose removal LOWERS the score supports OSCC; one whose removal RAISES
it argues against. Every number is a measured response of the real model: nothing is inferred from
its internals, so it also works on the exported ONNX file.

We deliberately do NOT use Grad-CAM: in our own masking test, Grad-CAM-ranked regions mattered no more
than random ones for this model.

Limits (shown to the user): patches overlap and the model can use context, so occlusion can miss
evidence that is spread out or redundant, and it says what the model is sensitive to, not what is cancer.
A self-check hides the top-ranked patches vs. the same number of random patches and reports whether the map
actually predicted what matters.
"""
import numpy as np

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _positions(n, win, stride):
    stride = max(1, min(stride, win))   # a stride above the window size would leave unexamined stripes
    pos = list(range(0, n - win + 1, stride))
    if pos[-1] != n - win:        # always cover the last row/column
        pos.append(n - win)
    return pos


def explain_occlusion(predict_batch, img, *, rows=9, cols=12, win=3, stride=2, fill=(240, 240, 240),
                      k_frac=0.15, n_random=4, seed=0, batch=8, min_gain=0.5, ratio=2.0):
    """predict_batch(uint8 [N,H,W,3]) -> P(OSCC) array [N]. img: uint8 [H,W,3] (already preprocessed)."""
    if not (1 <= win <= min(rows, cols)):
        raise ValueError("window must fit inside the grid")
    H, W = img.shape[:2]
    ys = np.linspace(0, H, rows + 1).astype(int)
    xs = np.linspace(0, W, cols + 1).astype(int)
    fill = np.asarray(fill, dtype=np.uint8)

    def occlude(cells):
        out = img.copy()
        for r, c in cells:
            out[ys[r]:ys[r + 1], xs[c]:xs[c + 1]] = fill
        return out

    def score(images):
        ps = []
        for i in range(0, len(images), batch):
            ps.append(np.asarray(predict_batch(np.stack(images[i:i + batch])), dtype=np.float64).reshape(-1))
        return np.concatenate(ps)

    p0 = float(score([img])[0])
    l0 = float(logit(p0))

    windows = [(r, c) for r in _positions(rows, win, stride) for c in _positions(cols, win, stride)]
    pw = score([occlude([(r + dr, c + dc) for dr in range(win) for dc in range(win)]) for r, c in windows])
    drop = l0 - logit(pw)                       # >0: hiding this window lowered the OSCC log-odds
    total = np.zeros((rows, cols)); count = np.zeros((rows, cols))
    for (r, c), d in zip(windows, drop):
        total[r:r + win, c:c + win] += d
        count[r:r + win, c:c + win] += 1
    importance = total / np.maximum(count, 1)   # per-cell mean over every window that covers it
    # Blank glass is itself unusual input, so hiding ANY patch shifts the score a little. 'relative' removes that
    # common shift: how much more (+) or less (-) than a typical patch this one matters.
    relative = importance - np.median(importance)

    # ---- self-check: does hiding the "most important" cells move the score more than hiding random ones? ----
    s = 1 if p0 >= 0.5 else -1                  # which class the model chose; "important" = supports that class
    k = max(1, int(round(k_frac * rows * cols)))
    order = np.argsort(-(s * importance).ravel(), kind="stable")
    top = [(int(i // cols), int(i % cols)) for i in order[:k]]
    rng = np.random.default_rng(seed)
    rand_sets = [[(int(i // cols), int(i % cols)) for i in rng.choice(rows * cols, k, replace=False)] for _ in range(n_random)]
    pc = score([occlude(top)] + [occlude(rs) for rs in rand_sets])
    delta = s * (l0 - logit(pc))                # >0: moved toward the other class
    d_top, d_rand = float(delta[0]), float(np.mean(delta[1:])) if n_random else 0.0
    supported = bool(d_top >= min_gain and d_top >= ratio * max(d_rand, 0.0) + min_gain / 2)

    return {
        "p_oscc": round(p0, 6),
        "grid": [rows, cols], "window": [win, win], "stride": stride,
        "importance": np.round(importance, 4).tolist(),   # nats of log-odds; + supports OSCC, - argues against
        "relative": np.round(relative, 4).tolist(),       # importance minus the median patch (what the UI draws)
        "max_abs": round(float(np.abs(importance).max()), 4),
        "max_rel": round(float(np.abs(relative).max()), 4),
        "check": {
            "cells_hidden": k, "direction": "oscc" if s == 1 else "normal",
            "p_before": round(p0, 6),
            "p_top_hidden": round(float(pc[0]), 6),
            "p_random_hidden": round(float(np.mean(pc[1:])) if n_random else p0, 6),
            "delta_top": round(d_top, 4), "delta_random": round(d_rand, 4),
            "supported": supported,
        },
        "method": "occlusion", "passes": 1 + len(windows) + 1 + n_random,
    }
