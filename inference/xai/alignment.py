"""Agreement with expert annotation, reported so that it cannot flatter the method.

Two different questions, kept apart:

  A. Does the EXPLANATION point where the pathologist marked tumour?          (alignment_metrics)
  B. Does the MODEL itself rely on the pathologist-marked region?             (annotation_ablation)

(B) needs no explainer at all. If (B) is weak the model is using something else (stain, borders, background) and no
explainer should be expected to match the annotation: low (A) is then a finding about the model, not a defect of the method.

Every (A) number is shown next to model-free controls and next to the chance level implied by the mask's own area.
A map that merely lights up stained tissue or nuclei will beat random but NOT the tissue/hematoxylin controls.
"""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .explainers import cell_mask, to_cells, upsample


def load_mask(path, hw, positive=None):
    """Binary expert mask resized (nearest) to the model-input frame. positive: iterable of label values counted as tumour (default: any > 0)."""
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m[..., 0] if m.shape[2] != 4 else m[..., 3] if m[..., 3].any() and not m[..., :3].any() else m[..., 0]
    b = np.isin(m, list(positive)) if positive is not None else m > 0
    return cv2.resize(b.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def cell_labels(mask, cell=16, thr=0.5):
    H, W = mask.shape
    return cv2.resize(mask.astype(np.float32), (W // cell, H // cell), interpolation=cv2.INTER_AREA) > thr


def alignment_metrics(attr_map, mask, hw, cell=16, min_area=0.03, max_area=0.9):
    """attr_map: native-resolution map, positive = OSCC evidence. Returns {} when the mask is degenerate (nothing to align to)."""
    area = float(mask.mean())
    if not (min_area <= area <= max_area):
        return {}
    cl = cell_labels(mask, cell).ravel()
    sc = to_cells(attr_map, hw, cell).ravel()
    if cl.all() or (~cl).all():
        return {}
    full = np.maximum(upsample(attr_map, hw), 0)
    mass_in = float(full[mask].sum() / max(full.sum(), 1e-12))
    k = int(cl.sum())
    top = np.argsort(-sc, kind="stable")[:k]
    rra = float(cl[top].mean())
    return {
        "area_frac": area,
        "auroc": float(roc_auc_score(cl, sc)),
        "ap": float(average_precision_score(cl, sc)), "ap_chance": float(cl.mean()),
        "pointing": float(cl[int(np.argmax(sc))]),
        "rma": mass_in, "rma_excess": mass_in - area,          # relevance mass accuracy; chance = area
        "rra": rra, "rra_excess": rra - float(cl.mean()),      # relevance rank accuracy (Arras et al.); chance = area
        "iou_topk": rra / (2 - rra) if rra < 2 else 1.0,
    }


def interrater(m1, m2):
    i, u = (m1 & m2).sum(), (m1 | m2).sum()
    return {"iou": float(i / u) if u else np.nan, "dice": float(2 * i / (m1.sum() + m2.sum())) if (m1.sum() + m2.sum()) else np.nan}


def _matched_controls(mask, tissue, n, rng, max_overlap=0.1, min_tissue=0.7, tries=400):
    """Shifted copies of the mask that avoid the annotation and still land on tissue: same shape, same area, different place."""
    H, W = mask.shape
    out = []
    for _ in range(tries):
        dy, dx = int(rng.integers(-H + 1, H)), int(rng.integers(-W + 1, W))
        m = np.zeros_like(mask)
        ys, xs = np.nonzero(mask)
        ys, xs = ys + dy, xs + dx
        ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        if ok.sum() < 0.95 * len(ys):
            continue
        m[ys[ok], xs[ok]] = True
        if (m & mask).sum() / m.sum() > max_overlap or (m & tissue).sum() / m.sum() < min_tissue:
            continue
        out.append(m)
        if len(out) == n:
            break
    return out


def annotation_ablation(adapter, img, mask, imputers: dict, tissue, n_controls=5, seed=0, min_area=0.03, max_area=0.7):
    """Question B. Removing the expert-marked region vs removing equal-area tissue elsewhere (and the dual 'keep only' test)."""
    area = float(mask.mean())
    if not (min_area <= area <= max_area):
        return []
    rng = np.random.default_rng(seed)
    l0 = float(adapter.logit(img[None])[0])
    s = 1 if l0 >= 0 else -1
    ctrls = _matched_controls(mask, tissue, n_controls, rng)
    if len(ctrls) < 2:
        return []
    rows = []
    for name, imp in imputers.items():
        rem = adapter.logit(np.stack([imp(img, mask, np.random.default_rng(seed))] + [imp(img, c, np.random.default_rng(seed)) for c in ctrls]))
        keep = adapter.logit(np.stack([imp(img, ~mask, np.random.default_rng(seed))] + [imp(img, ~c, np.random.default_rng(seed)) for c in ctrls]))
        rows.append({"imputer": name, "area_frac": area, "logit0": l0, "n_controls": len(ctrls),
                     "remove_expert": float(s * (l0 - rem[0])), "remove_control": float(s * (l0 - rem[1:]).mean()),
                     "necessity_gain": float(s * (l0 - rem[0]) - s * (l0 - rem[1:]).mean()),
                     "keep_expert": float(s * keep[0]), "keep_control": float(s * keep[1:].mean()),
                     "sufficiency_gain": float(s * keep[0] - s * keep[1:].mean())})
    return rows
