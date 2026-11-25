"""Sanity checks for saliency (Adebayo et al., 2018), adapted to this model.

Cascading model-parameter randomisation: re-initialise the network from the top (classifier head) downward, one stage at
a time, and recompute the explanation. An explanation that is "about the model" must DECORRELATE from the original as
the model is destroyed. If it does not, the method is describing the image (edges, stain, tissue density), not the model.
Needs the Keras backend (weights must be writable).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .explainers import to_cells

STAGES = [("head", ["bin", "grade"]), ("top", ["top_conv"]), ("block7", ["block7"]), ("block6", ["block6"]), ("block5", ["block5"]),
          ("block4", ["block4"]), ("block3", ["block3"]), ("block2", ["block2"]), ("block1", ["block1"]), ("stem", ["stem"])]


def cascading_randomisation(adapter, images, explain_fn, stages=STAGES, seed=0, cell=16):
    """explain_fn(adapter, img) -> Attribution. Returns rows: stage, image index, spearman(original, randomised), logit shift."""
    hw = images[0].shape[:2]
    base = [explain_fn(adapter, im) for im in images]
    rows, prefixes = [], []
    for si, (label, pre) in enumerate(stages):
        prefixes += pre
        rnd = adapter.clone_with_randomised(prefixes, seed=seed + si)
        for i, im in enumerate(images):
            at = explain_fn(rnd, im)
            a, b = to_cells(base[i].map, hw, cell).ravel(), to_cells(at.map, hw, cell).ravel()
            rho = float(spearmanr(a, b).statistic) if a.std() > 1e-12 and b.std() > 1e-12 else float("nan")
            rows.append({"stage": label, "stage_idx": si, "image": i, "spearman": rho, "abs_spearman": abs(rho),
                         "dlogit": float(at.logit - base[i].logit)})
    return rows
