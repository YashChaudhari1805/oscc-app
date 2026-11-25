"""Faithfulness: does removing what the explanation calls important change the model's output more than removing
unimportant (or random) content, *under every reasonable definition of "removed"*?

Per image, per explainer, per imputer we compute the removal curves (cells ranked by the explanation, at the shared
evaluation resolution so that coarse and pixel-level methods are compared on equal footing):

  MoRF  remove the most supportive cells first          -> logit falls toward the other class
  LeRF  remove the least supportive cells first         -> logit should barely move
  RAND  remove cells in random order (R draws, shared by all explainers)

  aopc            mean over removal fractions of  s*(l0 - l(f)),  s = +1 if the model said OSCC else -1   [nats]
  gain            aopc_morf - aopc_rand       > 0 means better than chance
  gap             aopc_morf - aopc_lerf       Samek-style ordering gap
  shock           mean |l_rand(f) - l0|       how hard the *imputer itself* shocks the model (artefact index)

Logits, not probabilities: probabilities saturate for a confident model and flatten every curve.
The claim "not an artefact of masking" is made only when `gain` is positive and consistent in rank across imputers,
and when the model-free controls do NOT achieve the same gain (see summarise_faithfulness in report.py).
"""
from __future__ import annotations

import numpy as np

from .explainers import cell_mask, to_cells

DEFAULT_FRACS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def direction(l0: float) -> int:
    return 1 if l0 >= 0 else -1


def rank_cells(attr_map, hw, cell, s):
    """Cell order, most evidence FOR the predicted class first. Stable tie-break for determinism."""
    cells = s * to_cells(attr_map, hw, cell)
    return np.argsort(-cells.ravel(), kind="stable"), cells.shape


def _curve(adapter, img, order, shape, imputer, fracs, cell, rng, insertion=False):
    n = order.size
    H, W = img.shape[:2]
    imgs = []
    for f in fracs:
        k = int(round(f * n))
        sel = np.zeros(n, bool)
        sel[order[:k]] = True
        m = cell_mask(~sel.reshape(shape) if insertion else sel.reshape(shape), (H, W), cell)
        imgs.append(imputer(img, m, rng))
    return adapter.logit(np.stack(imgs))


def faithfulness_image(adapter, img, attrs: dict, imputers: dict, fracs=DEFAULT_FRACS, cell=16, n_random=3, seed=0, insertion=False):
    """attrs: name -> Attribution. Returns a list of flat result rows (one per explainer x imputer)."""
    hw = img.shape[:2]
    l0 = float(adapter.logit(img[None])[0])
    s = direction(l0)
    n = (hw[0] // cell) * (hw[1] // cell)
    shape = (hw[0] // cell, hw[1] // cell)
    rows = []
    for iname, imp in imputers.items():
        rng = np.random.default_rng(seed)
        rand_curves = np.stack([_curve(adapter, img, np.random.default_rng(seed + 1 + r).permutation(n), shape, imp, fracs, cell, rng)
                                for r in range(n_random)])
        rand_drop = s * (l0 - rand_curves)                                   # [R, F]
        aopc_rand = float(rand_drop.mean())
        shock = float(np.abs(rand_curves - l0).mean())
        flip_rand = float(((np.sign(rand_curves) != s) & (rand_curves != 0)).any(1).mean())
        for name, at in attrs.items():
            order, _ = rank_cells(at.map, hw, cell, s)
            morf = _curve(adapter, img, order, shape, imp, fracs, cell, rng)
            lerf = _curve(adapter, img, order[::-1], shape, imp, fracs, cell, rng)
            d_m, d_l = s * (l0 - morf), s * (l0 - lerf)
            row = {"explainer": name, "imputer": iname, "logit0": l0, "direction": s,
                   "aopc_morf": float(d_m.mean()), "aopc_lerf": float(d_l.mean()), "aopc_rand": aopc_rand,
                   "gain": float(d_m.mean() - aopc_rand), "gap": float(d_m.mean() - d_l.mean()),
                   "shock": shock, "flip_rand": flip_rand,
                   "flips_at_morf_max": bool(np.sign(morf[-1]) != s)}
            if insertion:
                ins = _curve(adapter, img, order, shape, imp, fracs, cell, rng, insertion=True)
                row["aopc_ins"] = float((s * ins).mean())                    # higher = evidence alone restores the decision
                row["ins_gain"] = float((s * ins).mean() - (s * _curve(adapter, img, np.random.default_rng(seed + 99).permutation(n),
                                                                        shape, imp, fracs, cell, rng, insertion=True)).mean())
            rows.append(row)
    return rows


def completeness_rows(attrs: dict):
    """Axiomatic check for methods that claim it (cam: exact; eg: in expectation). Other methods have no claim."""
    out = []
    for name, at in attrs.items():
        if "completeness_rel_err" in at.meta:
            out.append({"explainer": name, "completeness_abs_err": at.meta["completeness_abs_err"],
                        "completeness_rel_err": at.meta["completeness_rel_err"]})
    return out
