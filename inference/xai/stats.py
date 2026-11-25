"""Statistics that survive review: images from one patient are not independent, so every interval resamples PATIENTS."""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def patient_means(values, groups):
    values, groups = np.asarray(values, float), np.asarray(groups)
    ok = ~np.isnan(values)
    keys = sorted(set(groups[ok]))
    return keys, np.array([values[ok & (groups == k)].mean() for k in keys])


def cluster_bootstrap_ci(values, groups, stat=np.mean, n_boot=4000, alpha=0.05, seed=0):
    """Patient-level percentile bootstrap. Returns (estimate, lo, hi, n_patients)."""
    keys, pm = patient_means(values, groups)
    if len(pm) == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pm), (n_boot, len(pm)))
    bs = np.array([stat(pm[i]) for i in idx])
    return float(stat(pm)), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2)), len(pm)


def paired_wilcoxon(a, b, groups):
    """Paired two-sided Wilcoxon on per-patient means (a, b aligned per row)."""
    ka, pa = patient_means(a, groups)
    kb, pb = patient_means(b, groups)
    d = pa - pb
    if len(d) < 6 or np.allclose(d, 0):
        return np.nan, float(np.median(d)) if len(d) else np.nan
    return float(wilcoxon(pa, pb).pvalue), float(np.median(d))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(np.where(np.isnan(p), 2, p))
    m = int((~np.isnan(p)).sum())
    out = np.full_like(p, np.nan)
    running = 0.0
    for rank, i in enumerate(order[:m]):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def kendall_across(table: dict):
    """table: imputer -> {explainer: score}. Mean pairwise Kendall tau of explainer rankings across imputers."""
    from scipy.stats import kendalltau
    names = sorted(next(iter(table.values())))
    ims = sorted(table)
    taus = {}
    for i in range(len(ims)):
        for j in range(i + 1, len(ims)):
            taus[f"{ims[i]}~{ims[j]}"] = float(kendalltau([table[ims[i]][n] for n in names], [table[ims[j]][n] for n in names]).statistic)
    return taus
