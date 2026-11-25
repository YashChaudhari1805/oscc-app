"""Turn per-image rows into the numbers and tables a paper needs. Every interval is a patient-clustered bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import cluster_bootstrap_ci, holm, kendall_across, paired_wilcoxon


def _is_control(name):
    return name.startswith("ctrl_")


def summarise_faithfulness(df: pd.DataFrame, metric="gain"):
    """df rows: patient, explainer, imputer, gain, gap, shock ... Returns (table, verdict)."""
    recs = []
    for (ex, imp), g in df.groupby(["explainer", "imputer"]):
        est, lo, hi, n = cluster_bootstrap_ci(g[metric].values, g["patient"].values)
        recs.append({"explainer": ex, "imputer": imp, metric: est, "lo": lo, "hi": hi, "n_patients": n,
                     "gap": g["gap"].mean(), "shock": g["shock"].mean()})
    tab = pd.DataFrame(recs)
    verdict = {}
    models = [e for e in tab.explainer.unique() if not _is_control(e)]
    ctrls = [e for e in tab.explainer.unique() if _is_control(e)]
    for ex in models:
        t = tab[tab.explainer == ex]
        v = {"positive_under_all_imputers": bool((t["lo"] > 0).all()),
             "per_imputer": {r.imputer: [round(r[metric], 4), round(r.lo, 4), round(r.hi, 4)] for _, r in t.iterrows()}}
        if ctrls:
            beats = {}
            for imp in df.imputer.unique():
                a = df[(df.explainer == ex) & (df.imputer == imp)].set_index("image_id")
                best, bp = None, None
                for c in ctrls:
                    b = df[(df.explainer == c) & (df.imputer == imp)].set_index("image_id")
                    idx = a.index.intersection(b.index)
                    d = a.loc[idx, metric].mean() - b.loc[idx, metric].mean()
                    if best is None or b.loc[idx, metric].mean() > best[1]:
                        p, _ = paired_wilcoxon(a.loc[idx, metric].values, b.loc[idx, metric].values, a.loc[idx, "patient"].values)
                        best, bp = (c, float(b.loc[idx, metric].mean()), float(d)), p
                beats[imp] = {"strongest_control": best[0], "control_gain": round(best[1], 4), "advantage": round(best[2], 4), "p": bp}
            ps = holm([b["p"] for b in beats.values()])
            for (imp, b), ph in zip(beats.items(), ps):
                b["p_holm"] = None if np.isnan(ph) else float(ph)
                b["beats_control"] = bool(b["advantage"] > 0 and (b["p_holm"] is not None and b["p_holm"] < 0.05))
            v["vs_controls"] = beats
            v["beats_all_controls_under_all_imputers"] = all(b["beats_control"] for b in beats.values())
        verdict[ex] = v
    table = {imp: {r.explainer: r[metric] for _, r in tab[tab.imputer == imp].iterrows() if not _is_control(r.explainer)} for imp in tab.imputer.unique()}
    rank_agreement = kendall_across(table) if len(table) > 1 and len(models) > 2 else {}
    return tab, {"methods": verdict, "ranking_agreement_kendall_tau": rank_agreement}


def imputer_artefact_index(df: pd.DataFrame):
    """How much the imputer ITSELF disturbs the model: mean |logit shift| under random removal + how often random removal flips the call."""
    g = df.drop_duplicates(["image_id", "imputer"]).groupby("imputer")
    return g.agg(shock_nats=("shock", "mean"), random_flip_rate=("flip_rand", "mean")).reset_index()


def summarise_stability(df: pd.DataFrame):
    recs = []
    for (ex, kind), g in df.groupby(["explainer", "kind"]):
        gs = g[g.pred_stable]
        rec = {"explainer": ex, "kind": kind, "n": len(g), "n_pred_stable": len(gs),
               "spearman_median": gs.spearman.median(), "spearman_q25": gs.spearman.quantile(.25), "spearman_q75": gs.spearman.quantile(.75),
               "jaccard_top20_median": gs.jaccard_top20.median()}
        if "chance_spearman" in g:
            rec["chance_spearman_median"] = g.chance_spearman.median()
        valid = g.dropna(subset=["spearman"])
        if len(valid) > 5 and valid.dlogit.abs().std() > 0:
            from scipy.stats import spearmanr
            rec["drift_coupling_rho"] = float(spearmanr(valid.dlogit.abs(), 1 - valid.spearman).statistic)
        recs.append(rec)
    return pd.DataFrame(recs)


def summarise_alignment(df: pd.DataFrame, metrics=("auroc", "rra_excess", "rma_excess", "pointing", "iou_topk")):
    recs = []
    for ex, g in df.groupby("explainer"):
        rec = {"explainer": ex, "n_images": len(g)}
        for m in metrics:
            est, lo, hi, n = cluster_bootstrap_ci(g[m].values, g["patient"].values)
            rec.update({m: est, m + "_lo": lo, m + "_hi": hi})
        recs.append(rec)
    tab = pd.DataFrame(recs)
    ctrl = [e for e in tab.explainer if _is_control(e)]
    out = {}
    for ex in [e for e in tab.explainer if not _is_control(e)]:
        a = df[df.explainer == ex].set_index("image_id")
        res = {}
        for c in ctrl:
            b = df[df.explainer == c].set_index("image_id")
            idx = a.index.intersection(b.index)
            p, med = paired_wilcoxon(a.loc[idx, "rra_excess"].values, b.loc[idx, "rra_excess"].values, a.loc[idx, "patient"].values)
            res[c] = {"median_diff_rra_excess": med, "p": p}
        ps = holm([r["p"] for r in res.values()])
        for r, ph in zip(res.values(), ps):
            r["p_holm"] = None if np.isnan(ph) else float(ph)
        out[ex] = res
    return tab, out


def to_latex(df: pd.DataFrame, caption, label, float_fmt="%.3f"):
    return df.to_latex(index=False, float_format=lambda x: float_fmt % x, caption=caption, label=label, escape=True)
