#!/usr/bin/env python
"""Evaluate explainers for the OSCC model: faithfulness, stability, artefact-robustness, expert alignment.

Manifest CSV columns:  path, patient, label (normal|oscc)  [, mask] [, mask2]
  mask / mask2 : binary PNG(s) of pathologist-marked tumour (any pixel > 0), any size (resized, nearest, to the model input frame).
  patient      : used for clustered bootstrap. Never leave it blank or per-image: images of one patient are not independent.

Example (offline evaluation on held-out TEST patients only):
  python research/xai_eval/run_eval.py --model-dir models/100x --manifest test_manifest.csv --out results/100x \\
         --methods cam,eg,occlusion --bank-dir data/train/normal --n-faith 200 --n-stab 60 --n-sanity 30

  python research/xai_eval/run_eval.py --smoke      # plumbing check on synthetic images. NOT a scientific result.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference"))
os.environ.setdefault("INTERNAL_TOKEN", "x" * 32)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from preprocess import preprocess_image, tissue_mask  # noqa: E402
from xai import report as R  # noqa: E402
from xai.adapter import KerasAdapter, OnnxAdapter, load_adapter  # noqa: E402
from xai.alignment import alignment_metrics, annotation_ablation, interrater, load_mask  # noqa: E402
from xai.explainers import CONTROLS, BaselineBank, explain_cam, explain_eg, explain_occlusion, run_control  # noqa: E402
from xai.faithfulness import DEFAULT_FRACS, completeness_rows, faithfulness_image  # noqa: E402
from xai.imputers import make_imputers  # noqa: E402
from xai.stability import default_perturbations, seed_stability_rows, stability_rows  # noqa: E402


def read_rgb(path, max_side=2400):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    s = max(im.shape[:2]) / max_side
    return cv2.resize(im, None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA) if s > 1 else im


def build_explainers(names, adapter, imputers, bank, eg_kw):
    ex = {}
    for n in names:
        if n == "cam":
            ex["cam"] = lambda img, seed=0: explain_cam(adapter, img)
        elif n == "eg":
            ex["eg"] = lambda img, seed=0: explain_eg(adapter, img, bank, seed=seed, **eg_kw)
        elif n == "occlusion":
            for iname in imputers:
                if iname in ("road", "constant"):                # the two ends of the artefact spectrum
                    ex[f"occlusion[{iname}]"] = (lambda im_, nm=iname: lambda img, seed=0: explain_occlusion(adapter, img, make_imputers((nm,))[nm], seed=seed))(iname)
        else:
            raise SystemExit(f"unknown method {n}")
    return ex


def smoke_dataset(out: Path, hw):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in range(4):
        for k in range(2):
            rng = np.random.default_rng(p * 10 + k)
            H, W = hw
            img = np.clip(cv2.GaussianBlur(rng.normal((200, 150, 190), 25, (H * 2, W * 2, 3)).astype(np.float32), (0, 0), 3), 0, 255).astype(np.uint8)
            m = np.zeros((H * 2, W * 2), np.uint8)
            cv2.ellipse(m, (int(W * (0.6 + 0.3 * rng.random())), int(H * (0.6 + 0.3 * rng.random()))), (W // 2, H // 2), 20, 0, 360, 255, -1)
            img[m > 0] = np.clip(img[m > 0] * np.array([0.55, 0.45, 0.8]), 0, 255)
            cv2.imwrite(str(out / f"p{p}_{k}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / f"p{p}_{k}_mask.png"), m)
            rows.append({"path": str(out / f"p{p}_{k}.png"), "patient": f"P{p}", "label": "oscc", "mask": str(out / f"p{p}_{k}_mask.png")})
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    return out / "manifest.csv"


def qualitative(path, items, hw, mask=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from xai.explainers import upsample
    n = 1 + len(items[1])
    fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 2.6))
    img = items[0]
    axs[0].imshow(img); axs[0].set_title("input", fontsize=8)
    for ax, (name, at) in zip(axs[1:], items[1].items()):
        m = upsample(at.map, hw); v = np.abs(m).max() + 1e-9
        ax.imshow(img); ax.imshow(m, cmap="RdBu_r", vmin=-v, vmax=v, alpha=0.55); ax.set_title(name, fontsize=8)
    for ax in axs:
        ax.axis("off")
        if mask is not None:
            ax.contour(mask.astype(float), levels=[0.5], colors="k", linewidths=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "100x"))
    ap.add_argument("--manifest"); ap.add_argument("--out", default="results/xai")
    ap.add_argument("--methods", default="cam,occlusion", help="cam,eg,occlusion")
    ap.add_argument("--imputers", default="constant,blur,shuffle,road")
    ap.add_argument("--backend", default="auto", choices=["auto", "onnx", "keras"])
    ap.add_argument("--bank-dir", help="folder of NORMAL-class TRAINING images for Expected Gradients baselines")
    ap.add_argument("--n-faith", type=int, default=100); ap.add_argument("--n-stab", type=int, default=40)
    ap.add_argument("--n-sanity", type=int, default=0); ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--fracs", default=",".join(map(str, DEFAULT_FRACS)))
    ap.add_argument("--eg-baselines", type=int, default=4); ap.add_argument("--eg-steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    methods = a.methods.split(","); imputer_names = a.imputers.split(","); fracs = tuple(float(f) for f in a.fracs.split(","))
    if a.smoke:
        print("=" * 78 + "\nSMOKE TEST: synthetic images. Checks the pipeline only. These numbers mean NOTHING about the model.\n" + "=" * 78)
        out = out / "_smoke"; out.mkdir(parents=True, exist_ok=True)
        cfg = json.loads((Path(a.model_dir) / "serving_config.json").read_text())
        a.manifest = str(smoke_dataset(out / "data", tuple(cfg["img_hw"])))
        a.n_faith, a.n_stab, a.n_random, fracs, imputer_names = 3, 1, 2, (0.1, 0.3), ["constant", "road"]
        methods = [m for m in methods if m != "eg"] or ["cam"]
    need_keras = "eg" in methods or a.n_sanity > 0
    adapter = KerasAdapter(Path(a.model_dir) / "oscc_calibrated.keras") if (need_keras or a.backend == "keras") else load_adapter(a.model_dir, a.backend)
    hw = adapter.img_hw
    if "cam" in methods and not adapter.has_cam:
        raise SystemExit(f"CAM unavailable: {getattr(adapter, 'cam_error', '')}")
    bank = BaselineBank.from_folder(a.bank_dir, lambda raw, h: preprocess_image(raw, h)[0], hw) if a.bank_dir else BaselineBank(None)
    if "eg" in methods and a.bank_dir is None:
        print("WARNING: no --bank-dir; EG uses the synthetic fallback baselines (glass/blur/noise). Report this, or supply normal-tissue training images.")
    imputers = make_imputers(imputer_names)
    explainers = build_explainers(methods, adapter, imputer_names, bank, {"n_baselines": a.eg_baselines, "n_steps": a.eg_steps})
    pre = lambda raw: preprocess_image(raw, hw)[0]

    mp = Path(a.manifest)
    if not mp.exists() or mp.stat().st_size == 0:
        raise SystemExit(f"manifest is empty or missing: {mp}\n"
                         "Create it with:  python research/xai_eval/make_manifest.py --root <folder with normal/ and oscc/> "
                         "--patient-regex \"<regex with the patient id in group 1>\" --out test_manifest.csv")
    man = pd.read_csv(mp)
    missing = {'path', 'patient', 'label'} - set(man.columns)
    if missing or len(man) == 0:
        raise SystemExit(f"manifest needs columns path,patient,label and at least one row (missing: {sorted(missing)}, rows: {len(man)})")
    man["image_id"] = np.arange(len(man))
    if "mask" not in man: man["mask"] = np.nan
    man["mask"] = man["mask"].where(man["mask"].astype(str).str.len() > 0, np.nan)
    if "mask2" not in man: man["mask2"] = np.nan
    man["mask2"] = man["mask2"].where(man["mask2"].astype(str).str.len() > 0, np.nan)
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(man))
    t0 = time.time()
    cache, faith, comp, align, ablate, stab, seedstab, rater = {}, [], [], [], [], [], [], []

    def get(i):
        if i not in cache:
            row = man.iloc[i]
            raw = read_rgb(row.path)
            cache[i] = (raw, pre(raw))
        return cache[i]

    # ---- 1) faithfulness + completeness + alignment (same maps, computed once)
    for n, i in enumerate(order[:a.n_faith]):
        row = man.iloc[i]
        raw, img = get(i)
        attrs = {k: fn(img, a.seed) for k, fn in explainers.items()}
        attrs.update({c: run_control(c, adapter, img, a.seed) for c in CONTROLS})
        for r in faithfulness_image(adapter, img, attrs, imputers, fracs=fracs, n_random=a.n_random, seed=a.seed):
            faith.append({**r, "patient": row.patient, "image_id": int(i), "label": row.label})
        comp += [{**c, "patient": row.patient, "image_id": int(i)} for c in completeness_rows(attrs)]
        if isinstance(row["mask"], str):
            m = load_mask(row["mask"], hw)
            for k, at in attrs.items():
                met = alignment_metrics(at.map, m, hw)
                if met: align.append({"explainer": k, "patient": row.patient, "image_id": int(i), "correct": (at.logit > 0) == (row.label == "oscc"), **met})
            ablate += [{**r, "patient": row.patient, "image_id": int(i)} for r in annotation_ablation(adapter, img, m, imputers, tissue_mask(img), seed=a.seed)]
            if isinstance(row["mask2"], str):
                rater.append({**interrater(m, load_mask(row["mask2"], hw)), "patient": row.patient})
        if n < 4:
            qualitative(out / f"qualitative_{n}.png", (img, {k: v for k, v in attrs.items() if not k.startswith("ctrl_")}), hw, load_mask(row["mask"], hw) if isinstance(row["mask"], str) else None)
        print(f"[faith {n + 1}/{min(a.n_faith, len(man))}] {time.time() - t0:.0f}s", flush=True)

    # ---- 2) stability
    perts = default_perturbations(hw)
    stab_ids = order[:a.n_stab]
    for n, i in enumerate(stab_ids):
        raw, img = get(i)
        other = get(stab_ids[(n + 1) % len(stab_ids)])[1]
        fns = {**explainers, "ctrl_random": lambda im, sd=0: run_control("ctrl_random", adapter, im, sd)}
        others = {k: fn(other, a.seed).map for k, fn in explainers.items()}
        others["ctrl_random"] = run_control("ctrl_random", adapter, other, a.seed + 1).map
        for r in stability_rows(raw, pre, adapter, fns, perts, seed=a.seed, other_maps=others):
            stab.append({**r, "patient": man.iloc[i].patient, "image_id": int(i)})
        for k, fn in explainers.items():
            s = 1 if next(iter(fns.values()))(img, a.seed).logit >= 0 else -1
            for r in seed_stability_rows(img, fn, seeds=(0, 1), s=s):
                seedstab.append({"explainer": k, "image_id": int(i), "patient": man.iloc[i].patient, **r})
        print(f"[stab {n + 1}/{len(stab_ids)}] {time.time() - t0:.0f}s", flush=True)

    # ---- 3) sanity: cascading model randomisation (needs Keras)
    sanity_rows = []
    if a.n_sanity > 0 and isinstance(adapter, KerasAdapter):
        from xai.sanity import cascading_randomisation
        imgs = [get(i)[1] for i in order[:a.n_sanity]]
        sanity_rows = cascading_randomisation(adapter, imgs, lambda ad, im: explain_cam(ad, im) if "cam" in methods else next(iter(build_explainers(methods, ad, imputer_names, bank, {}).values()))(im))

    # ---- write-out
    summary = {"smoke_test": a.smoke, "n_images": int(len(man)), "n_patients": int(man.patient.nunique()), "model_dir": str(a.model_dir),
               "config": {"fracs": fracs, "imputers": imputer_names, "seed": a.seed, "methods": methods, "eg_baseline": bank.kind,
                          "head_identity_error": getattr(adapter, "identity_error", None)}}
    for name, rows in [("faithfulness", faith), ("stability", stab), ("seed_stability", seedstab), ("alignment", align),
                       ("ablation", ablate), ("completeness", comp), ("sanity_randomisation", sanity_rows), ("interrater", rater)]:
        if rows: pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    if faith:
        df = pd.DataFrame(faith)
        tab, verdict = R.summarise_faithfulness(df)
        tab.to_csv(out / "faithfulness_summary.csv", index=False)
        (out / "faithfulness_gain.tex").write_text(R.to_latex(tab[["explainer", "imputer", "gain", "lo", "hi", "gap", "shock"]], "Faithfulness gain over random removal (nats; 95\\% patient-clustered CI), by imputer.", "tab:faith"))
        summary["faithfulness"] = verdict
        summary["imputer_artefact_index"] = R.imputer_artefact_index(df).to_dict("records")
    if comp:
        c = pd.DataFrame(comp)
        summary["completeness"] = {k: {"median_rel_err": float(g.completeness_rel_err.median()), "p95_rel_err": float(g.completeness_rel_err.quantile(.95))} for k, g in c.groupby("explainer")}
    if stab:
        st = R.summarise_stability(pd.DataFrame(stab)); st.to_csv(out / "stability_summary.csv", index=False)
        summary["stability"] = st.round(4).to_dict("records")
        if seedstab:
            summary["seed_stability_spearman_median"] = pd.DataFrame(seedstab).groupby("explainer").spearman.median().round(4).to_dict()
    if align:
        al = pd.DataFrame(align)
        at_tab, tests = R.summarise_alignment(al); at_tab.to_csv(out / "alignment_summary.csv", index=False)
        summary["alignment"] = {"table": at_tab.round(4).to_dict("records"), "vs_controls_rra_excess": tests,
                                "correctly_classified_only": R.summarise_alignment(al[al.correct])[0].round(4).to_dict("records") if al.correct.any() else []}
    if ablate:
        ab = pd.DataFrame(ablate)
        summary["annotation_ablation"] = {k: {"necessity_gain": list(np.round(R.cluster_bootstrap_ci(g.necessity_gain.values, g.patient.values)[:3], 4)),
                                              "sufficiency_gain": list(np.round(R.cluster_bootstrap_ci(g.sufficiency_gain.values, g.patient.values)[:3], 4))} for k, g in ab.groupby("imputer")}
    if sanity_rows:
        sr = pd.DataFrame(sanity_rows)
        summary["sanity_randomisation_abs_spearman_median"] = sr.groupby("stage", sort=False).abs_spearman.median().round(4).to_dict()
    if rater:
        summary["interrater_ceiling"] = pd.DataFrame(rater)[["iou", "dice"]].median().round(4).to_dict()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"done in {time.time() - t0:.0f}s -> {out}/summary.json")


if __name__ == "__main__":
    main()
