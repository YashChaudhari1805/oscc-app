"""Build a manifest CSV from a folder laid out as  ROOT/normal/*.jpg  and  ROOT/oscc/*.jpg  (optionally ROOT/masks/<stem>.png).

Patient id is taken from the file name with --patient-regex (first capture group), e.g. --patient-regex "^(\\d+)_".
If your files carry no patient id, fix that before evaluating: per-image ids make the confidence intervals too narrow.
"""
import argparse, re
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--out", default="test_manifest.csv")
ap.add_argument("--patient-regex", required=True)
a = ap.parse_args()
root, rows, rx = Path(a.root), [], re.compile(a.patient_regex)
for label in ("normal", "oscc"):
    for p in sorted((root / label).glob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            continue
        m = rx.search(p.name)
        if not m:
            raise SystemExit(f"no patient id in {p.name} with {a.patient_regex}")
        mask = root / "masks" / f"{p.stem}.png"
        rows.append({"path": str(p), "patient": m.group(1), "label": label, "mask": str(mask) if mask.exists() else ""})
df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
print(df.groupby("label").size().to_string(), "\npatients:", df.patient.nunique(), "-> ", a.out)
