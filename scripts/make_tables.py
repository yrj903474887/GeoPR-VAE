"""Generate editable statistics and mineral-association table bodies."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation.regional import ELEMENTS, ASSOCIATIONS

METHODS = ["AE", "VAE", "VAE_patch", "GANomaly", "Skip_GANomaly", "GeoPR-VAE"]
LABELS = {"AE": "AE", "VAE": "VAE", "VAE_patch": "Patch VAE", "GANomaly": "GANomaly",
          "Skip_GANomaly": "Skip-GANomaly", "GeoPR-VAE": "GeoPR-VAE"}


def descriptive_statistics(data):
    values = data[list(ELEMENTS)].to_numpy(float)
    if len(values) < 3 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("At least three complete positive concentration records are required")
    rows = []
    for j, element in enumerate(ELEMENTS):
        v = values[:, j]
        mean, sd, n = v.mean(), v.std(ddof=1), len(v)
        skew = 0.0 if sd == 0 else n / ((n - 1) * (n - 2)) * np.sum(((v - mean) / sd) ** 3)
        rows.append(dict(element=element, n=n, min=v.min(), median=np.median(v), mean=mean,
                         max=v.max(), SD=sd, CV=sd / mean, skewness=skew,
                         unit="ppb" if element in ("Au", "Ag") else "ppm"))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("statistics", "association"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.kind == "statistics":
        table = descriptive_statistics(data)
        table.to_csv(args.output / "descriptive_statistics.csv", index=False)
        lines = [r"Element & Min & Median & Mean & Max & CV & Skewness \\", r"\midrule"]
        for row in table.to_dict("records"):
            lines.append(row["element"] + " & " + " & ".join(f"{row[k]:.3g}" for k in ("min", "median", "mean", "max", "CV", "skewness")) + r" \\")
    else:
        lines = [r"Fixed association & Method & $n$ & PB-AUC & $N_{10}$ & $A_{50}$ (\%) \\", r"\midrule"]
        for name in ASSOCIATIONS:
            group = data[(data.association == name) & data.model.isin(METHODS)]
            if len(group) != len(METHODS) or group.model.nunique() != len(METHODS):
                raise ValueError("Expected one row for each of the six compared methods")
            best = dict(PB_AUC=group.PB_AUC.max(), N10=group.N10.max(), A50=group.A50.min())
            for i, method in enumerate(METHODS):
                row = group[group.model == method].iloc[0]
                cells = []
                for key in ("PB_AUC", "N10", "A50"):
                    text = f"{row[key]:.4f}" if key == "PB_AUC" else str(int(row[key])) if key == "N10" else f"{row[key] * 100:.2f}"
                    cells.append(r"\textbf{" + text + "}" if abs(row[key] - best[key]) <= 1e-12 else text)
                lines.append((name.replace("-", "--") if i == 0 else "") + " & " + LABELS[method] + " & " + str(int(row["n"])) + " & " + " & ".join(cells) + r" \\")
    (args.output / "table_body.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
