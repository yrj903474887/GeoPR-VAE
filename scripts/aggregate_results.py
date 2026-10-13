"""Aggregate supplied case-level results for the paper figures."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation.controlled import recovery_tables, weak_fold_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("controlled", "weak", "weak-json"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cases", type=Path)
    source.add_argument("--manifest", type=Path, help="weak-json CSV: method,buffer_km,fold,path (relative to manifest)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "weak-json":
        if args.manifest is None:
            parser.error("weak-json requires --manifest")
        manifest = pd.read_csv(args.manifest)
        if not {"method", "buffer_km", "fold", "path"} <= set(manifest.columns):
            parser.error("Manifest requires method,buffer_km,fold,path")
        if manifest.duplicated(["method", "buffer_km", "fold"]).any():
            parser.error("Duplicate method/buffer/fold manifest entries")
        records = []
        for item in manifest.to_dict("records"):
            path = Path(item["path"])
            if not path.is_absolute():
                path = args.manifest.parent / path
            doc = json.loads(path.read_text(encoding="utf-8"))
            if len(doc["cases"]) != 144 or len({r["case_id"] for r in doc["cases"]}) != 144:
                raise ValueError("Expected 144 distinct weak-injection cases: " + str(path))
            for metric in ("delta_auc", "delta_core_recall_top10"):
                computed = np.mean([row[metric] for row in doc["cases"]])
                np.testing.assert_allclose(computed, doc["fold_mean"][metric], atol=1e-14, rtol=0)
            records.extend(dict(method=item["method"], buffer_km=item["buffer_km"], fold=item["fold"], **row) for row in doc["cases"])
        frame = pd.DataFrame(records)
    else:
        if args.cases is None:
            parser.error("controlled/weak require --cases")
        frame = pd.read_csv(args.cases)
    args.output.mkdir(parents=True, exist_ok=True)
    tables = recovery_tables(frame) if args.kind == "controlled" else {"fold_metrics": weak_fold_table(frame)}
    for name, table in tables.items():
        table.to_csv(args.output / (name + ".csv"), index=False)
    print(f"Wrote {len(tables)} tables to {args.output}")


if __name__ == "__main__":
    main()
