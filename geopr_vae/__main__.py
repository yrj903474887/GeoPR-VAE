"""Command line: python -m geopr_vae --help."""
import argparse
import json
from pathlib import Path
import numpy as np
from . import pipeline
from .mapping import grid_scores


def read_npz(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {k: arrays[k] for k in arrays.files}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="GeoPR-VAE research workflow")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Fit and score one complete input field")
    train.add_argument("--input", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--method", choices=[*pipeline.CONFIGS, "all"], default="GeoPR-VAE")
    train.add_argument("--teacher-epochs", type=int, default=600)
    train.add_argument("--head-epochs", type=int, default=300)
    train.add_argument("--seed", type=int, default=20260929)
    train.add_argument("--head-seed", type=int, help="Recovery-head seed; defaults to --seed")
    infer = commands.add_parser("predict", help="Score a complete new query field")
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    grid = commands.add_parser("grid", help="Interpolate scores to a prefiltered metre-coordinate grid")
    grid.add_argument("--prediction", type=Path, required=True)
    grid.add_argument("--grid", type=Path, required=True, help="NPZ with xy[M,2] in the same CRS")
    grid.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        data = read_npz(args.input)
        methods = pipeline.CONFIGS if args.method == "all" else [args.method]
        for method in methods:
            destination = args.output / method
            if destination.exists() and any(destination.iterdir()):
                raise FileExistsError(f"Use a new output directory: {destination}")
            destination.mkdir(parents=True, exist_ok=True)
            bundle, history = pipeline.fit(data, method, data.get("train"), data.get("calibration"),
                                           args.seed, args.teacher_epochs, args.head_epochs, head_seed=args.head_seed)
            pipeline.save(bundle, destination / "model.pt")
            arrays, details = pipeline.predict(bundle, data)
            np.savez_compressed(destination / "prediction.npz", **arrays)
            write_json(destination / "training.json", dict(method=method, **bundle["training"], history=history))
            write_json(destination / "scoring.json", details)
            print(f"{method}: {len(arrays['ids'])} query rows -> {destination}", flush=True)
    elif args.command == "predict":
        if args.output.exists():
            raise FileExistsError(args.output)
        arrays, details = pipeline.predict(pipeline.load(args.model), read_npz(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **arrays)
        write_json(args.output.with_suffix(".json"), details)
        print(f"Scored {len(arrays['ids'])} query rows", flush=True)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        scores, points = read_npz(args.prediction), read_npz(args.grid)
        values = grid_scores(scores["xy"], scores["affine_scores"], points["xy"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **values)
        print(f"Interpolated {len(values['xy'])} cells", flush=True)


if __name__ == "__main__":
    main()
