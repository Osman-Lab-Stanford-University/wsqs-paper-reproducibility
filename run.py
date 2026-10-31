"""Command-line interface for the WSQS computational package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from wsqs.features import construct_features, feature_specification
from wsqs.models import model_settings, run_benchmark
from wsqs.response_coding import score_responses


def checksum(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1048576), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def demonstrate():
    example = pd.read_csv(
        ROOT / "example/synthetic25.csv",
        keep_default_na=False,
    )

    X, categorical, numeric = construct_features(example)

    scores = np.asarray([
        score_responses(row.to_dict())["CORE8"]
        for _, row in example.iterrows()
    ])

    settings = model_settings()["catboost"]

    model = CatBoostRegressor(
        iterations=12,
        learning_rate=settings["learning_rate"],
        depth=settings["depth"],
        loss_function=settings["loss_function"],
        random_seed=settings["random_seed"],
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(
        X.iloc[:20],
        scores[:20],
        cat_features=categorical,
    )

    predictions = model.predict(X.iloc[20:])

    if (
        predictions.shape != (5,)
        or not np.isfinite(predictions).all()
    ):
        raise RuntimeError("Synthetic demonstration failed")

    print("PASS | 25 synthetic records")
    print("PASS | 33 constructed predictors")
    print("PASS | CORE-8 scoring and CatBoost demonstration")
    print("No empirical model was fitted.")


def reproduce(data, output):
    data = data.resolve()
    output = output.resolve()

    if not data.is_file():
        raise ValueError("Analytical input file not found")

    if output == ROOT or ROOT in output.parents:
        raise ValueError(
            "Empirical outputs must be kept outside the repository"
        )

    if output.exists():
        raise FileExistsError(
            "Output directory already exists; choose a new one"
        )

    frame = pd.read_csv(
        data,
        low_memory=False,
        dtype={"ResponseId": str},
    )

    if "municipal_supply_applicable" not in frame.columns:
        if "source09M_municipal_eligible" not in frame.columns:
            raise ValueError(
                "Missing municipal supply applicability variable"
            )

        frame = frame.rename(columns={
            "source09M_municipal_eligible":
                "municipal_supply_applicable"
        })

    X, categorical, numeric = construct_features(frame)

    domains = list(
        feature_specification()["domains"]
    )

    raw_columns = [
        f"{domain}_raw_severity"
        for domain in domains
    ]

    missing = [
        name for name in raw_columns
        if name not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Missing harmonized domain-severity inputs: "
            + ", ".join(missing)
        )

    severities = frame[raw_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    y = (5.0 - severities.mean(axis=1)) * 25.0

    if (
        len(frame) != 6145
        or not y.notna().all()
        or not y.between(0, 100).all()
    ):
        raise ValueError(
            "Analytical sample or CORE-8 outcome mismatch"
        )

    print("Analytical records:", len(frame), flush=True)
    print("Predictor columns:", X.shape[1], flush=True)

    split_results, summary = run_benchmark(
        X,
        y,
        categorical,
        numeric,
    )

    output.mkdir(parents=True, exist_ok=False)

    split_results.to_csv(
        output / "benchmark_split_level.csv",
        index=False,
    )

    summary.to_csv(
        output / "benchmark_summary.csv",
        index=False,
    )

    metadata = {
        "input_sha256": checksum(data),
        "n_households": len(frame),
        "n_predictors": X.shape[1],
        "seeds": model_settings()["repeated_split_seeds"],
        "outcome": "Equal-weight CORE-8 WSQS",
        "output_type": "aggregate_metrics_only",
    }

    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("BENCHMARK SUMMARY")
    print(summary.to_string(index=False))
    print()
    print("PASS | Aggregate output files created")
    print("No individual predictions were exported.")


def main():
    parser = argparse.ArgumentParser(
        description="WSQS computational reproduction"
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=("demo", "reproduce"),
    )

    parser.add_argument(
        "--data",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
    )

    args = parser.parse_args()

    if args.mode == "demo":
        demonstrate()
    else:
        if args.data is None or args.output is None:
            parser.error(
                "Reproduction requires --data and --output"
            )

        reproduce(args.data, args.output)


if __name__ == "__main__":
    main()
