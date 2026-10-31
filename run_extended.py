"""Execute additional WSQS analysis using authorized local data."""

from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from wsqs.features import construct_features
from wsqs.extended import (
    alternative_outcomes,
    block_shapley,
    reproduce_extended,
    scientific_plans,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", choices=("demo", "reproduce"),
        required=True,
    )

    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    if args.mode == "demo":
        plans = scientific_plans()

        assert tuple(
            len(plans["blocks"][key])
            for key in ("S", "C", "P")
        ) == (22, 3, 8)

        example = {
            "": 0.0,
            "S": 0.5,
            "C": 0.04,
            "P": 0.03,
            "SC": 0.52,
            "SP": 0.53,
            "CP": 0.06,
            "SCP": 0.55,
        }

        allocated = block_shapley(example)

        assert np.isclose(
            sum(allocated.values()), 0.55
        )

        print("PASS | Original 33-predictor block specification")
        print("PASS | Eight cross-domain predictor plans")
        print("PASS | Three-block Shapley mathematical identity")
        print("No empirical model was fitted.")
        return

    if args.data is None or args.output is None:
        parser.error(
            "Empirical reproduction requires --data and --output"
        )

    data = args.data.resolve()
    output = args.output.resolve()

    if not data.is_file():
        raise FileNotFoundError(data)

    if output.exists():
        raise FileExistsError(
            "Output already exists. Historical artifacts are protected."
        )

    if output == HERE or HERE in output.parents:
        raise ValueError(
            "Empirical results must be stored outside public source."
        )

    df = pd.read_csv(
        data, low_memory=False,
        dtype={"ResponseId": str},
    )

    if "municipal_supply_applicable" not in df:
        df = df.rename(columns={
            "source09M_municipal_eligible":
                "municipal_supply_applicable"
        })

    X, categorical, numeric = construct_features(df)

    print("Analytical sample:", len(df), flush=True)
    print("Predictor columns:", X.shape[1], flush=True)

    tables, metadata = reproduce_extended(
        df, X, categorical, numeric
    )

    output.mkdir(parents=True, exist_ok=False)

    for name, table in tables.items():
        table.to_csv(
            output / (name + ".csv"),
            index=False,
        )

        print()
        print(name)
        print(table.to_string(index=False))

    h = hashlib.sha256()

    with data.open("rb") as source:
        for chunk in iter(
            lambda: source.read(1048576), b""
        ):
            h.update(chunk)

    metadata["input_sha256"] = h.hexdigest()

    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("PASS | Aggregate analysis results written")
    print("No household predictions exported.")


if __name__ == "__main__":
    main()
