"""Reproduce the WSQS forecasting benchmark.

The target is the equal-weight CORE-8 WSQS on a 0–100 scale.
Model selection uses internal validation. The held-out test set
is used only for evaluation after fitting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LEARNERS = (
    "CatBoost",
    "HistGradientBoosting",
    "Extra Trees",
    "Random Forest",
    "ElasticNet",
    "Ridge",
    "Mean baseline",
)


def model_settings():
    path = (
        Path(__file__).resolve().parents[2]
        / "config/models.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def partition_indices(n, seed):
    development, test = train_test_split(
        np.arange(n),
        test_size=0.20,
        random_state=seed,
    )

    fitting, validation = train_test_split(
        development,
        test_size=0.20,
        random_state=seed,
    )

    full_development = np.sort(
        np.concatenate((fitting, validation))
    )

    if (len(fitting), len(validation), len(test)) != (
        3932, 984, 1229
    ):
        raise ValueError(
            "Reproduction requires the 6,145-household dataset"
        )

    if len(
        set(fitting) | set(validation) | set(test)
    ) != n:
        raise ValueError("Partitions do not cover the dataset")

    if (
        set(fitting) & set(validation)
        or set(fitting) & set(test)
        or set(validation) & set(test)
    ):
        raise ValueError("Train-validation-test overlap")

    return (
        fitting,
        validation,
        test,
        full_development,
    )


def prepare_catboost(X, categorical, numeric):
    prepared = X.copy()

    for col in categorical:
        prepared[col] = (
            prepared[col]
            .astype("string")
            .fillna("__UNANSWERED__")
            .astype(str)
        )

    for col in numeric:
        prepared[col] = pd.to_numeric(
            prepared[col], errors="coerce"
        ).astype(float)

    return prepared


def prepare_sklearn(X, categorical, numeric):
    prepared = X.copy()

    for col in categorical:
        values = prepared[col].astype("string")
        prepared[col] = (
            values.astype(object)
            .where(values.notna(), np.nan)
        )

    for col in numeric:
        prepared[col] = pd.to_numeric(
            prepared[col], errors="coerce"
        ).astype(float)

    return prepared


def sklearn_preprocessor(categorical, numeric, linear):
    steps = [
        ("impute", SimpleImputer(strategy="median"))
    ]

    if linear:
        steps.append(("scale", StandardScaler()))

    return ColumnTransformer([
        (
            "numeric",
            Pipeline(steps),
            numeric,
        ),
        (
            "categorical",
            Pipeline([
                (
                    "impute",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__UNANSWERED__",
                    ),
                ),
                (
                    "encode",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]),
            categorical,
        ),
    ])


def summary_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite predictions")

    rmse = float(
        np.sqrt(mean_squared_error(actual, predicted))
    )

    rho = (
        float(spearmanr(actual, predicted).statistic)
        if np.ptp(predicted) > 1e-12
        else np.nan
    )

    return {
        "rmse_100": rmse,
        "mae_100": float(
            mean_absolute_error(actual, predicted)
        ),
        "rmse_raw_equivalent": rmse / 25.0,
        "r2": float(r2_score(actual, predicted)),
        "spearman": rho,
    }


def catboost_predictions(
    X,
    y,
    categorical,
    fitting,
    validation,
    test,
    development,
    config,
):
    params = dict(config["catboost"])
    params.update(
        verbose=False,
        allow_writing_files=False,
    )

    first = CatBoostRegressor(**params)

    first.fit(
        Pool(
            X.iloc[fitting],
            y.iloc[fitting],
            cat_features=categorical,
        ),
        eval_set=Pool(
            X.iloc[validation],
            y.iloc[validation],
            cat_features=categorical,
        ),
        use_best_model=True,
    )

    best = first.get_best_iteration()

    if best is None or int(best) < 0:
        raise RuntimeError(
            "CatBoost validation did not select an iteration"
        )

    n_trees = int(best) + 1

    final = dict(params)
    final["iterations"] = n_trees
    final.pop("od_type", None)
    final.pop("od_wait", None)

    model = CatBoostRegressor(**final)

    model.fit(
        Pool(
            X.iloc[development],
            y.iloc[development],
            cat_features=categorical,
        )
    )

    prediction = model.predict(
        Pool(
            X.iloc[test],
            cat_features=categorical,
        )
    )

    return prediction, n_trees


def fixed_estimator(name, seed, config):
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            random_state=seed,
            **config["histogram_gradient_boosting"],
        )

    if name == "Extra Trees":
        return ExtraTreesRegressor(
            random_state=seed,
            **config["extra_trees"],
        )

    if name == "Random Forest":
        return RandomForestRegressor(
            random_state=seed,
            **config["random_forest"],
        )

    if name == "ElasticNet":
        return ElasticNet(
            random_state=seed,
            **config["elastic_net"],
        )

    raise ValueError("Unknown learner")


def other_predictions(
    name,
    X,
    y,
    categorical,
    numeric,
    seed,
    fitting,
    validation,
    test,
    development,
    config,
):
    if name == "Mean baseline":
        return (
            np.full(
                len(test),
                float(y.iloc[development].mean()),
            ),
            np.nan,
        )

    if name == "Ridge":
        candidates = []

        for alpha in config["ridge"]["alpha_candidates"]:
            trial = Pipeline([
                (
                    "preprocess",
                    sklearn_preprocessor(
                        categorical,
                        numeric,
                        linear=True,
                    ),
                ),
                ("model", Ridge(alpha=alpha)),
            ])

            trial.fit(
                X.iloc[fitting],
                y.iloc[fitting],
            )

            predictions = trial.predict(
                X.iloc[validation]
            )

            validation_rmse = summary_metrics(
                y.iloc[validation],
                predictions,
            )["rmse_100"]

            candidates.append(
                (validation_rmse, alpha)
            )

        _, alpha = min(candidates)

        pipeline = Pipeline([
            (
                "preprocess",
                sklearn_preprocessor(
                    categorical,
                    numeric,
                    linear=True,
                ),
            ),
            ("model", Ridge(alpha=alpha)),
        ])

    else:
        alpha = np.nan

        pipeline = Pipeline([
            (
                "preprocess",
                sklearn_preprocessor(
                    categorical,
                    numeric,
                    linear=(name == "ElasticNet"),
                ),
            ),
            (
                "model",
                fixed_estimator(name, seed, config),
            ),
        ])

    pipeline.fit(
        X.iloc[development],
        y.iloc[development],
    )

    return pipeline.predict(X.iloc[test]), alpha


def run_benchmark(
    X,
    y,
    categorical,
    numeric,
    *,
    progress=print,
):
    config = model_settings()
    n = len(X)

    if n != 6145 or X.shape[1] != 33 or len(y) != n:
        raise ValueError(
            "Expected the study's 6,145 × 33 analytical inputs"
        )

    if len(categorical) + len(numeric) != 33:
        raise ValueError(
            "Predictor type specification mismatch"
        )

    X = X.reset_index(drop=True)

    y = pd.Series(
        np.asarray(y, dtype=float).reshape(-1)
    )

    if (
        not y.notna().all()
        or not y.between(0, 100).all()
    ):
        raise ValueError(
            "CORE-8 outcome has invalid or missing values"
        )

    cat_X = prepare_catboost(
        X, categorical, numeric
    )

    sklearn_X = prepare_sklearn(
        X, categorical, numeric
    )

    records = []
    seeds = config["repeated_split_seeds"]

    for repeat, seed in enumerate(seeds, 1):
        fit, val, test, dev = partition_indices(
            n, int(seed)
        )

        progress(
            f"SPLIT {repeat}/{len(seeds)} | seed={seed} | "
            f"fit={len(fit)} validation={len(val)} "
            f"test={len(test)}"
        )

        predicted, iterations = catboost_predictions(
            cat_X,
            y,
            categorical,
            fit,
            val,
            test,
            dev,
            config,
        )

        metrics = summary_metrics(
            y.iloc[test], predicted
        )

        records.append({
            "seed": seed,
            "learner": "CatBoost",
            "selected_iterations": iterations,
            "selected_alpha": np.nan,
            **metrics,
        })

        progress(
            f"  CatBoost | RMSE={metrics['rmse_100']:.9f} "
            f"| selected_iterations={iterations}"
        )

        for name in LEARNERS[1:]:
            predicted, alpha = other_predictions(
                name,
                sklearn_X,
                y,
                categorical,
                numeric,
                int(seed),
                fit,
                val,
                test,
                dev,
                config,
            )

            metrics = summary_metrics(
                y.iloc[test], predicted
            )

            records.append({
                "seed": seed,
                "learner": name,
                "selected_iterations": np.nan,
                "selected_alpha": alpha,
                **metrics,
            })

            extra = (
                f" | alpha={alpha}"
                if name == "Ridge"
                else ""
            )

            progress(
                f"  {name} | RMSE="
                f"{metrics['rmse_100']:.9f}{extra}"
            )

    detailed = pd.DataFrame.from_records(records)

    summary = (
        detailed
        .groupby("learner", sort=False)
        .agg(
            n_splits=("seed", "size"),
            mean_rmse_100=("rmse_100", "mean"),
            sd_rmse_100=("rmse_100", "std"),
            mean_rmse_raw_equivalent=(
                "rmse_raw_equivalent", "mean"
            ),
            sd_rmse_raw_equivalent=(
                "rmse_raw_equivalent", "std"
            ),
            mean_mae_100=("mae_100", "mean"),
            mean_r2=("r2", "mean"),
            mean_spearman=("spearman", "mean"),
        )
        .reset_index()
    )

    summary = (
        summary
        .set_index("learner")
        .loc[list(LEARNERS)]
        .reset_index()
    )

    return detailed, summary
