"""Additional WSQS analyses using the verified model implementation.

All evaluation statistics are predictive and descriptive.
No causal interpretation is implied.

Only aggregate results are returned for publication output.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from wsqs.models import (
    catboost_predictions,
    model_settings,
    partition_indices,
    prepare_catboost,
    summary_metrics,
)


DOMAINS = (
    "Affordability",
    "Access",
    "Reliability",
    "Efficiency",
    "Acceptability",
    "Quality",
    "Quantity",
    "Customer_service",
)

BLOCK_ORDER = ("S", "C", "P")
BLOCK_KEYS = ("S", "C", "P", "SC", "SP", "CP", "SCP")

MISSING_CATEGORY = "__MISSING__"


def scientific_plans():
    path = (
        Path(__file__).resolve().parents[2]
        / "config/scientific_plans.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def block_columns(block, plans=None):
    plans = plans or scientific_plans()

    if block not in BLOCK_KEYS:
        raise ValueError("Unknown predictor block")

    return [
        column
        for column in plans["full"]
        if any(
            column in plans["blocks"][letter]
            for letter in block
        )
    ]


OUTCOME_BLOCK = {
    "CORE7A": "SC",
    "CORE7B": "SCP",
    "POPULATION_PRIORITY": "SC",
    "INDIVIDUAL_PRIORITY": "SC",
}


def outcome_columns(name, plans=None):
    if name not in OUTCOME_BLOCK:
        raise ValueError("Unknown sensitivity outcome")
    return block_columns(OUTCOME_BLOCK[name], plans)



def fit_core7b_historical(
    X, target, columns, categorical, numeric,
    fit, validation, test, development,
):
    from catboost import CatBoostRegressor, Pool
    from sklearn.model_selection import train_test_split
    from wsqs.models import (
        prepare_catboost, summary_metrics, model_settings
    )

    columns = list(columns)

    if (
        columns != scientific_plans()["full"]
        or len(columns) != 33
    ):
        raise ValueError("CORE7B requires 33 SCP predictors")

    trainval, historical_test = train_test_split(
        np.arange(len(X)),
        test_size=0.20,
        random_state=20260714,
    )

    historical_fit, historical_val = train_test_split(
        trainval,
        test_size=0.20,
        random_state=20260714,
    )

    if not (
        np.array_equal(historical_test, test)
        and np.array_equal(historical_fit, fit)
        and np.array_equal(historical_val, validation)
        and np.array_equal(
            np.sort(trainval),
            np.sort(development),
        )
    ):
        raise ValueError("Historical partition mismatch")

    cats = [c for c in columns if c in categorical]
    nums = [c for c in columns if c in numeric]

    if len(cats) + len(nums) != 33:
        raise ValueError("Incomplete CORE7B feature types")

    prepared = prepare_catboost(
        X[columns], cats, nums
    )

    params = dict(model_settings()["catboost"])
    params.pop("od_type", None)
    params.pop("od_wait", None)
    params.update(
        verbose=False,
        allow_writing_files=False,
    )

    selector = CatBoostRegressor(**params)
    selector.fit(
        Pool(
            prepared.iloc[fit],
            target.iloc[fit],
            cat_features=cats,
        ),
        eval_set=Pool(
            prepared.iloc[validation],
            target.iloc[validation],
            cat_features=cats,
        ),
        use_best_model=True,
        verbose=False,
    )

    trees = int(selector.tree_count_)

    if not 1 <= trees <= 1400:
        raise RuntimeError("Invalid selected iterations")

    final_params = dict(params, iterations=trees)
    model = CatBoostRegressor(**final_params)

    model.fit(
        Pool(
            prepared.iloc[trainval],
            target.iloc[trainval],
            cat_features=cats,
        ),
        verbose=False,
    )

    prediction = model.predict(
        Pool(prepared.iloc[test], cat_features=cats)
    )

    return {
        **summary_metrics(target.iloc[test], prediction),
        "selected_iterations": trees,
        "heldout_n": len(test),
        "fit_n": len(fit),
        "validation_n": len(validation),
    }


def block_shapley(r2):
    """Order-independent three-block Shapley R² allocation."""

    expected = {"", *BLOCK_KEYS}

    if set(r2) != expected:
        raise ValueError("All eight coalition values are required")

    result = {}

    for block in BLOCK_ORDER:
        others = [
            item for item in BLOCK_ORDER
            if item != block
        ]

        allocation = 0.0

        for n in range(3):
            for coalition in combinations(others, n):
                before = "".join(
                    item for item in BLOCK_ORDER
                    if item in coalition
                )

                after = "".join(
                    item for item in BLOCK_ORDER
                    if item in set(coalition) | {block}
                )

                coefficient = (
                    math.factorial(n)
                    * math.factorial(2 - n)
                    / math.factorial(3)
                )

                allocation += (
                    coefficient * (r2[after] - r2[before])
                )

        result[block] = allocation

    improvement = r2["SCP"] - r2[""]

    if not np.isclose(
        sum(result.values()),
        improvement,
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Shapley allocation does not reconcile")

    return result


def domain_severities(frame):
    required = [
        f"{domain}_raw_severity"
        for domain in DOMAINS
    ]

    missing = [
        column for column in required
        if column not in frame
    ]

    if missing:
        raise ValueError("Missing CORE-8 severity columns")

    raw = frame[required].apply(
        pd.to_numeric, errors="coerce"
    )

    raw.columns = list(DOMAINS)

    if not raw.stack().between(1, 5).all():
        raise ValueError("Invalid domain severity")

    if not raw.notna().sum(axis=1).ge(7).all():
        raise ValueError("Unexpected domain availability")

    return raw


def household_priorities(frame):
    names = [
        f"{domain}_priority_0_10"
        for domain in DOMAINS
    ]

    if any(name not in frame for name in names):
        raise ValueError("Missing priority ratings")

    priorities = frame[names].apply(
        pd.to_numeric, errors="coerce"
    )

    priorities.columns = list(DOMAINS)

    if (
        not priorities.notna().all().all()
        or not priorities.stack().between(0, 10).all()
    ):
        raise ValueError("Invalid household priority ratings")

    return priorities


def alternative_outcomes(frame, development):
    """Original sensitivity definitions and training-only weights."""

    raw = domain_severities(frame)
    priority = household_priorities(frame)

    development = np.asarray(development, dtype=int)

    if (
        development.size == 0
        or len(np.unique(development)) != len(development)
        or development.min() < 0
        or development.max() >= len(frame)
    ):
        raise ValueError("Invalid development partition")

    equal = raw.mean(axis=1)

    core7a = raw.drop(
        columns="Acceptability"
    ).mean(axis=1)

    core7b = raw.drop(
        columns="Customer_service"
    ).mean(axis=1)

    weights = priority.iloc[development].mean(axis=0)

    if weights.sum() <= 0:
        raise ValueError("Invalid population priority weights")

    weights = weights / weights.sum()

    available = raw.notna()

    population_num = raw.mul(
        weights, axis=1
    ).sum(axis=1)

    population_den = available.mul(
        weights, axis=1
    ).sum(axis=1)

    population = population_num.div(
        population_den.where(population_den > 0)
    )

    effective = priority.where(available)

    individual_den = effective.sum(axis=1)

    individual = (
        raw.mul(effective)
        .sum(axis=1)
        .div(individual_den.where(individual_den > 0))
    )

    fallback = individual_den.le(0)

    individual = individual.where(~fallback, equal)

    severity_means = {
        "CORE8": equal,
        "CORE7A": core7a,
        "CORE7B": core7b,
        "POPULATION_PRIORITY": population,
        "INDIVIDUAL_PRIORITY": individual,
    }

    outcomes = {
        key: 25.0 * (5.0 - value)
        for key, value in severity_means.items()
    }

    for key, score in outcomes.items():
        if (
            not score.notna().all()
            or not score.between(0, 100).all()
        ):
            raise ValueError("Invalid outcome: " + key)

    return outcomes, weights, int(fallback.sum())


def supplementary_features(frame, X, categorical, numeric):
    """Four historically defined feature sensitivities."""

    plans = scientific_plans()

    variants = {}

    income = (
        frame["Q 15.17_raw_value"]
        .astype("string")
        .str.strip()
    )

    income = income.mask(income.eq(""), pd.NA)

    income_X = X.copy()
    income_X["context_reported_income"] = (
        income.fillna("__UNANSWERED__").astype(str)
    )

    variants["income"] = (
        income_X,
        list(categorical) + ["context_reported_income"],
        list(numeric),
    )

    pressure_column = plans["pressure_feature"]

    pressure = (
        frame["Q 4.2_raw_value"]
        .astype("string")
        .str.strip()
    )

    if pressure.isna().any():
        raise ValueError(
            "Broader pressure response contains missing values"
        )

    pressure_X = X.copy()
    pressure_X[pressure_column] = (
        pressure.fillna(MISSING_CATEGORY).astype(str)
    )

    variants["pressure"] = (
        pressure_X,
        list(categorical),
        list(numeric),
    )

    age = pd.to_numeric(
        frame["Q 15.3_raw_value"]
        .astype("string").str.strip(),
        errors="coerce",
    )

    age = age.clip(lower=18).mask(age > 100)

    age_X = X.copy()
    age_X["context_respondent_age"] = age.astype(float)

    variants["age"] = (
        age_X,
        list(categorical),
        list(numeric) + ["context_respondent_age"],
    )

    gender = (
        frame["Q 15.2_raw_value"]
        .astype("string").str.strip().str.lower()
    )

    gender_map = {
        "female": "Female",
        "male": "Male",
        "other": "Other",
        "others": "Other",
        "choose not to specify": "Not disclosed",
    }

    unexpected = set(gender.dropna()) - set(gender_map)

    if unexpected:
        raise ValueError(
            "Unexpected gender coding requires review"
        )

    gender_X = X.copy()
    gender_X["context_respondent_gender"] = (
        gender.map(gender_map)
        .fillna(MISSING_CATEGORY)
        .astype(str)
    )

    variants["gender"] = (
        gender_X,
        list(categorical) + ["context_respondent_gender"],
        list(numeric),
    )

    for name, (extended, cats, nums) in variants.items():
        expected_size = 33 if name == "pressure" else 34

        if extended.shape != (len(X), expected_size):
            raise ValueError(
                "Sensitivity feature dimensions changed: " + name
            )

        if set(cats) & set(nums):
            raise ValueError("Overlapping predictor types")

        if set(cats) | set(nums) != set(extended.columns):
            raise ValueError("Missing predictor type assignment")

    return variants


def fit_one(
    X, target, columns, categorical, numeric,
    fit, validation, test, development,
):
    """Fit only on development observations."""

    columns = list(columns)

    cat = [
        name for name in columns
        if name in categorical
    ]

    num = [
        name for name in columns
        if name in numeric
    ]

    if len(cat) + len(num) != len(columns):
        raise ValueError("Incomplete feature typing")

    prepared = prepare_catboost(
        X[columns], cat, num
    )

    predicted, iterations = catboost_predictions(
        prepared,
        target,
        cat,
        fit,
        validation,
        test,
        development,
        model_settings(),
    )

    return {
        **summary_metrics(
            target.iloc[test],
            predicted,
        ),
        "selected_iterations": int(iterations),
        "heldout_n": len(test),
        "fit_n": len(fit),
        "validation_n": len(validation),
    }


def reproduce_extended(frame, X, categorical, numeric, progress=print):
    """Aggregate-only empirical reproduction of revised analyses."""

    plans = scientific_plans()

    if (
        len(frame) != 6145
        or X.shape != (6145, 33)
        or list(X.columns) != plans["full"]
    ):
        raise ValueError("Authoritative study specification mismatch")

    fit, val, test, dev = partition_indices(
        len(frame), 20260714
    )

    outcomes, weights, fallback = alternative_outcomes(
        frame, dev
    )

    primary = outcomes["CORE8"].reset_index(drop=True)

    blocks = []
    r2 = {}

    baseline = np.full(
        len(test), primary.iloc[dev].mean()
    )

    baseline_metrics = summary_metrics(
        primary.iloc[test], baseline
    )

    r2[""] = baseline_metrics["r2"]

    blocks.append({
        "block": "Mean baseline",
        **baseline_metrics,
        "selected_iterations": np.nan,
        "heldout_n": len(test),
    })

    for key in BLOCK_KEYS:
        columns = block_columns(key, plans)

        metrics = fit_one(
            X, primary, columns, categorical, numeric,
            fit, val, test, dev,
        )

        r2[key] = metrics["r2"]

        blocks.append({
            "block": key,
            **metrics,
        })

        progress(
            f"Block {key} | RMSE={metrics['rmse_100']:.5f} "
            f"| R2={metrics['r2']:.5f}"
        )

    allocations = block_shapley(r2)

    improvement = r2["SCP"] - r2[""]

    shapley_table = pd.DataFrame([
        {
            "block": key,
            "allocated_r2": allocations[key],
            "share_percent": (
                100 * allocations[key] / improvement
                if improvement != 0 else np.nan
            ),
        }
        for key in BLOCK_ORDER
    ])

    sensitivity_rows = []

    for name in (
        "CORE7A",
        "CORE7B",
        "POPULATION_PRIORITY",
        "INDIVIDUAL_PRIORITY",
    ):
        target = outcomes[name].reset_index(drop=True)
        columns = outcome_columns(name, plans)

        if name == "CORE7B":
            metrics = fit_core7b_historical(
                X, target, columns, categorical, numeric,
                fit, val, test, dev,
            )
        else:
            metrics = fit_one(
                X, target, columns, categorical, numeric,
                fit, val, test, dev,
            )

        sensitivity_rows.append({
            "specification": name,
            "family": "outcome",
            "predictor_block": OUTCOME_BLOCK[name],
            "predictor_count": len(columns),
            **metrics,
        })

        progress(
            f"Outcome {name} | RMSE={metrics['rmse_100']:.5f}"
        )

    variants = supplementary_features(
        frame, X, categorical, numeric
    )

    for name, (extended, cats, nums) in variants.items():
        metrics = fit_one(
            extended, primary,
            extended.columns, cats, nums,
            fit, val, test, dev,
        )

        sensitivity_rows.append({
            "specification": name,
            "family": "feature",
            **metrics,
        })

        progress(
            f"Feature {name} | RMSE={metrics['rmse_100']:.5f}"
        )

    domain_rows = []
    raw = domain_severities(frame)

    for domain in DOMAINS:
        target = 25 * (5 - raw[domain])

        dfit = fit[target.iloc[fit].notna().to_numpy()]
        dval = val[target.iloc[val].notna().to_numpy()]
        dtest = test[target.iloc[test].notna().to_numpy()]
        ddev = np.sort(np.r_[dfit, dval])

        if min(len(dfit), len(dval), len(dtest)) == 0:
            raise ValueError(
                "No eligible observations for domain " + domain
            )

        columns = plans["cross_domain"][domain]

        metrics = fit_one(
            X,
            target.reset_index(drop=True),
            columns,
            categorical,
            numeric,
            dfit, dval, dtest, ddev,
        )

        domain_rows.append({
            "domain": domain,
            **metrics,
        })

        progress(
            f"Cross-domain {domain} | n={len(dtest)} "
            f"| R2={metrics['r2']:.5f}"
        )

    metadata = {
        "households": len(frame),
        "predictors": X.shape[1],
        "selected_split_seed": 20260714,
        "individual_priority_fallback_n": fallback,
        "population_weights": weights.to_dict(),
        "household_level_outputs": False,
        "interpretation": "Predictive, not causal",
        "outcome_predictor_blocks": dict(OUTCOME_BLOCK),
    }

    return {
        "block_models": pd.DataFrame(blocks),
        "block_shapley": shapley_table,
        "sensitivity_models": pd.DataFrame(sensitivity_rows),
        "cross_domain_models": pd.DataFrame(domain_rows),
    }, metadata
