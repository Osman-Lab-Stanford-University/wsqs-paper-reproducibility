"""WSQS predictor construction from harmonized survey inputs."""

import json
from pathlib import Path
import pandas as pd


def feature_specification():
    path = (
        Path(__file__).resolve().parents[2]
        / "config/feature_spec.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def construct_features(df):
    spec = feature_specification()
    domains = spec["domains"]
    context = spec["context"]
    provider = set(spec["provider_items"])
    special = spec["special"]
    three = set(spec["native_three"])
    two = set(spec["native_two"])
    priorities = spec["priority_columns"]

    required = (
        [c for questions in domains.values()
         for c in questions.values()]
        + list(context.values())
        + priorities
        + ["municipal_supply_applicable"]
        + [v[0] for v in special.values()]
    )

    absent = sorted(set(required) - set(df.columns))
    if absent:
        raise ValueError(f"Missing source columns: {absent}")

    eligible = (
        df["municipal_supply_applicable"]
        .astype(str).str.strip().str.lower()
        .isin(("true", "1", "yes"))
    )

    X = pd.DataFrame(index=df.index)
    categorical = []
    numeric = []

    for domain, questions in domains.items():
        for q, source in questions.items():
            name = f"service_{q.replace('.', '_')}"

            if q in provider:
                score = pd.to_numeric(
                    df[source], errors="coerce"
                )
                observed = eligible & score.notna()

                allowed = (
                    {1, 2, 3} if q in three
                    else {1, 2, 3, 4, 5}
                )

                if not set(
                    score.loc[observed].unique()
                ).issubset(allowed):
                    raise ValueError(
                        f"Invalid provider ordinal values: {q}"
                    )

                values = pd.Series(
                    "__UNANSWERED__",
                    index=df.index,
                    dtype="object",
                )
                values.loc[~eligible] = (
                    "__OUTSIDE_MUNICIPAL_EVIDENCE__"
                )
                values.loc[observed] = (
                    score.loc[observed]
                    .astype(int)
                    .map(lambda v: f"severity_{v}")
                )

                if q in special:
                    raw, response, category = special[q]

                    mask = (
                        eligible
                        & df[raw].astype(str).str.strip().eq(response)
                    )

                    if not score.loc[mask].isna().all():
                        raise ValueError(
                            f"Special response improperly scored: {q}"
                        )

                    values.loc[mask] = category

                X[name] = values
                categorical.append(name)

            elif q == "Q2.8":
                values = df[source].astype("string").str.strip()

                X[name] = (
                    values.fillna("__UNANSWERED__")
                    .replace("", "__UNANSWERED__")
                    .astype(str)
                )
                categorical.append(name)

            else:
                values = pd.to_numeric(
                    df[source], errors="coerce"
                )

                allowed = (
                    {1, 2} if q in two
                    else {1, 2, 3} if q in three
                    else {1, 2, 3, 4, 5}
                )

                if not set(
                    values.dropna().unique()
                ).issubset(allowed):
                    raise ValueError(
                        f"Unexpected ordinal levels: {q}"
                    )

                X[name] = values
                numeric.append(name)

    for name in (
        "context_primary_provision",
        "context_dwelling",
    ):
        values = (
            df[context[name]]
            .astype("string")
            .str.strip()
        )

        X[name] = (
            values.fillna("__UNANSWERED__")
            .replace("", "__UNANSWERED__")
            .astype(str)
        )
        categorical.append(name)

    size = (
        df[context["context_household_size"]]
        .astype("string")
        .str.strip()
    )

    allowed_size = (
        {str(i) for i in range(1, 8)}
        | {"8 or more"}
    )

    if not (
        set(size.dropna().unique()) - {""}
    ).issubset(allowed_size):
        raise ValueError("Unexpected household-size category")

    X["context_household_size"] = (
        size.fillna("__UNANSWERED__")
        .replace({
            "": "__UNANSWERED__",
            "8 or more": "8_plus",
        })
        .astype(str)
    )
    categorical.append("context_household_size")

    for name in priorities:
        values = pd.to_numeric(
            df[name], errors="coerce"
        )

        if not (
            values.notna().all()
            and values.between(0, 10).all()
            and (values % 1 == 0).all()
        ):
            raise ValueError(
                f"Invalid stated-priority values: {name}"
            )

        X[name] = values
        numeric.append(name)

    ordered = (
        [
            f"service_{q.replace('.', '_')}"
            for questions in domains.values()
            for q in questions
        ]
        + list(context)
        + priorities
    )

    X = X.loc[:, ordered].copy()

    if (
        len(ordered) != 33
        or set(categorical).intersection(numeric)
        or set(categorical + numeric) != set(ordered)
    ):
        raise ValueError("Predictor specification mismatch")

    return X, categorical, numeric
