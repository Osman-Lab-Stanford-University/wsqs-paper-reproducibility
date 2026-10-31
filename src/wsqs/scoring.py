"""WSQS aggregation from harmonized item severities.

Each input must already have been classified using the
questionnaire coding rules.

Missing and structurally inapplicable values are represented
as None. Higher WSQS values indicate better water service.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

CORE8 = (
    ("Q2.2", "Affordability"),
    ("Q3.1", "Access"),
    ("Q4.1", "Reliability"),
    ("Q5.1", "Efficiency"),
    ("Q6.1", "Acceptability"),
    ("Q7.1", "Quality"),
    ("Q8.1", "Quantity"),
    ("Q9.1", "Customer service"),
)


def wsqs_from_severities(
    values: Mapping[str, object],
    excluded: tuple[str, ...] = (),
) -> float:
    """Average valid applicable domain scores on a 0–100 scale."""

    allowed = {qid for qid, _ in CORE8}

    if set(values) - allowed:
        raise ValueError("Unknown CORE item identifier")

    scores = []

    for qid, _ in CORE8:
        if qid in excluded:
            continue

        severity = values.get(qid)

        if severity is None:
            continue

        if isinstance(severity, bool):
            raise ValueError(f"Invalid severity for {qid}")

        try:
            number = float(severity)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid severity for {qid}"
            ) from None

        if (
            not math.isfinite(number)
            or number not in (1, 2, 3, 4, 5)
        ):
            raise ValueError(
                f"Invalid severity for {qid}"
            )

        scores.append(25.0 * (5.0 - number))

    if not scores:
        raise ValueError("No valid applicable CORE items")

    return sum(scores) / len(scores)


def core8(values: Mapping[str, object]) -> float:
    """Equal-weight eight-domain WSQS."""
    return wsqs_from_severities(values)


def core7a(values: Mapping[str, object]) -> float:
    """Exclude Acceptability, Q6.1."""
    return wsqs_from_severities(
        values, excluded=("Q6.1",)
    )


def core7b(values: Mapping[str, object]) -> float:
    """Exclude Customer Service, Q9.1."""
    return wsqs_from_severities(
        values, excluded=("Q9.1",)
    )
