"""Original response coding for the CORE-8 questionnaire items."""

import csv
from pathlib import Path

from .scoring import core8, core7a, core7b


def coding():
    path = (
        Path(__file__).resolve().parents[2]
        / "questionbook/core8_response_coding.csv"
    )

    with path.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    mapping = {}

    for row in records:
        key = (
            row["Question_ID"],
            row["Original_response_label"],
        )

        if key in mapping:
            raise ValueError(f"Duplicate questionnaire response: {key}")

        if (
            row["Response_classification"]
            == "Scored substantive response"
        ):
            mapping[key] = int(
                row["Final_severity_1_best_5_worst"]
            )
        else:
            mapping[key] = None

    return mapping


def score_responses(row):
    mapping = coding()

    qids = (
        "Q2.2", "Q3.1", "Q4.1", "Q5.1",
        "Q6.1", "Q7.1", "Q8.1", "Q9.1",
    )

    values = {}

    for q in qids:
        response = row.get(q)

        if response is None or response == "":
            values[q] = None
            continue

        key = (q, str(response))

        if key not in mapping:
            raise ValueError(
                f"Unrecognized response for {q}: {response!r}"
            )

        values[q] = mapping[key]

    return {
        "CORE8": core8(values),
        "CORE7A": core7a(values),
        "CORE7B": core7b(values),
    }
