"""
Evaluates the results in results/ against my manually labelled sets.

Labels were made BEFORE building the system (see WRITEUP.md):
    query 1  - all 26 Romanian companies
    query 6  - all 43 Swiss companies
    query 12 - 69 battery-related candidates
"Borderline" companies are excluded from the scores: even my own label is uncertain.
"""

import os

import pandas as pd

RESULTS_DIR = "results"

LABELS = {
    1: {
        "yes": [15],                               # Posta Romana
        "borderline": [12, 19],                    # Romgaz (gas + storage), Valbur (trader + transport)
    },
    6: {
        "yes": [443, 444, 445, 447, 448, 449, 450, 451, 453, 454, 455, 456, 457, 458,
                459, 460, 461, 463, 464, 466, 467, 468, 470, 472, 473, 474, 475, 476],
        "borderline": [],
    },
    12: {
        "yes": [330, 334, 336, 339, 340, 341, 342, 343, 344, 345, 346, 349, 350, 351,
                352, 353, 354, 360, 361, 362, 363, 367, 369, 370, 371, 372, 373, 375,
                377,                                           # materials / components
                332, 333, 337, 347, 348, 357, 358, 359, 365, 366, 368, 374, 378,  # EV cells
                376],                                          # ReFire_ev
        "borderline": [331, 335, 356, 379, 338, 364],
    },
}


def evaluate(query_id, labels):
    path = os.path.join(RESULTS_DIR, f"query_{query_id}.csv")
    result = pd.read_csv(path, index_col=0)
    predicted = set(result[result.qualified].index) - set(labels["borderline"])
    truth = set(labels["yes"])

    found = predicted & truth            # correct companies the system returned
    missed = truth - predicted           # correct companies the system did not return
    wrong = predicted - truth            # companies returned that should not be

    precision = len(found) / len(predicted) if predicted else 0.0
    recall = len(found) / len(truth) if truth else 0.0

    print(f"Query {query_id}: found {len(found)} | missed {len(missed)} | wrong {len(wrong)}"
          f" | precision {precision:.0%} | recall {recall:.0%}")
    if missed:
        print(f"   missed: {sorted(missed)}")
    if wrong:
        print(f"   wrong:  {sorted(wrong)}")


if __name__ == "__main__":
    for qid, labels in LABELS.items():
        evaluate(qid, labels)
