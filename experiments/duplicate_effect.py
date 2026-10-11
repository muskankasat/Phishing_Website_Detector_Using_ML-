"""Measure how duplicate rows inflate test accuracy on the UCI Phishing dataset.

Same split settings as train.py, run twice: on the raw 11,055 rows (duplicates kept)
and on the de-duplicated 5,849 rows. Output: results/duplicate_effect.csv
Run from the project root: python -m experiments.duplicate_effect
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from phishing_models import RANDOM_STATE, build_models

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "phishing_websites_raw.csv"


def evaluate(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    out = {}
    for name, model in build_models().items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        out[name] = (accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro"))
    return out, len(X_te)


def main():
    raw = pd.read_csv(RAW)
    y_raw = raw.pop("Result").map({-1: "Phishing", 1: "Legitimate"})
    X_raw = raw

    full = X_raw.copy()
    full["y"] = y_raw
    # How many test-style rows have an exact twin elsewhere in the data?
    dup_share = full.duplicated(keep=False).mean()
    dedup = full.drop_duplicates().reset_index(drop=True)
    y_dd = dedup.pop("y")

    with_dups, n1 = evaluate(X_raw, y_raw)
    without, n2 = evaluate(dedup, y_dd)
    rows = [{
        "Model": m,
        "Accuracy (duplicates kept)": with_dups[m][0],
        "Accuracy (de-duplicated)": without[m][0],
        "Drop (points)": (with_dups[m][0] - without[m][0]) * 100,
    } for m in with_dups]
    df = pd.DataFrame(rows).sort_values("Accuracy (duplicates kept)", ascending=False)
    df.to_csv(ROOT / "results" / "duplicate_effect.csv", index=False)
    print(f"Raw rows: {len(full)} | rows with an exact twin: {dup_share:.1%} | "
          f"test sizes: {n1} vs {n2}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
