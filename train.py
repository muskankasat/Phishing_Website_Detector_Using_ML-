"""
Train and evaluate five classifiers on the UCI Phishing Websites dataset.
Run: python train.py

The script downloads the dataset through ucimlrepo, performs basic checks,
uses a stratified hold-out split, evaluates the models, and saves outputs.
"""
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from ucimlrepo import fetch_ucirepo

from phishing_models import ADVANCED_MODELS, CLASSIC_MODELS, RANDOM_STATE, build_models

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
for folder in (DATA_DIR, MODEL_DIR, RESULTS_DIR, FIGURES_DIR):
    folder.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.20
CV_FOLDS = 5


RAW_LOCAL = DATA_DIR / "raw" / "phishing_websites_raw.csv"


def fetch_raw():
    """Fetch the UCI dataset; fall back to a local raw CSV if offline."""
    try:
        dataset = fetch_ucirepo(id=327)  # UCI Phishing Websites
        return dataset.data.features.copy(), dataset.data.targets.copy(), "ucimlrepo (UCI id=327)"
    except Exception as exc:  # network blocked, server down, etc.
        if not RAW_LOCAL.exists():
            raise RuntimeError(
                f"Could not download from UCI ({exc}) and no local copy at {RAW_LOCAL}."
            ) from exc
        print(f"UCI download failed ({exc}); using local copy {RAW_LOCAL}")
        raw = pd.read_csv(RAW_LOCAL)
        target_col = "Result" if "Result" in raw.columns else raw.columns[-1]
        return raw.drop(columns=[target_col]), raw[[target_col]], f"local file {RAW_LOCAL.name}"


def load_dataset():
    """Load dataset; save a cleaned copy for reproducibility."""
    X, y, data_source = fetch_raw()

    # Normalize column names and remove accidental whitespace.
    X.columns = [str(c).strip().replace(" ", "_") for c in X.columns]
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = pd.Series(y, name="label").reset_index(drop=True)
    X = X.reset_index(drop=True)

    # Dataset convention: -1 = phishing, 1 = legitimate.
    # Convert to readable labels; fail loudly if the dataset differs.
    raw_labels = set(pd.Series(y).dropna().unique().tolist())
    if raw_labels.issubset({-1, 1, "-1", "1"}):
        y = y.astype(int).map({-1: "Phishing", 1: "Legitimate"})
    else:
        # If an upstream package returns text labels, normalize them.
        y_text = y.astype(str).str.strip().str.lower()
        mapping = {}
        for value in y_text.unique():
            if "phish" in value or value in {"-1", "0"}:
                mapping[value] = "Phishing"
            elif "legit" in value or value in {"1"}:
                mapping[value] = "Legitimate"
        if len(mapping) != len(y_text.unique()):
            raise ValueError(f"Unrecognized target labels: {sorted(y_text.unique())}")
        y = y_text.map(mapping)

    # Basic cleaning: numeric conversion, missing-value inspection, duplicates.
    X = X.apply(pd.to_numeric, errors="coerce")
    rows_before = len(X)
    missing_before = int(X.isna().sum().sum())
    duplicate_rows_before = int(X.duplicated().sum())

    # A simple and transparent baseline cleaning strategy:
    # remove rows with missing target; median-impute numeric feature gaps.
    valid = y.notna()
    X, y = X.loc[valid].copy(), y.loc[valid].copy()
    X = X.fillna(X.median(numeric_only=True))
    # If any all-missing column existed, remove it rather than invent values.
    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        X = X.drop(columns=all_missing)
    X = X.fillna(0)

    # Remove exact duplicate feature/label records.
    combined = X.copy()
    combined["__label__"] = y.values
    combined = combined.drop_duplicates().reset_index(drop=True)
    y = combined.pop("__label__")
    X = combined

    # Save cleaned dataset for inspection. This file can be large-ish but is small here.
    cleaned = X.copy()
    cleaned["label"] = y.values
    cleaned.to_csv(DATA_DIR / "phishing_websites_cleaned.csv", index=False)

    cleaning_summary = {
        "data_source": data_source,
        "rows_before_cleaning": rows_before,
        "feature_count": int(X.shape[1]),
        "missing_feature_values_before_cleaning": missing_before,
        "duplicate_feature_rows_before_cleaning": duplicate_rows_before,
        "rows_after_duplicate_removal": len(X),
        "class_distribution": {k: int(v) for k, v in y.value_counts().items()},
        "feature_names": list(X.columns),
        "target_mapping": {"-1": "Phishing", "1": "Legitimate"},
    }
    with open(RESULTS_DIR / "cleaning_summary.json", "w", encoding="utf-8") as f:
        json.dump(cleaning_summary, f, indent=2)
    return X, y


def positive_class_scores(model, X_test, positive_label="Phishing"):
    """Return scores for the phishing class where the model supports probabilities."""
    if not hasattr(model, "predict_proba"):
        return None
    classes = list(model.classes_)
    if positive_label not in classes:
        return None
    idx = classes.index(positive_label)
    return model.predict_proba(X_test)[:, idx]


def evaluate_model(name, model, X_test, y_test, train_seconds):
    """Compute hold-out metrics for one fitted model."""
    preds = model.predict(X_test)
    row = {
        "Model": name,
        "Type": "Advanced" if name in ADVANCED_MODELS else "Classic",
        "Accuracy": accuracy_score(y_test, preds),
        "Precision (Phishing)": precision_score(
            y_test, preds, pos_label="Phishing", zero_division=0),
        "Recall (Phishing)": recall_score(y_test, preds, pos_label="Phishing", zero_division=0),
        "F1-score (Phishing)": f1_score(y_test, preds, pos_label="Phishing", zero_division=0),
        "Macro F1": f1_score(y_test, preds, average="macro", zero_division=0),
        "Training Time (s)": train_seconds,
    }
    scores = positive_class_scores(model, X_test)
    if scores is not None:
        row["ROC-AUC (Phishing)"] = roc_auc_score((y_test == "Phishing").astype(int), scores)
    return preds, row


def save_confusion_matrix(name, y_test, preds):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, labels=["Phishing", "Legitimate"],
        display_labels=["Phishing", "Legitimate"], cmap="Blues", values_format="d", ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(FIGURES_DIR / f"confusion_matrix_{safe}.png", dpi=160)
    plt.close(fig)


def run_cross_validation(models, X_train, y_train):
    """Stratified k-fold CV on the TRAINING split only; the test set stays untouched."""
    scoring = {
        "Accuracy": "accuracy",
        "Recall (Phishing)": make_scorer(recall_score, pos_label="Phishing", zero_division=0),
        "Precision (Phishing)": make_scorer(precision_score, pos_label="Phishing", zero_division=0),
        "Macro F1": "f1_macro",
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        print(f"Cross-validating {name}...")
        res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        row = {"Model": name}
        for metric in scoring:
            vals = res[f"test_{metric}"]
            row[f"{metric} mean"] = vals.mean()
            row[f"{metric} std"] = vals.std()
        rows.append(row)
    cv_df = pd.DataFrame(rows).sort_values("Macro F1 mean", ascending=False)
    cv_df.to_csv(RESULTS_DIR / "cv_results.csv", index=False)

    ordered = cv_df.set_index("Model")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(ordered.index[::-1], ordered["Macro F1 mean"][::-1],
            xerr=ordered["Macro F1 std"][::-1], color="#4C78A8", capsize=4)
    ax.set_xlim(0.88, 1.0)
    ax.set_xlabel(f"Macro F1 ({CV_FOLDS}-fold CV on training split, mean +/- std)")
    ax.set_title("Cross-validated model comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cv_comparison.png", dpi=180)
    plt.close(fig)
    return cv_df


def save_comparison_chart(metrics):
    cols = ["Accuracy", "Precision (Phishing)", "Recall (Phishing)",
            "F1-score (Phishing)", "Macro F1"]
    ax = metrics.set_index("Model")[cols].plot(kind="bar", figsize=(13, 6), ylim=(0.8, 1))
    ax.set_title("Model Performance Comparison (held-out test set)")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=180)
    plt.close()


def save_feature_importance(rf, columns):
    """Random Forest feature importance: predictive importance, not causation."""
    importances = pd.Series(rf.feature_importances_, index=columns).sort_values(ascending=False)
    importances.rename("importance").to_csv(RESULTS_DIR / "feature_importance.csv")
    top = importances.head(15).sort_values()
    plt.figure(figsize=(9, 6))
    top.plot(kind="barh")
    plt.title("Top 15 Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=180)
    plt.close()


def main():
    print("Loading dataset...")
    X, y = load_dataset()
    print(f"Dataset shape: {X.shape}; labels: {y.value_counts().to_dict()}")

    # Split BEFORE model fitting. Stratification preserves class proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    test_export = X_test.copy()
    test_export["label"] = y_test.values
    test_export.to_csv(DATA_DIR / "test_samples.csv", index=False)

    models = build_models()
    metric_rows, fitted = [], {}
    for name, model in models.items():
        print(f"Training {name}...")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        preds, row = evaluate_model(name, model, X_test, y_test, train_seconds)
        metric_rows.append(row)
        fitted[name] = model
        print(classification_report(y_test, preds, zero_division=0))
        save_confusion_matrix(name, y_test, preds)

    metrics = pd.DataFrame(metric_rows).sort_values("Macro F1", ascending=False)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print("\nModel comparison (hold-out test set):")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    save_comparison_chart(metrics)
    save_feature_importance(fitted["Random Forest"], X.columns)

    cv_df = run_cross_validation(build_models(), X_train, y_train)
    print(f"\n{CV_FOLDS}-fold cross-validation (training split):")
    print(cv_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    bundle = {
        "models": fitted,
        "feature_names": list(X.columns),
        "labels": ["Phishing", "Legitimate"],
        "positive_label": "Phishing",
        "random_state": RANDOM_STATE,
        "classic_models": CLASSIC_MODELS,
        "advanced_models": ADVANCED_MODELS,
    }
    joblib.dump(bundle, MODEL_DIR / "model_bundle.joblib", compress=3)
    print(f"\nSaved models to {MODEL_DIR / 'model_bundle.joblib'}")
    print(f"Saved metrics and figures under {RESULTS_DIR}")
    print("NOTE: These metrics are hold-out test results, not guarantees for real-world URLs.")


if __name__ == "__main__":
    main()
