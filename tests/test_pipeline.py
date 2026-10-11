"""Smoke tests. Run `python train.py` first, then `pytest`."""
from pathlib import Path

import joblib
import pandas as pd
import pytest

import phishing_models  # noqa: F401  (needed to unpickle the saved models)
from phishing_models import ADVANCED_MODELS, CLASSIC_MODELS, LabelEncodedClassifier, build_models

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "models" / "model_bundle.joblib"
TEST_CSV = ROOT / "data" / "test_samples.csv"
RAW = ROOT / "data" / "raw" / "phishing_websites_raw.csv"

needs_artifacts = pytest.mark.skipif(not BUNDLE.exists(), reason="run python train.py first")


def test_build_models_contains_all_expected():
    names = set(build_models())
    assert names == set(CLASSIC_MODELS + ADVANCED_MODELS)


def test_label_wrapper_keeps_string_labels():
    X = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1, 0, 1], "b": [1, 1, 0, 0, 1, 1, 0, 0]})
    y = pd.Series(["Phishing", "Legitimate"] * 4)
    model = build_models()["XGBoost"].fit(X, y)
    assert isinstance(model, LabelEncodedClassifier)
    assert set(model.predict(X)) <= {"Phishing", "Legitimate"}
    assert list(model.classes_) == ["Legitimate", "Phishing"]
    assert model.predict_proba(X).shape == (8, 2)


def test_raw_dataset_shape():
    raw = pd.read_csv(RAW)
    assert raw.shape == (11055, 31)
    assert set(raw["Result"].unique()) == {-1, 1}


@needs_artifacts
def test_bundle_has_all_models_and_30_features():
    bundle = joblib.load(BUNDLE)
    assert set(bundle["models"]) == set(CLASSIC_MODELS + ADVANCED_MODELS)
    assert len(bundle["feature_names"]) == 30


@needs_artifacts
def test_predictions_valid_for_every_model():
    bundle = joblib.load(BUNDLE)
    sample = pd.read_csv(TEST_CSV).head(20)
    X = sample[bundle["feature_names"]]
    for name, model in bundle["models"].items():
        preds = model.predict(X)
        assert set(preds) <= {"Phishing", "Legitimate"}, name
        probs = model.predict_proba(X)
        assert probs.shape == (20, 2) and (probs.sum(axis=1) > 0.99).all(), name


@needs_artifacts
def test_no_train_test_overlap_in_cleaned_data():
    cleaned = pd.read_csv(ROOT / "data" / "phishing_websites_cleaned.csv")
    assert not cleaned.duplicated().any()
    test = pd.read_csv(TEST_CSV)
    assert len(test) == pytest.approx(0.2 * len(cleaned), abs=2)


@needs_artifacts
def test_streamlit_app_runs_without_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
