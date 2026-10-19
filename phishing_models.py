"""Model definitions shared by train.py and app.py."""
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

# Groups used in the report and the UI.
CLASSIC_MODELS = ["Logistic Regression", "Decision Tree", "Random Forest"]
ADVANCED_MODELS = ["XGBoost", "CatBoost"]

# Dropped from the project; filtered out of older saved bundles.
REMOVED_MODELS = (
    "Gradient Boosting",
    "LightGBM",
    "MLP (Neural Network)",
)


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """Wrap an estimator that needs integer labels (e.g. XGBoost) so that the
    rest of the project can keep using string labels ("Phishing"/"Legitimate")."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.estimator_ = clone(self.estimator).fit(X, self.encoder_.transform(y))
        return self

    def predict(self, X):
        return self.encoder_.inverse_transform(self.estimator_.predict(X).astype(int))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def active_model_names():
    """Canonical list of models used by training and the UI."""
    return CLASSIC_MODELS + ADVANCED_MODELS


def filter_models(models):
    """Keep only active models (drops removed ones from older bundles)."""
    allowed = set(active_model_names())
    return {name: model for name, model in models.items() if name in allowed}


def build_models():
    """Return an ordered dict of unfitted models (classic baselines + modern models)."""
    # Imported here so the app can load saved models even if only some libraries exist.
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier

    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                         random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": LabelEncodedClassifier(XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)),
        "CatBoost": CatBoostClassifier(
            iterations=500, learning_rate=0.1, depth=6, random_seed=RANDOM_STATE, verbose=0,
            allow_writing_files=False),  # do not create a catboost_info/ folder
    }
