"""Model definitions shared by train.py and app.py."""
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

# Groups used in the report and the UI.
CLASSIC_MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
ADVANCED_MODELS = ["XGBoost", "LightGBM", "CatBoost", "MLP (Neural Network)"]


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


def build_models():
    """Return an ordered dict of unfitted models (classic baselines + modern models)."""
    # Imported here so the app can load saved models even if only some libraries exist.
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
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
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": LabelEncodedClassifier(XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.1, num_leaves=31,
            random_state=RANDOM_STATE, verbose=-1),
        "CatBoost": CatBoostClassifier(
            iterations=500, learning_rate=0.1, depth=6, random_seed=RANDOM_STATE, verbose=0),
        # Integer labels are required by MLP early stopping in recent scikit-learn.
        "MLP (Neural Network)": LabelEncodedClassifier(Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                    early_stopping=True, random_state=RANDOM_STATE)),
        ])),
    }
