"""
Streamlit UI for the Explainable Phishing Website Risk Detector.
Run training first: python train.py
Then run: streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import phishing_models  # noqa: F401  (required so joblib can unpickle the saved models)

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "models" / "model_bundle.joblib"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TEST_PATH = ROOT / "data" / "test_samples.csv"
DEFAULT_MODEL = "CatBoost"

st.set_page_config(page_title="Phishing Risk Detector", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


@st.cache_data
def read_csv(path):
    return pd.read_csv(path) if Path(path).exists() else None


def show_image(name, caption):
    path = FIGURES / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


st.title("🛡️ Explainable Phishing Website Risk Detector")
st.caption("Academic demonstration using the UCI Phishing Websites dataset")
st.warning(
    "This prototype uses pre-extracted website features, not a live URL scanner. "
    "It is an educational model and must not be treated as a security guarantee."
)

if not BUNDLE_PATH.exists():
    st.error("Trained models not found. In the project folder run: python train.py")
    st.stop()

bundle = load_bundle()
models = bundle["models"]
feature_names = bundle["feature_names"]

tab_predict, tab_perf, tab_evidence, tab_about = st.tabs(
    ["🔎 Predict", "📊 Model performance", "📚 Literature & evidence", "ℹ️ About"]
)

# ---------------------------------------------------------------- Predict
with tab_predict:
    st.subheader("Classify a feature record")
    st.write(
        "The UCI dataset contains 30 encoded features with values -1, 0 or 1. "
        "Pick a held-out example for a quick demo, or enter values manually."
    )
    source = st.radio(
        "Input method", ["Use a test-set example", "Enter feature values"], horizontal=True
    )

    actual_label = None
    input_values = {}
    if source == "Use a test-set example":
        test_df = read_csv(TEST_PATH)
        if test_df is None:
            st.error("Test samples are missing. Run python train.py first.")
            st.stop()
        chosen = st.selectbox(
            "Choose a test example",
            options=list(range(min(len(test_df), 100))),
            format_func=lambda i: f"Example {i + 1} (actual: {test_df.iloc[i]['label']})",
        )
        sample = test_df.iloc[chosen]
        input_values = {f: int(sample[f]) for f in feature_names}
        actual_label = str(sample["label"])
        st.info(f"Actual dataset label for this example: **{actual_label}**")
    else:
        with st.expander("Feature inputs (encoded values)", expanded=True):
            st.caption("Values are -1, 0 or 1. See the UCI feature documentation for meanings.")
            cols = st.columns(3)
            for idx, feature in enumerate(feature_names):
                with cols[idx % 3]:
                    input_values[feature] = st.selectbox(
                        feature, options=[-1, 0, 1], index=2, key=f"feature_{feature}"
                    )

    model_names = list(models)
    selected = st.selectbox(
        "Model",
        options=model_names,
        index=model_names.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_names else 0,
    )
    compare_all = st.checkbox("Show what all models predict for this record")

    if st.button("Analyze record", type="primary"):
        input_df = pd.DataFrame([[input_values[f] for f in feature_names]], columns=feature_names)
        model = models[selected]
        prediction = model.predict(input_df)[0]
        st.divider()
        left, right = st.columns(2)
        with left:
            if prediction == "Phishing":
                st.error("Prediction: PHISHING")
                st.write("The model classifies this feature record as phishing.")
            else:
                st.success("Prediction: LEGITIMATE")
                st.write("The model classifies this feature record as legitimate.")
        with right:
            probs = model.predict_proba(input_df)[0]
            st.dataframe(
                pd.DataFrame({"Class": model.classes_, "Model score": probs}),
                hide_index=True, width="stretch",
            )
            st.caption("Scores are model estimates, not calibrated real-world probabilities.")
        if compare_all:
            votes = pd.DataFrame(
                {"Model": name, "Prediction": m.predict(input_df)[0]} for name, m in models.items()
            )
            st.dataframe(votes, hide_index=True, width="stretch")
        if actual_label is not None:
            match = "Yes" if prediction == actual_label else "No"
            st.write(f"**Actual label:** {actual_label}")
            st.write(f"**Prediction matches this example's label:** {match}")
        st.caption(
            "A prediction is not proof that a website is safe or malicious. "
            "Verify suspicious sites independently."
        )

# ------------------------------------------------------------ Performance
with tab_perf:
    st.subheader("Evaluation on the held-out test set")
    metrics = read_csv(RESULTS / "metrics.csv")
    if metrics is None:
        st.info("Metrics not found. Run python train.py first.")
    else:
        st.dataframe(metrics.round(4), hide_index=True, width="stretch")
        chart_cols = [
            c for c in ["Accuracy", "Precision (Phishing)", "Recall (Phishing)",
                        "F1-score (Phishing)", "Macro F1"] if c in metrics.columns
        ]
        st.bar_chart(metrics.set_index("Model")[chart_cols])

    st.subheader("5-fold cross-validation (training split)")
    cv = read_csv(RESULTS / "cv_results.csv")
    if cv is not None:
        st.dataframe(cv.round(4), hide_index=True, width="stretch")
        st.caption(
            "Top models overlap within one standard deviation, so small differences "
            "between them should not be over-interpreted."
        )
        show_image("cv_comparison.png", "Macro F1 with standard deviation across folds")

    st.subheader("Confusion matrix")
    pick = st.selectbox("Model", options=list(models), key="cm_model")
    safe = pick.lower().replace(" ", "_").replace("(", "").replace(")", "")
    show_image(f"confusion_matrix_{safe}.png", f"Confusion matrix: {pick}")
    show_image("feature_importance.png", "Random Forest feature importance (not causation)")
    st.markdown(
        "**Interpretation tip:** for phishing detection, inspect phishing recall and "
        "false negatives, not accuracy alone."
    )

# ------------------------------------------------------ Literature/evidence
with tab_evidence:
    st.subheader("Comparison with published work")
    lit = read_csv(RESULTS / "literature_comparison.csv")
    if lit is not None:
        st.dataframe(
            lit, hide_index=True, width="stretch",
            column_config={"URL": st.column_config.LinkColumn("URL")},
        )
    st.subheader("Why published scores look higher: duplicate rows")
    dup = read_csv(RESULTS / "duplicate_effect.csv")
    if dup is not None:
        st.dataframe(dup.round(4), hide_index=True, width="stretch")
        st.write(
            "The raw dataset has many exact duplicate rows. When duplicates are kept, "
            "identical records can land in both training and test sets, inflating accuracy. "
            "This project removes them before splitting, which is stricter but more honest."
        )
    st.caption(
        "Published results use different splits and preprocessing, so numbers are "
        "indicative rather than a like-for-like benchmark."
    )

# ------------------------------------------------------------------ About
with tab_about:
    st.subheader("Project workflow")
    st.markdown("""
    1. Download the public UCI dataset (offline copy included).
    2. Inspect data types, missing values, duplicates and class balance.
    3. Remove duplicates, then split into training and test data with stratification.
    4. Train 8 models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting,
       XGBoost, LightGBM, CatBoost and an MLP neural network.
    5. Compare on the same held-out test set and with 5-fold cross-validation.
    6. Inspect confusion matrices and Random Forest feature importance.
    """)
    st.markdown("**Limitations**")
    st.markdown("""
    - The dataset contains engineered features, not raw URLs.
    - Performance may not generalize to current websites or new phishing campaigns.
    - The app does not fetch websites, open links, or verify live site safety.
    - Feature importance describes model behavior and does not establish causality.
    """)
    st.markdown(
        "[Dataset and documentation: UCI Phishing Websites]"
        "(https://archive.ics.uci.edu/dataset/327/phishing+websites)"
    )
