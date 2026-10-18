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
from url_features import extract_features

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


def render_prediction(models, selected, feature_names, input_values, compare_all, actual_label=None):
    input_df = pd.DataFrame([[input_values[f] for f in feature_names]], columns=feature_names)
    model = models[selected]
    prediction = model.predict(input_df)[0]
    st.divider()
    left, right = st.columns(2)
    with left:
        if prediction == "Phishing":
            st.error("Prediction: PHISHING")
            st.write("The model classifies this site as phishing based on extracted features.")
        else:
            st.success("Prediction: LEGITIMATE")
            st.write("The model classifies this site as legitimate based on extracted features.")
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


st.title("🛡️ Explainable Phishing Website Risk Detector")
st.caption("Academic demonstration using the UCI Phishing Websites dataset")
st.warning(
    "Paste a URL to extract features and classify it. Some signals (WHOIS age, "
    "Alexa traffic, PageRank, Google Index) are approximated because those services "
    "are unavailable. This is an educational prototype — not a security guarantee."
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
    st.subheader("Check a URL")
    source = st.radio(
        "Input method",
        ["Paste a URL", "Use a test-set example", "Enter feature values"],
        horizontal=True,
    )

    actual_label = None
    input_values = {}
    extraction_info = None

    if source == "Paste a URL":
        url_text = st.text_input(
            "Website URL",
            placeholder="https://example.com/login",
            help="http:// and https:// links only. The app extracts UCI-style features, then runs the trained model.",
        )
        fetch_page = st.checkbox(
            "Fetch page HTML for richer features (anchors, forms, iframes, redirects)",
            value=True,
        )
        st.caption(
            "With fetching enabled, the app briefly downloads the page HTML (size-limited, timed out). "
            "Disable it to classify from the URL string alone."
        )
    elif source == "Use a test-set example":
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

    analyze_label = "Analyze URL" if source == "Paste a URL" else "Analyze record"
    if st.button(analyze_label, type="primary"):
        if source == "Paste a URL":
            if not url_text.strip():
                st.error("Enter a URL first.")
                st.stop()
            try:
                with st.spinner("Extracting features from URL..."):
                    extraction_info = extract_features(url_text, fetch_page=fetch_page)
                input_values = extraction_info["features"]
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

            st.write(f"Normalized URL: `{extraction_info['url']}`")
            if extraction_info["fetched"]:
                st.success("Page HTML fetched and used for HTML-based features.")
            elif fetch_page:
                st.warning(
                    "Could not fetch page HTML"
                    + (f": {extraction_info['fetch_error']}" if extraction_info["fetch_error"] else ".")
                    + " Classification uses URL-string features plus safe defaults."
                )
            if extraction_info["approximated"]:
                with st.expander("Approximated / defaulted features", expanded=False):
                    st.write(
                        "These features could not be measured exactly (missing page HTML "
                        "or retired external APIs) and used a neutral/suspicious default of 0, "
                        "or a heuristic:"
                    )
                    st.code(", ".join(extraction_info["approximated"]))

            with st.expander("Extracted feature vector", expanded=False):
                st.dataframe(
                    pd.DataFrame(
                        [{"Feature": k, "Value": v} for k, v in input_values.items()]
                    ),
                    hide_index=True,
                    width="stretch",
                )

        render_prediction(
            models, selected, feature_names, input_values, compare_all, actual_label
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
    - Models were trained on UCI's engineered features; URL mode *approximates* those features.
    - WHOIS age, Alexa traffic, PageRank, Google Index and backlink counts are defaulted when unavailable.
    - Performance may not generalize to current websites or new phishing campaigns.
    - Feature importance describes model behavior and does not establish causality.
    """)
    st.markdown(
        "[Dataset and documentation: UCI Phishing Websites]"
        "(https://archive.ics.uci.edu/dataset/327/phishing+websites)"
    )
