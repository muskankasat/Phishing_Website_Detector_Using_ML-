"""
Streamlit UI for the Explainable Phishing Website Risk Detector.
Run training first: python train.py
Then run: streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import phishing_models  # noqa: F401  (required so joblib can unpickle)
from url_features import extract_features

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "models" / "model_bundle.joblib"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TEST_PATH = ROOT / "data" / "test_samples.csv"
DEFAULT_MODEL = "CatBoost"

st.set_page_config(
    page_title="Phishing Risk Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Connect4-inspired shell: pure black, bold type, yellow/emerald accents.
st.markdown(
    """
<style>
@import url("https://cdn.jsdelivr.net/npm/geist@1.3.1/dist/fonts/geist-sans/style.min.css");
@import url("https://cdn.jsdelivr.net/npm/geist@1.3.1/dist/fonts/geist-mono/style.min.css");

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .stApp {
  background: #000000 !important;
  color: #ffffff !important;
  font-family: "Geist Sans", "Geist", "Segoe UI", sans-serif !important;
}

/* Make all UI text selectable / copyable */
html, body, .stApp, .stApp *, p, span, label, h1, h2, h3, h4,
h5, h6, li, td, th, code, pre, div, [data-testid="stMarkdownContainer"],
[data-testid="stText"], [data-testid="stCaptionContainer"] {
  -webkit-user-select: text !important;
  -moz-user-select: text !important;
  -ms-user-select: text !important;
  user-select: text !important;
  cursor: text;
}

button, [role="button"], a, input, textarea, select,
[data-testid="stCheckbox"], [data-baseweb="tab"] {
  cursor: pointer !important;
}

/* Subtle grid overlay like the reference landing page */
[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  opacity: 0.40;
  background-image:
    linear-gradient(rgba(255,255,255,0.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.045) 1px, transparent 1px),
    radial-gradient(ellipse 80% 50% at 50% -10%,
      rgba(250,200,0,0.12), transparent 55%);
  background-size: 48px 48px, 48px 48px, 100% 100%;
}

[data-testid="stHeader"] {
  background: transparent !important;
}
[data-testid="stToolbar"], #MainMenu,
footer, [data-testid="stDecoration"] {
  visibility: hidden;
  height: 0;
}

.block-container {
  position: relative;
  z-index: 1;
  max-width: 980px !important;
  padding-top: 2.5rem !important;
  padding-bottom: 4rem !important;
}

/* Hero */
.hero-kicker {
  color: rgba(255,255,255,0.50);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.4em;
  text-transform: uppercase;
  margin-bottom: 1.25rem;
}
.hero-title {
  font-size: clamp(3rem, 9vw, 6.5rem);
  font-weight: 700;
  letter-spacing: -0.05em;
  line-height: 0.88;
  color: #ffffff;
  margin: 0 0 1.5rem 0;
}
.hero-sub {
  color: rgba(255,255,255,0.90);
  font-size: 1.05rem;
  font-weight: 500;
  line-height: 1.65;
  max-width: 34rem;
  margin-bottom: 0.25rem;
}
.hero-sub .accent-yellow { color: rgba(250, 200, 0, 0.75); }
.hero-sub .accent-emerald { color: rgba(0, 210, 148, 0.70); }

.section-label {
  color: rgba(255,255,255,0.50);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.4em;
  text-transform: uppercase;
  margin: 2.5rem 0 1rem 0;
}

.note-box {
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
  border-radius: 1rem;
  padding: 1rem 1.15rem;
  color: rgba(255,255,255,0.55);
  font-size: 0.9rem;
  font-weight: 500;
  line-height: 1.55;
  margin: 1.25rem 0 1.75rem 0;
}
.note-box strong { color: rgba(255,255,255,0.85); }

/* Tabs */
[data-baseweb="tab-list"] {
  gap: 0.35rem;
  background: transparent !important;
  border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}
[data-baseweb="tab"] {
  background: transparent !important;
  color: rgba(255,255,255,0.45) !important;
  font-weight: 500 !important;
  letter-spacing: 0.02em;
}
[data-baseweb="tab"][aria-selected="true"] {
  color: #ffffff !important;
  border-bottom: 2px solid #fac800 !important;
}
[data-baseweb="tab-highlight"],
[data-baseweb="tab-border"] {
  background-color: #fac800 !important;
}

/* Inputs / controls */
.stTextInput input, .stSelectbox [data-baseweb="select"] > div,
.stNumberInput input, textarea {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 999px !important;
  color: #fff !important;
}
.stTextInput input:focus {
  border-color: rgba(250,200,0,0.55) !important;
  box-shadow: 0 0 0 1px rgba(250,200,0,0.25) !important;
}

/* Primary button → white pill like Connect4 "Play" */
.stButton > button[kind="primary"],
.stButton > button {
  background: #ffffff !important;
  color: #000000 !important;
  border: none !important;
  border-radius: 999px !important;
  font-weight: 600 !important;
  padding: 0.65rem 1.75rem !important;
  transition: background 0.15s ease, transform 0.15s ease;
}
.stButton > button:hover {
  background: rgba(255,255,255,0.88) !important;
  transform: translateY(-1px);
}

h2, h3, .stSubheader {
  letter-spacing: -0.03em !important;
  font-weight: 700 !important;
  color: #ffffff !important;
}
p, label, .stMarkdown, .stCaption {
  color: rgba(255,255,255,0.60) !important;
  font-weight: 500 !important;
}
[data-testid="stCaptionContainer"] p {
  color: rgba(255,255,255,0.40) !important;
}

/* Alerts */
[data-testid="stAlert"] {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 1rem !important;
  color: rgba(255,255,255,0.85) !important;
}

/* Dataframes / code — keep copyable */
[data-testid="stDataFrame"], pre, code, .stCodeBlock {
  border-radius: 0.75rem !important;
  -webkit-user-select: text !important;
  user-select: text !important;
}

.mono-meta {
  font-family: "Geist Mono", ui-monospace, monospace;
  color: rgba(255,255,255,0.45);
  font-size: 0.75rem;
  letter-spacing: 0.02em;
}

.pred-phish {
  color: #ff6568;
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: -0.03em;
}
.pred-legit {
  color: #00d294;
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: -0.03em;
}
</style>
""",
    unsafe_allow_html=True,
)


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


def render_prediction(
    models, selected, feature_names, input_values, compare_all,
    actual_label=None,
):
    input_df = pd.DataFrame(
        [[input_values[f] for f in feature_names]],
        columns=feature_names,
    )
    model = models[selected]
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    score_map = {
        str(cls): float(prob)
        for cls, prob in zip(model.classes_, probs)
    }

    st.markdown(
        '<p class="section-label">Result</p>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        if prediction == "Phishing":
            st.markdown(
                '<p class="pred-phish">PHISHING</p>',
                unsafe_allow_html=True,
            )
            st.write(
                "The model classifies this site as phishing "
                "based on extracted features."
            )
        else:
            st.markdown(
                '<p class="pred-legit">LEGITIMATE</p>',
                unsafe_allow_html=True,
            )
            st.write(
                "The model classifies this site as legitimate "
                "based on extracted features."
            )
        # Copyable summary block
        summary = (
            f"prediction: {prediction}\n"
            f"model: {selected}\n"
            + "\n".join(
                f"{cls}: {score_map[cls]:.4f}" for cls in score_map
            )
        )
        st.code(summary, language="text")
    with right:
        st.dataframe(
            pd.DataFrame(
                {"Class": model.classes_, "Model score": probs}
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Scores are model estimates, not calibrated "
            "real-world probabilities. Select / copy freely."
        )
    if compare_all:
        votes = pd.DataFrame(
            {
                "Model": name,
                "Prediction": m.predict(input_df)[0],
            }
            for name, m in models.items()
        )
        st.dataframe(votes, hide_index=True, width="stretch")
        vote_text = "\n".join(
            f"{row.Model}: {row.Prediction}"
            for row in votes.itertuples()
        )
        st.code(vote_text, language="text")
    if actual_label is not None:
        match = "Yes" if prediction == actual_label else "No"
        st.code(
            f"actual_label: {actual_label}\nmatches: {match}",
            language="text",
        )
    st.caption(
        "A prediction is not proof that a website is safe or malicious. "
        "Verify suspicious sites independently."
    )


# ------------------------------------------------------------------ Hero
st.markdown(
    '<p class="hero-kicker">Academic ML · UCI dataset</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h1 class="hero-title">Phishing<br/>Detector</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    """
<p class="hero-sub">
  Paste a URL. Eight classifiers —
  <span class="accent-yellow">CatBoost</span> to
  <span class="accent-emerald">XGBoost</span> —
  score UCI-style features extracted from the link.
  Educational prototype, not a security guarantee.
</p>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
<div class="note-box">
  <strong>Note.</strong> Some live signals (WHOIS age, Alexa,
  PageRank, Google Index) are approximated. Text across this app
  is selectable — use the copy button on code blocks for results.
</div>
""",
    unsafe_allow_html=True,
)

if not BUNDLE_PATH.exists():
    st.error(
        "Trained models not found. In the project folder run: "
        "python train.py"
    )
    st.stop()

bundle = load_bundle()
models = bundle["models"]
feature_names = bundle["feature_names"]

tab_predict, tab_perf, tab_evidence, tab_about = st.tabs(
    [
        "Predict",
        "Performance",
        "Literature",
        "About",
    ]
)

# ---------------------------------------------------------------- Predict
with tab_predict:
    st.markdown(
        '<p class="section-label">Check a URL</p>',
        unsafe_allow_html=True,
    )
    st.subheader("Classify a website")
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
            help=(
                "http:// and https:// links only. The app extracts "
                "UCI-style features, then runs the trained model."
            ),
        )
        fetch_page = st.checkbox(
            "Fetch page HTML for richer features "
            "(anchors, forms, iframes, redirects)",
            value=True,
        )
        st.caption(
            "With fetching enabled, the app briefly downloads the page "
            "HTML (size-limited, timed out). Disable it to classify "
            "from the URL string alone."
        )
    elif source == "Use a test-set example":
        test_df = read_csv(TEST_PATH)
        if test_df is None:
            st.error(
                "Test samples are missing. Run python train.py first."
            )
            st.stop()
        chosen = st.selectbox(
            "Choose a test example",
            options=list(range(min(len(test_df), 100))),
            format_func=lambda i: (
                f"Example {i + 1} "
                f"(actual: {test_df.iloc[i]['label']})"
            ),
        )
        sample = test_df.iloc[chosen]
        input_values = {f: int(sample[f]) for f in feature_names}
        actual_label = str(sample["label"])
        st.info(
            f"Actual dataset label for this example: **{actual_label}**"
        )
    else:
        with st.expander(
            "Feature inputs (encoded values)", expanded=True
        ):
            st.caption(
                "Values are -1, 0 or 1. See the UCI feature "
                "documentation for meanings."
            )
            cols = st.columns(3)
            for idx, feature in enumerate(feature_names):
                with cols[idx % 3]:
                    input_values[feature] = st.selectbox(
                        feature,
                        options=[-1, 0, 1],
                        index=2,
                        key=f"feature_{feature}",
                    )

    model_names = list(models)
    selected = st.selectbox(
        "Model",
        options=model_names,
        index=(
            model_names.index(DEFAULT_MODEL)
            if DEFAULT_MODEL in model_names
            else 0
        ),
    )
    compare_all = st.checkbox(
        "Show what all models predict for this record"
    )

    analyze_label = (
        "Analyze URL" if source == "Paste a URL" else "Analyze record"
    )
    if st.button(analyze_label, type="primary"):
        if source == "Paste a URL":
            if not url_text.strip():
                st.error("Enter a URL first.")
                st.stop()
            try:
                with st.spinner("Extracting features from URL..."):
                    extraction_info = extract_features(
                        url_text, fetch_page=fetch_page
                    )
                input_values = extraction_info["features"]
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

            st.code(
                f"normalized_url: {extraction_info['url']}",
                language="text",
            )
            if extraction_info["fetched"]:
                st.success(
                    "Page HTML fetched and used for HTML-based features."
                )
            elif fetch_page:
                err = extraction_info["fetch_error"]
                detail = f": {err}" if err else "."
                st.warning(
                    "Could not fetch page HTML"
                    + detail
                    + " Classification uses URL-string features "
                    "plus safe defaults."
                )
            if extraction_info["approximated"]:
                with st.expander(
                    "Approximated / defaulted features",
                    expanded=False,
                ):
                    st.write(
                        "These features could not be measured exactly "
                        "(missing page HTML or retired external APIs) "
                        "and used a neutral/suspicious default of 0, "
                        "or a heuristic:"
                    )
                    st.code(
                        ", ".join(extraction_info["approximated"]),
                        language="text",
                    )

            with st.expander(
                "Extracted feature vector", expanded=False
            ):
                feat_lines = "\n".join(
                    f"{k}={v}" for k, v in input_values.items()
                )
                st.code(feat_lines, language="text")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Feature": k, "Value": v}
                            for k, v in input_values.items()
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )

        render_prediction(
            models,
            selected,
            feature_names,
            input_values,
            compare_all,
            actual_label,
        )

# ------------------------------------------------------------ Performance
with tab_perf:
    st.markdown(
        '<p class="section-label">Hold-out evaluation</p>',
        unsafe_allow_html=True,
    )
    st.subheader("Evaluation on the held-out test set")
    metrics = read_csv(RESULTS / "metrics.csv")
    if metrics is None:
        st.info("Metrics not found. Run python train.py first.")
    else:
        st.dataframe(
            metrics.round(4), hide_index=True, width="stretch"
        )
        chart_cols = [
            c
            for c in [
                "Accuracy",
                "Precision (Phishing)",
                "Recall (Phishing)",
                "F1-score (Phishing)",
                "Macro F1",
            ]
            if c in metrics.columns
        ]
        st.bar_chart(metrics.set_index("Model")[chart_cols])

    st.subheader("5-fold cross-validation (training split)")
    cv = read_csv(RESULTS / "cv_results.csv")
    if cv is not None:
        st.dataframe(cv.round(4), hide_index=True, width="stretch")
        st.caption(
            "Top models overlap within one standard deviation, so "
            "small differences between them should not be "
            "over-interpreted."
        )
        show_image(
            "cv_comparison.png",
            "Macro F1 with standard deviation across folds",
        )

    st.subheader("Confusion matrix")
    pick = st.selectbox(
        "Model", options=list(models), key="cm_model"
    )
    safe = (
        pick.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )
    show_image(
        f"confusion_matrix_{safe}.png",
        f"Confusion matrix: {pick}",
    )
    show_image(
        "feature_importance.png",
        "Random Forest feature importance (not causation)",
    )
    st.markdown(
        "**Interpretation tip:** for phishing detection, inspect "
        "phishing recall and false negatives, not accuracy alone."
    )

# ------------------------------------------------------ Literature/evidence
with tab_evidence:
    st.markdown(
        '<p class="section-label">Published work</p>',
        unsafe_allow_html=True,
    )
    st.subheader("Comparison with published work")
    lit = read_csv(RESULTS / "literature_comparison.csv")
    if lit is not None:
        st.dataframe(
            lit,
            hide_index=True,
            width="stretch",
            column_config={
                "URL": st.column_config.LinkColumn("URL")
            },
        )
    st.subheader("Why published scores look higher: duplicate rows")
    dup = read_csv(RESULTS / "duplicate_effect.csv")
    if dup is not None:
        st.dataframe(
            dup.round(4), hide_index=True, width="stretch"
        )
        st.write(
            "The raw dataset has many exact duplicate rows. When "
            "duplicates are kept, identical records can land in both "
            "training and test sets, inflating accuracy. This project "
            "removes them before splitting, which is stricter but "
            "more honest."
        )
    st.caption(
        "Published results use different splits and preprocessing, "
        "so numbers are indicative rather than a like-for-like "
        "benchmark."
    )

# ------------------------------------------------------------------ About
with tab_about:
    st.markdown(
        '<p class="section-label">How it works</p>',
        unsafe_allow_html=True,
    )
    st.subheader("Project workflow")
    st.markdown("""
    1. Download the public UCI dataset (offline copy included).
    2. Inspect data types, missing values, duplicates and class
       balance.
    3. Remove duplicates, then split into training and test data
       with stratification.
    4. Train 8 models: Logistic Regression, Decision Tree, Random
       Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost and
       an MLP neural network.
    5. Compare on the same held-out test set and with 5-fold
       cross-validation.
    6. Inspect confusion matrices and Random Forest feature
       importance.
    """)
    st.markdown("**Limitations**")
    st.markdown("""
    - Models were trained on UCI's engineered features; URL mode
      *approximates* those features.
    - WHOIS age, Alexa traffic, PageRank, Google Index and backlink
      counts are defaulted when unavailable.
    - Performance may not generalize to current websites or new
      phishing campaigns.
    - Feature importance describes model behavior and does not
      establish causality.
    """)
    st.markdown(
        "[Dataset and documentation: UCI Phishing Websites]"
        "(https://archive.ics.uci.edu/dataset/327/phishing+websites)"
    )
