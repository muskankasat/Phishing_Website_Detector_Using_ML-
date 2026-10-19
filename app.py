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

st.markdown(
    """
<style>
@import url("https://cdn.jsdelivr.net/npm/geist@1.3.1/dist/fonts/geist-sans/style.min.css");
@import url("https://cdn.jsdelivr.net/npm/geist@1.3.1/dist/fonts/geist-mono/style.min.css");

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .stApp,
[data-testid="stAppViewContainer"] > .main,
section.main, [data-testid="stVerticalBlockBorderWrapper"] {
  background: transparent !important;
  color: #ffffff !important;
  font-family: "Geist Sans", "Geist", "Segoe UI", sans-serif !important;
}

/* Fixed black canvas under the aurora */
[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: -2;
  background: #000000;
  pointer-events: none;
}

/* Dynamic B&W aurora light */
.aurora-bg {
  position: fixed;
  inset: 0;
  z-index: -1;
  pointer-events: none;
  overflow: hidden;
  background: #000;
}
.aurora-bg .blob {
  position: absolute;
  border-radius: 50%;
  filter: blur(90px);
  will-change: transform, opacity;
  mix-blend-mode: screen;
}
.aurora-bg .b1 {
  width: 60vw; height: 60vw;
  left: -12%; top: -18%;
  opacity: 0.55;
  background: radial-gradient(circle, rgba(255,255,255,0.95) 0%, transparent 68%);
  animation: drift1 14s ease-in-out infinite alternate,
             pulse1 8s ease-in-out infinite;
}
.aurora-bg .b2 {
  width: 48vw; height: 48vw;
  right: -10%; top: 18%;
  opacity: 0.42;
  background: radial-gradient(circle, rgba(220,220,220,0.9) 0%, transparent 70%);
  animation: drift2 18s ease-in-out infinite alternate,
             pulse2 10s ease-in-out infinite;
}
.aurora-bg .b3 {
  width: 55vw; height: 55vw;
  left: 18%; bottom: -22%;
  opacity: 0.32;
  background: radial-gradient(circle, rgba(255,255,255,0.85) 0%, transparent 70%);
  animation: drift3 22s ease-in-out infinite alternate,
             pulse1 12s ease-in-out infinite reverse;
}
.aurora-bg .b4 {
  width: 35vw; height: 35vw;
  left: 40%; top: 35%;
  opacity: 0.22;
  background: radial-gradient(circle, rgba(180,180,180,0.85) 0%, transparent 72%);
  animation: drift4 16s ease-in-out infinite alternate,
             pulse2 9s ease-in-out infinite;
}
.aurora-bg .ribbon {
  position: absolute;
  width: 140%;
  height: 45%;
  left: -20%;
  top: 20%;
  opacity: 0.18;
  background: linear-gradient(
    105deg,
    transparent 0%,
    rgba(255,255,255,0.35) 25%,
    rgba(200,200,200,0.15) 45%,
    rgba(255,255,255,0.4) 60%,
    transparent 85%
  );
  filter: blur(40px);
  transform: rotate(-8deg);
  animation: ribbon 20s ease-in-out infinite alternate;
}
.aurora-bg .veil {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 55% at 50% -5%,
      rgba(255,255,255,0.10), transparent 58%),
    linear-gradient(180deg, rgba(0,0,0,0.2), rgba(0,0,0,0.82));
}
@keyframes drift1 {
  0%   { transform: translate(0, 0) scale(1); }
  100% { transform: translate(14%, 20%) scale(1.18); }
}
@keyframes drift2 {
  0%   { transform: translate(0, 0) scale(1.05); }
  100% { transform: translate(-20%, -12%) scale(1.22); }
}
@keyframes drift3 {
  0%   { transform: translate(0, 0) scale(1); }
  100% { transform: translate(-12%, -18%) scale(1.28); }
}
@keyframes drift4 {
  0%   { transform: translate(0, 0) scale(0.95); }
  100% { transform: translate(10%, -14%) scale(1.2); }
}
@keyframes pulse1 {
  0%, 100% { opacity: 0.45; }
  50%      { opacity: 0.7; }
}
@keyframes pulse2 {
  0%, 100% { opacity: 0.28; }
  50%      { opacity: 0.5; }
}
@keyframes ribbon {
  0%   { transform: rotate(-8deg) translateX(-4%); opacity: 0.12; }
  100% { transform: rotate(-4deg) translateX(6%); opacity: 0.28; }
}

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

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"], #MainMenu,
footer, [data-testid="stDecoration"] {
  visibility: hidden; height: 0;
}

.block-container {
  position: relative;
  z-index: 1;
  max-width: 920px !important;
  padding-top: 1.25rem !important;
  padding-bottom: 3rem !important;
}

.hero-kicker {
  color: rgba(255,255,255,0.50);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.4em;
  text-transform: uppercase;
  margin: 0 0 0.65rem 0;
}
.hero-title {
  font-size: clamp(2.4rem, 7vw, 4.25rem);
  font-weight: 700;
  letter-spacing: -0.05em;
  line-height: 0.9;
  color: #ffffff;
  margin: 0 0 0.85rem 0;
}
.hero-sub {
  color: rgba(255,255,255,0.72);
  font-size: 0.95rem;
  font-weight: 500;
  line-height: 1.55;
  max-width: 32rem;
  margin: 0 0 1.35rem 0;
}
.hero-sub .accent-yellow { color: rgba(250, 200, 0, 0.85); }
.hero-sub .accent-emerald { color: rgba(0, 210, 148, 0.75); }

.section-label {
  color: rgba(255,255,255,0.50);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.4em;
  text-transform: uppercase;
  margin: 0.5rem 0 0.75rem 0;
}

/* Keep URL + model controls on one visual row */
div[data-testid="stHorizontalBlock"] {
  align-items: end !important;
  gap: 0.75rem !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
  display: flex !important;
  flex-direction: column !important;
  justify-content: flex-end !important;
}

[data-baseweb="tab-list"] {
  gap: 0.35rem;
  background: transparent !important;
  border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}
[data-baseweb="tab"] {
  background: transparent !important;
  color: rgba(255,255,255,0.45) !important;
  font-weight: 500 !important;
}
[data-baseweb="tab"][aria-selected="true"] {
  color: #ffffff !important;
  border-bottom: 2px solid #fac800 !important;
}
[data-baseweb="tab-highlight"],
[data-baseweb="tab-border"] {
  background-color: #fac800 !important;
}

.stTextInput input, .stSelectbox [data-baseweb="select"] > div,
.stNumberInput input, textarea {
  background: rgba(0,0,0,0.45) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 999px !important;
  color: #fff !important;
  backdrop-filter: blur(8px);
  min-height: 2.75rem !important;
}
.stTextInput input:focus {
  border-color: rgba(255,255,255,0.65) !important;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.35) !important;
}

/* Analyze CTA — flat, dark text, no glow */
div[data-testid="stButton"] > button[kind="primary"],
div.analyze-wrap button {
  background: linear-gradient(180deg, #ffffff 0%, #e8e8e8 100%) !important;
  color: #111111 !important;
  border: 1px solid rgba(0,0,0,0.18) !important;
  border-radius: 999px !important;
  font-weight: 700 !important;
  font-size: 1.05rem !important;
  letter-spacing: 0.02em !important;
  padding: 0.9rem 2.4rem !important;
  min-height: 3.1rem !important;
  width: 100% !important;
  box-shadow: none !important;
  text-shadow: none !important;
  transition: transform 0.15s ease, background 0.15s ease !important;
}
div[data-testid="stButton"] > button[kind="primary"] p,
div[data-testid="stButton"] > button[kind="primary"] span,
div[data-testid="stButton"] > button[kind="primary"] div {
  color: #111111 !important;
  text-shadow: none !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
  transform: translateY(-1px) !important;
  background: linear-gradient(180deg, #f5f5f5 0%, #dddddd 100%) !important;
  box-shadow: none !important;
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

[data-testid="stAlert"] {
  background: rgba(0,0,0,0.45) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 1rem !important;
  color: rgba(255,255,255,0.85) !important;
  backdrop-filter: blur(10px);
}

[data-testid="stDataFrame"], pre, code, .stCodeBlock {
  border-radius: 0.75rem !important;
  -webkit-user-select: text !important;
  user-select: text !important;
}

.result-panel {
  margin-top: 1.1rem;
  padding: 1.25rem 1.35rem;
  border-radius: 1.1rem;
  backdrop-filter: blur(14px);
}
.result-panel.panel-phish {
  border: 1px solid rgba(255,101,104,0.45);
  background: rgba(255, 40, 55, 0.14);
  box-shadow: 0 0 28px rgba(255, 60, 70, 0.12);
}
.result-panel.panel-legit {
  border: 1px solid rgba(0,210,148,0.45);
  background: rgba(0, 180, 120, 0.14);
  box-shadow: 0 0 28px rgba(0, 210, 148, 0.12);
}
.pred-phish {
  color: #ff4d52 !important;
  font-size: 1.85rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  margin: 0 0 0.35rem 0;
  text-shadow: 0 0 18px rgba(255, 80, 90, 0.35);
}
.pred-legit {
  color: #00e0a0 !important;
  font-size: 1.85rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  margin: 0 0 0.35rem 0;
  text-shadow: 0 0 18px rgba(0, 210, 148, 0.35);
}
.tag-legit {
  color: #00e0a0 !important;
  font-weight: 700 !important;
}
.tag-phish {
  color: #ff4d52 !important;
  font-weight: 700 !important;
}
</style>
<div class="aurora-bg" aria-hidden="true">
  <div class="blob b1"></div>
  <div class="blob b2"></div>
  <div class="blob b3"></div>
  <div class="blob b4"></div>
  <div class="ribbon"></div>
  <div class="veil"></div>
</div>
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
    """Result badge + class/score table directly under Analyze."""
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

    is_phish = prediction == "Phishing"
    css = "pred-phish" if is_phish else "pred-legit"
    panel = "panel-phish" if is_phish else "panel-legit"
    tag = "tag-phish" if is_phish else "tag-legit"
    blurb = (
        'The model classifies this site as '
        f'<span class="{tag}">{prediction.lower()}</span> '
        "based on extracted features."
    )
    st.markdown(
        f"""
<div class="result-panel {panel}">
  <p class="{css}">{prediction.upper()}</p>
  <p style="color:rgba(255,255,255,0.55);margin:0 0 0.35rem 0;
     font-weight:500;">{blurb}</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame(
            {"Class": model.classes_, "Model score": probs}
        ),
        hide_index=True,
        width="stretch",
    )
    summary = (
        f"prediction: {prediction}\n"
        f"model: {selected}\n"
        + "\n".join(f"{cls}: {score_map[cls]:.4f}" for cls in score_map)
    )
    st.code(summary, language="text")

    if compare_all:
        votes = pd.DataFrame(
            {
                "Model": name,
                "Prediction": m.predict(input_df)[0],
            }
            for name, m in models.items()
        )
        st.dataframe(votes, hide_index=True, width="stretch")
        vote_lines = []
        for row in votes.itertuples():
            vote_tag = (
                "tag-phish"
                if row.Prediction == "Phishing"
                else "tag-legit"
            )
            vote_lines.append(
                f'{row.Model}: '
                f'<span class="{vote_tag}">{row.Prediction}</span>'
            )
        st.markdown(
            "<br>".join(vote_lines),
            unsafe_allow_html=True,
        )
    if actual_label is not None:
        match = "Yes" if prediction == actual_label else "No"
        label_tag = (
            "tag-phish"
            if actual_label == "Phishing"
            else "tag-legit"
        )
        st.markdown(
            f'actual_label: '
            f'<span class="{label_tag}">{actual_label}</span> · '
            f"matches: {match}",
            unsafe_allow_html=True,
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
    '<h1 class="hero-title">Phishing Detector</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    """
<p class="hero-sub">
  Paste a URL. Eight classifiers —
  <span class="accent-yellow">CatBoost</span> to
  <span class="accent-emerald">XGBoost</span> —
  score features from the link.
</p>
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
model_names = list(models)

tab_predict, tab_perf, tab_evidence, tab_about = st.tabs(
    ["Predict", "Performance", "Literature", "About"]
)

# ---------------------------------------------------------------- Predict
with tab_predict:
    source = st.radio(
        "Input method",
        ["Paste a URL", "Use a test-set example", "Enter feature values"],
        horizontal=True,
        label_visibility="collapsed",
    )

    actual_label = None
    input_values = {}
    extraction_info = None
    url_text = ""

    if source == "Paste a URL":
        st.markdown(
            '<p class="section-label">URL &amp; model</p>',
            unsafe_allow_html=True,
        )
        url_col, model_col = st.columns([2.6, 1], gap="small")
        with url_col:
            st.caption("Website URL")
            url_text = st.text_input(
                "Website URL",
                placeholder="https://example.com/login",
                label_visibility="collapsed",
                help=(
                    "http:// and https:// only. Extracts UCI-style "
                    "features, then runs the trained model."
                ),
            )
        with model_col:
            st.caption("Model")
            selected = st.selectbox(
                "Model",
                options=model_names,
                index=(
                    model_names.index(DEFAULT_MODEL)
                    if DEFAULT_MODEL in model_names
                    else 0
                ),
                label_visibility="collapsed",
            )
        fetch_page = st.checkbox(
            "Fetch page HTML for richer features",
            value=True,
        )
    elif source == "Use a test-set example":
        test_df = read_csv(TEST_PATH)
        if test_df is None:
            st.error(
                "Test samples are missing. Run python train.py first."
            )
            st.stop()
        st.markdown(
            '<p class="section-label">Example &amp; model</p>',
            unsafe_allow_html=True,
        )
        ex_col, model_col = st.columns([2.6, 1], gap="small")
        with ex_col:
            st.caption("Test example")
            chosen = st.selectbox(
                "Choose a test example",
                options=list(range(min(len(test_df), 100))),
                format_func=lambda i: (
                    f"Example {i + 1} "
                    f"(actual: {test_df.iloc[i]['label']})"
                ),
                label_visibility="collapsed",
            )
            sample = test_df.iloc[chosen]
            input_values = {
                f: int(sample[f]) for f in feature_names
            }
            actual_label = str(sample["label"])
            label_tag = (
                "tag-phish"
                if actual_label == "Phishing"
                else "tag-legit"
            )
            st.markdown(
                f'Actual dataset label: '
                f'<span class="{label_tag}">{actual_label}</span>',
                unsafe_allow_html=True,
            )
        with model_col:
            st.caption("Model")
            selected = st.selectbox(
                "Model",
                options=model_names,
                index=(
                    model_names.index(DEFAULT_MODEL)
                    if DEFAULT_MODEL in model_names
                    else 0
                ),
                key="model_test",
                label_visibility="collapsed",
            )
        fetch_page = False
    else:
        st.markdown(
            '<p class="section-label">Features &amp; model</p>',
            unsafe_allow_html=True,
        )
        feat_col, model_col = st.columns([2.6, 1], gap="small")
        with feat_col:
            with st.expander(
                "Feature inputs (encoded values)", expanded=True
            ):
                st.caption("Values are -1, 0 or 1.")
                cols = st.columns(3)
                for idx, feature in enumerate(feature_names):
                    with cols[idx % 3]:
                        input_values[feature] = st.selectbox(
                            feature,
                            options=[-1, 0, 1],
                            index=2,
                            key=f"feature_{feature}",
                        )
        with model_col:
            st.caption("Model")
            selected = st.selectbox(
                "Model",
                options=model_names,
                index=(
                    model_names.index(DEFAULT_MODEL)
                    if DEFAULT_MODEL in model_names
                    else 0
                ),
                key="model_manual",
                label_visibility="collapsed",
            )
        fetch_page = False

    compare_all = st.checkbox(
        "Show what all models predict for this record"
    )

    analyze_label = (
        "Analyze URL" if source == "Paste a URL" else "Analyze record"
    )
    clicked = st.button(analyze_label, type="primary")

    if clicked:
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

            # Primary result block sits immediately under Analyze
            render_prediction(
                models,
                selected,
                feature_names,
                input_values,
                compare_all,
                actual_label,
            )

            with st.expander("Extraction details", expanded=False):
                st.code(
                    f"normalized_url: {extraction_info['url']}",
                    language="text",
                )
                if extraction_info["fetched"]:
                    st.success(
                        "Page HTML fetched for HTML-based features."
                    )
                elif fetch_page:
                    err = extraction_info["fetch_error"]
                    detail = f": {err}" if err else "."
                    st.warning(
                        "Could not fetch page HTML"
                        + detail
                        + " Used URL-string features + defaults."
                    )
                if extraction_info["approximated"]:
                    st.code(
                        ", ".join(extraction_info["approximated"]),
                        language="text",
                    )
                feat_lines = "\n".join(
                    f"{k}={v}" for k, v in input_values.items()
                )
                st.code(feat_lines, language="text")
        else:
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
