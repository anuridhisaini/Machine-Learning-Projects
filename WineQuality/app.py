import streamlit as st
import numpy as np
import joblib
import pickle
import os

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Lato:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Lato', sans-serif; }
.stApp { background: linear-gradient(135deg, #1a0a0a 0%, #2d0f0f 50%, #1a0a0a 100%); color: #f0e6d3; }
h1 { font-family: 'Playfair Display', serif !important; color: #c9964a !important; }
h2, h3 { font-family: 'Playfair Display', serif !important; color: #e8c98a !important; }
section[data-testid="stSidebar"] { background: #120606 !important; border-right: 1px solid #3d1a1a; }
section[data-testid="stSidebar"] * { color: #f0e6d3 !important; }
input[type="number"] { background: #2d1010 !important; color: #f0e6d3 !important; border: 1px solid #5a2a2a !important; border-radius: 6px !important; }
div.stButton > button { background: linear-gradient(135deg, #8b1a1a, #c9964a); color: #fff1e0; font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 600; border: none; border-radius: 10px; padding: 0.7rem 2.5rem; width: 100%; }
div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(201,150,74,0.35); }
.result-card { background: linear-gradient(135deg, rgba(139,26,26,0.3), rgba(45,16,16,0.6)); border: 1px solid #c9964a; border-radius: 16px; padding: 2rem; text-align: center; margin-top: 1.5rem; }
.result-score { font-family:'Playfair Display',serif; font-size:3.5rem; color:#c9964a; line-height:1; }
.result-label { font-size:1.2rem; color:#e8c98a; margin-top:0.4rem; letter-spacing:2px; text-transform:uppercase; }
.result-desc  { font-size:0.95rem; color:#c8b49a; margin-top:0.6rem; }
.badge        { display:inline-block; padding:0.3rem 1.2rem; border-radius:20px; font-size:0.9rem; font-weight:600; margin-top:0.8rem; }
.badge-poor    { background:#6a2d2d; color:#f0b6b6; }
.badge-average { background:#6a4e2d; color:#f0d9b6; }
.badge-good    { background:#2d6a2d; color:#b6f0b6; }
hr { border-color: #3d1a1a !important; }
div[data-testid="metric-container"] { background: rgba(45,16,16,0.5); border: 1px solid #3d1a1a; border-radius: 10px; padding: 0.6rem 1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    base   = os.path.dirname(os.path.abspath(__file__))
    model  = joblib.load(os.path.join(base, "rb_winequality.pkl"))
    scaler = joblib.load(os.path.join(base, "scaler (1).pkl"))
    with open(os.path.join(base, "columns (1).pkl"), "rb") as f:
        columns = pickle.load(f)
    return model, scaler, columns   # all 12 columns including quality_encoded


try:
    model, scaler, feature_cols = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    load_error = str(e)


QUALITY_MAP = {
    0: ("Poor",    "🔴", "badge-poor",    "The wine shows noticeable flaws in balance or finish."),
    1: ("Average", "🟡", "badge-average", "A pleasant, everyday wine with no major faults."),
    2: ("Good",    "🟢", "badge-good",    "Excellent structure, aroma and lingering finish."),
}

FEATURE_CONFIG = {
    "fixed acidity":        (4.0,   16.0,  7.4,   0.1,    "g/dm³"),
    "volatile acidity":     (0.08,   1.58, 0.52,  0.01,   "g/dm³"),
    "citric acid":          (0.0,    1.0,  0.26,  0.01,   "g/dm³"),
    "residual sugar":       (0.9,   65.0,  2.2,   0.1,    "g/dm³"),
    "chlorides":            (0.009,  0.61, 0.047, 0.001,  "g/dm³"),
    "free sulfur dioxide":  (1.0,  289.0,  35.0,  1.0,    "mg/dm³"),
    "total sulfur dioxide": (6.0,  440.0, 138.0,  1.0,    "mg/dm³"),
    "density":              (0.987,  1.004, 0.994, 0.0001, "g/cm³"),
    "pH":                   (2.72,   4.01,  3.19,  0.01,  ""),
    "sulphates":            (0.22,   2.0,   0.49,  0.01,  "g/dm³"),
    "alcohol":              (8.0,   14.9,  10.3,   0.1,   "% vol"),
    "quality_encoded":      (3.0,    8.0,   6.0,   1.0,   "score 3–8"),
}


st.markdown("# 🍷 Wine Quality Predictor")
st.markdown("*Enter the physicochemical properties of the wine to predict its quality class.*")
st.markdown("---")

if not artifacts_loaded:
    st.error(f"❌ Could not load model files. Make sure the `.pkl` files are in the same folder as `app.py`.\n\n**Error:** `{load_error}`")
    st.stop()


st.subheader("🧪 Physicochemical Parameters")
col1, col2 = st.columns(2)
inputs = {}

for i, (feat, (mn, mx, default, step, unit)) in enumerate(FEATURE_CONFIG.items()):
    label = f"{feat.title()} ({unit})" if unit else feat.title()
    fmt   = "%.4f" if step < 0.01 else ("%.3f" if step < 0.1 else "%.2f")
    with (col1 if i % 2 == 0 else col2):
        inputs[feat] = st.number_input(
            label,
            min_value=float(mn),
            max_value=float(mx),
            value=float(default),
            step=float(step),
            format=fmt,
        )

st.markdown("---")


if st.button("✨ Predict Wine Quality"):

    input_values = np.array([[inputs[f] for f in feature_cols]])
    input_scaled = scaler.transform(input_values)

    pred_class = int(model.predict(input_scaled)[0])
    pred_proba = model.predict_proba(input_scaled)[0]

    label, emoji, badge_cls, description = QUALITY_MAP[pred_class]
    confidence = pred_proba[pred_class] * 100

    st.markdown(f"""
    <div class="result-card">
        <div class="result-score">{emoji} {label}</div>
        <div class="result-label">Predicted Quality Class &nbsp;·&nbsp; {pred_class}</div>
        <div class="result-desc">{description}</div>
        <span class="badge {badge_cls}">Confidence: {confidence:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Class Probabilities")
    class_names = {0: "Poor (0)", 1: "Average (1)", 2: "Good (2)"}
    prob_cols = st.columns(3)
    for cls_idx, pcol in enumerate(prob_cols):
        with pcol:
            st.metric(
                label=class_names[cls_idx],
                value=f"{pred_proba[cls_idx]*100:.1f}%",
                delta="← predicted" if cls_idx == pred_class else None,
            )

    for cls_idx in range(3):
        st.markdown(f"**{class_names[cls_idx]}**")
        st.progress(float(pred_proba[cls_idx]))

    st.success(f"✅ This wine is classified as **{label}** with **{confidence:.1f}%** confidence.")


with st.sidebar:
    st.markdown("## ℹ️ Model Info")
    st.markdown(f"**Algorithm:** `{model.__class__.__name__}`")
    st.markdown(f"**Input Features:** `{len(feature_cols)}`")
    st.markdown(f"**Output Classes:** `3` (Poor / Average / Good)")
    if hasattr(model, "n_estimators"):
        st.markdown(f"**Trees:** `{model.n_estimators}`")
    st.markdown("---")
    st.markdown("### 📁 Pickle Files")
    base = os.path.dirname(os.path.abspath(__file__))
    for fname in ["rb_winequality.pkl", "scaler__1_.pkl", "columns__1_.pkl"]:
        exists = os.path.exists(os.path.join(base, fname))
        st.markdown(f"{'✅' if exists else '❌'} `{fname}`")
    st.markdown("---")
    st.markdown("### 🍷 Class Reference")
    st.markdown("""
| Class | Label   | Meaning           |
|:-----:|---------|-------------------|
| 0     | Poor    | Quality score 3–4 |
| 1     | Average | Quality score 5–6 |
| 2     | Good    | Quality score 7–8 |
""")
    st.caption("Wine Quality Predictor · RandomForest · UCI Dataset")