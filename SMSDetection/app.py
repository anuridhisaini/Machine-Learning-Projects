# ============================================================
#   📩 SMS Spam Detector — Streamlit App
#   Model: Multinomial Naive Bayes + TF-IDF Vectorizer
# ============================================================

import time
import string
import joblib
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# ── Download required NLTK data (only on first run) ──────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

# ── Page config (must be the very first Streamlit call) ──────
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
#  CUSTOM CSS — dark, modern, card-based UI
# ═══════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    /* ── Global resets ── */
    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ── Main container ── */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 720px !important;
    }

    /* ── Title ── */
    .main-title {
        text-align: center;
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }

    /* ── Subtitle ── */
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    /* ── Card wrapper ── */
    .card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }

    /* ── Result cards ── */
    .result-spam {
        background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(220,38,38,0.08));
        border: 1.5px solid rgba(239,68,68,0.5);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        text-align: center;
        animation: fadeSlideIn 0.5s ease;
    }

    .result-ham {
        background: linear-gradient(135deg, rgba(52,211,153,0.18), rgba(16,185,129,0.08));
        border: 1.5px solid rgba(52,211,153,0.5);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        text-align: center;
        animation: fadeSlideIn 0.5s ease;
    }

    .result-label-spam {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #f87171;
        margin-bottom: 0.4rem;
    }

    .result-label-ham {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #34d399;
        margin-bottom: 0.4rem;
    }

    .result-sub {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    /* ── Fade-slide animation ── */
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Streamlit button override ── */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        cursor: pointer;
        transition: opacity 0.2s ease, transform 0.15s ease;
        letter-spacing: 0.5px;
    }
    div.stButton > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
    }
    div.stButton > button:active {
        transform: translateY(0);
    }

    /* ── Text area ── */
    textarea {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.92rem !important;
        background: rgba(255,255,255,0.06) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px !important;
    }
    textarea:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 2px rgba(124,58,237,0.3) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.92) !important;
        border-right: 1px solid rgba(255,255,255,0.07) !important;
    }
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #a78bfa !important;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.82rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }

    /* ── Confidence badge ── */
    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.08);
        border-radius: 999px;
        padding: 0.2rem 0.85rem;
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        font-family: 'Space Mono', monospace;
    }

    /* ── Example chips ── */
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.8rem;
    }
    .chip {
        background: rgba(124,58,237,0.15);
        border: 1px solid rgba(124,58,237,0.3);
        border-radius: 999px;
        padding: 0.25rem 0.85rem;
        font-size: 0.78rem;
        color: #c4b5fd;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════
#  LOAD MODEL & VECTORIZER  (cached so they load only once)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the trained model and TF-IDF vectorizer from disk."""
    model      = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

# ═══════════════════════════════════════════════════════════════
#  TEXT PREPROCESSING  — must match training pipeline exactly
# ═══════════════════════════════════════════════════════════════
def preprocess(text: str) -> str:
    """
    1. Lowercase
    2. Remove punctuation
    3. Tokenise
    4. Remove English stopwords
    5. Apply Porter Stemming
    """
    ps          = PorterStemmer()
    stop_words  = set(stopwords.words("english"))

    # Step 1 — lowercase
    text = text.lower()

    # Step 2 — remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Step 3 — simple whitespace tokenisation
    tokens = text.split()

    # Steps 4 & 5 — stopword removal + stemming
    tokens = [ps.stem(t) for t in tokens if t not in stop_words]

    return " ".join(tokens)

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📩 SMS Spam Detector")
    st.markdown("---")

    st.markdown("### 🧠 About this App")
    st.markdown(
        "This tool analyses any SMS or short text message and tells you "
        "whether it looks like **spam** or a legitimate (**ham**) message — "
        "instantly, right in your browser."
    )

    st.markdown("---")
    st.markdown("### ⚙️ Model Details")

    st.markdown("**Algorithm**")
    st.markdown("Multinomial Naive Bayes")

    st.markdown("**Features**")
    st.markdown("TF-IDF Vectorization")

    st.markdown("**Preprocessing**")
    st.markdown(
        "- Lowercase conversion\n"
        "- Punctuation removal\n"
        "- Stopword filtering\n"
        "- Porter Stemming"
    )

    st.markdown("---")
    st.markdown("### 📊 How to Use")
    st.markdown(
        "1. Type or paste a message in the box\n"
        "2. Click **Check Message**\n"
        "3. See the verdict instantly!"
    )

    st.markdown("---")
    st.caption("Model trained on the UCI SMS Spam Collection dataset.")

# ═══════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════

# ── Load artifacts with a one-time spinner ───────────────────
with st.spinner("Loading model..."):
    model, vectorizer = load_artifacts()

# ── Header ───────────────────────────────────────────────────
st.markdown('<p class="main-title">📩 SMS Spam Detector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Paste any message below and find out if it\'s spam or ham in seconds.</p>',
    unsafe_allow_html=True,
)

# ── Input card ───────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)

user_input = st.text_area(
    label="✉️  Your Message",
    placeholder="e.g.  Congratulations! You've won a free iPhone. Click here to claim now…",
    height=140,
    label_visibility="visible",
)

# Quick-try example messages
st.markdown(
    """
    <p style="color:#64748b; font-size:0.8rem; margin-top:0.4rem;">
        💡 Try an example:
    </p>
    <div class="chip-row">
        <span class="chip" onclick="void(0)">Free prize claim</span>
        <span class="chip" onclick="void(0)">Hey, are you free tonight?</span>
        <span class="chip" onclick="void(0)">Win cash now!</span>
        <span class="chip" onclick="void(0)">Meeting at 3pm tomorrow</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)

# ── Check button ─────────────────────────────────────────────
check_clicked = st.button("🔍  Check Message", use_container_width=True)

# ── Prediction logic ─────────────────────────────────────────
if check_clicked:
    if not user_input.strip():
        st.warning("⚠️  Please enter a message before clicking **Check Message**.")
    else:
        # Show spinner while "processing"
        with st.spinner("Analysing your message…"):
            time.sleep(0.8)          # brief pause for UX effect

            # Preprocess → vectorize → predict
            processed   = preprocess(user_input)
            vectorized  = vectorizer.transform([processed])
            prediction  = model.predict(vectorized)[0]

            # Probability scores (if the model supports predict_proba)
            try:
                proba       = model.predict_proba(vectorized)[0]
                confidence  = max(proba) * 100
                has_proba   = True
            except AttributeError:
                has_proba   = False

        # ── Display result ────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == 1:          # SPAM
            confidence_html = (
                f'<span class="badge">Confidence: {confidence:.1f}%</span>'
                if has_proba else ""
            )
            st.markdown(
                f"""
                <div class="result-spam">
                    <div class="result-label-spam">🚨 SPAM</div>
                    <div class="result-sub">
                        This message looks like spam. Be careful!
                    </div>
                    {confidence_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:                        # HAM
            confidence_html = (
                f'<span class="badge">Confidence: {confidence:.1f}%</span>'
                if has_proba else ""
            )
            st.markdown(
                f"""
                <div class="result-ham">
                    <div class="result-label-ham">✅ HAM</div>
                    <div class="result-sub">
                        This message appears to be legitimate. All good!
                    </div>
                    {confidence_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Debug expander (optional) ─────────────────────────
        with st.expander("🔬  Preprocessing details"):
            st.markdown(f"**Original message:**\n\n`{user_input}`")
            st.markdown(f"**After preprocessing:**\n\n`{processed}`")

# ── Footer ───────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Built with ❤️ using Streamlit &nbsp;|&nbsp; '
    'Powered by Naive Bayes + TF-IDF</div>',
    unsafe_allow_html=True,
)
