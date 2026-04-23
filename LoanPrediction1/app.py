import streamlit as st
import numpy as np
import joblib
import time

# ── Page config (must be first Streamlit command) ───────────────────
st.set_page_config(
    page_title="Smart Loan Approval Predictor by Sainiji",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load model ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("loan_model.pkl")

model = load_model()

# ── Global CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0c10 !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(99,102,241,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(16,185,129,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 100% 100% at 50% 50%, rgba(10,12,16,0.96) 0%, transparent 100%),
        url('https://images.unsplash.com/photo-1486325212027-8081e485255e?w=1800&q=80') center/cover no-repeat;
    z-index: -1;
    pointer-events: none;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.block-container { padding: 2rem 1rem 4rem !important; max-width: 920px !important; margin: auto; }

[data-testid="stSidebar"] {
    background: rgba(10,12,16,0.94) !important;
    border-right: 1px solid rgba(99,102,241,0.18);
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
[data-testid="stSelectbox"] > div > div:hover,
[data-testid="stNumberInput"] input:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}
[data-testid="stSelectbox"] svg { fill: #818cf8 !important; }

label, .stSelectbox label, .stNumberInput label {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%, #4338ca 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
    margin-top: 0.5rem !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.5) !important;
}

[data-testid="stButton"] > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    color: #818cf8 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: rgba(99,102,241,0.1) !important;
    border-color: rgba(99,102,241,0.6) !important;
}

hr { border-color: rgba(99,102,241,0.15) !important; margin: 1.5rem 0 !important; }

[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #6366f1, #10b981) !important;
    border-radius: 99px !important;
}
[data-testid="stProgress"] > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 99px !important;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-family: 'Playfair Display', serif !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem'>
        <div style='font-size:2rem;margin-bottom:0.5rem'>💳</div>
        <div style='font-family:Playfair Display,serif;font-size:1.3rem;
        color:#818cf8;margin-bottom:0.25rem;font-weight:700'>Loan Predictor</div>
        <p style='font-size:0.78rem;color:#475569;line-height:1.6'>
        AI-powered loan approval system built with machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<div style='font-size:0.7rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;font-weight:500;margin-bottom:0.75rem'>About the Project</div>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:0.82rem;color:#64748b;line-height:1.7'>
    Uses a <strong style='color:#818cf8'>Soft Voting Ensemble</strong> of
    Logistic Regression, Random Forest and XGBoost trained on real loan data.
    </p>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("<div style='font-size:0.7rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;font-weight:500;margin-bottom:0.75rem'>Model Stats</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.82rem;color:#64748b;line-height:2.2'>
    🎯 &nbsp;Local Accuracy: <strong style='color:#10b981'>86.99%</strong><br>
    🌿 &nbsp;Ensemble: LR + RF + XGBoost<br>
    📊 &nbsp;14 engineered features<br>
    📁 &nbsp;614 training samples
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("<div style='font-size:0.7rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;font-weight:500;margin-bottom:0.75rem'>Developer</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.82rem;color:#64748b;line-height:2.2'>
    👨‍💻 &nbsp;Python & Streamlit<br>
    🤖 &nbsp;XGBoost, scikit-learn<br>
    📅 &nbsp;2025
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("<p style='font-size:0.72rem;color:#334155;text-align:center'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)


# ── Main header ─────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:2.5rem 0 2rem'>
    <div style='font-size:3rem;margin-bottom:0.75rem'>💳</div>
    <h1 style='
        font-family:Playfair Display,serif;
        font-size:clamp(1.8rem,4vw,2.8rem);
        font-weight:700;color:#f1f5f9;
        letter-spacing:-0.02em;line-height:1.2;margin-bottom:0.6rem;
    '>Smart Loan Approval Predictor</h1>
    <p style='
        font-family:DM Sans,sans-serif;font-size:0.88rem;
        color:#64748b;letter-spacing:0.1em;text-transform:uppercase;
    '>AI-powered decision system for loan approval</p>
</div>
""", unsafe_allow_html=True)


# ── Glass card open ──────────────────────────────────────────────────
st.markdown("""
<div style='
    background:rgba(255,255,255,0.035);
    backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
    border:1px solid rgba(99,102,241,0.18);border-radius:24px;
    padding:2.5rem 2.5rem 1.5rem;
    box-shadow:0 25px 60px rgba(0,0,0,0.4),inset 0 1px 0 rgba(255,255,255,0.06);
    margin-bottom:1.5rem;
'>
<p style='font-size:0.68rem;color:#475569;letter-spacing:0.12em;
text-transform:uppercase;margin-bottom:1.5rem;font-weight:500'>— Application Details</p>
""", unsafe_allow_html=True)

# ── Inputs row 1 ────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    gender       = st.selectbox("Gender", ["Male", "Female"])
    married      = st.selectbox("Married", ["Yes", "No"])
    education    = st.selectbox("Education", ["Graduate", "Not Graduate"])
with c2:
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    dependents    = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
with c3:
    credit_history = st.selectbox("Credit History", [1.0, 0.0],
                                   format_func=lambda x: "Good ✓" if x == 1.0 else "Bad ✗")
    loan_term      = st.selectbox("Loan Term (months)", [360, 180, 240, 120, 300, 480, 60, 36, 12])

st.markdown("<br>", unsafe_allow_html=True)

# ── Inputs row 2 ────────────────────────────────────────────────────
c4, c5, c6 = st.columns(3)
with c4:
    applicant_income   = st.number_input("Applicant Income ₹", min_value=0, value=5000, step=500)
with c5:
    coapplicant_income = st.number_input("Co-applicant Income ₹", min_value=0, value=0, step=500)
with c6:
    loan_amount        = st.number_input("Loan Amount (thousands) ₹", min_value=1, value=150, step=10)

st.markdown("</div>", unsafe_allow_html=True)


# ── Feature encoding ────────────────────────────────────────────────
def get_features():
    gender_enc        = 1 if gender == "Male" else 0
    married_enc       = 1 if married == "Yes" else 0
    dependents_enc    = 3 if dependents == "3+" else int(dependents)
    education_enc     = 0 if education == "Graduate" else 1
    self_employed_enc = 1 if self_employed == "Yes" else 0
    property_enc      = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

    total_income      = applicant_income + coapplicant_income
    total_income_log  = np.log1p(total_income)
    loan_amount_log   = np.log1p(loan_amount)
    applicant_log     = np.log1p(applicant_income)
    emi               = loan_amount / loan_term if loan_term > 0 else 0
    balance_income    = total_income - (emi * 1000)
    debt_to_income    = loan_amount / (total_income + 1)

    return np.array([[
        gender_enc, married_enc, dependents_enc, education_enc,
        self_employed_enc, loan_term, credit_history, property_enc,
        total_income_log, loan_amount_log, applicant_log,
        emi, balance_income, debt_to_income
    ]])


# ── Action buttons ───────────────────────────────────────────────────
b1, b2 = st.columns([3, 1])
with b1:
    predict_clicked = st.button("🔍  Analyze Loan Application", type="primary")
with b2:
    reset_clicked = st.button("↺  Reset", type="secondary")

if reset_clicked:
    st.rerun()


# ── Result ───────────────────────────────────────────────────────────
if predict_clicked:
    with st.spinner("Analyzing application..."):
        time.sleep(1.2)

    features    = get_features()
    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    approved_pct = probability[1] * 100
    rejected_pct = probability[0] * 100

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(f"""
        <div style='
            background:linear-gradient(135deg,rgba(16,185,129,0.12),rgba(5,150,105,0.08));
            border:1px solid rgba(16,185,129,0.35);border-radius:20px;
            padding:2rem 2.5rem;text-align:center;
            box-shadow:0 8px 32px rgba(16,185,129,0.15);
        '>
            <div style='font-size:3.5rem;margin-bottom:0.5rem'>✅</div>
            <h2 style='font-family:Playfair Display,serif;font-size:2rem;
            color:#10b981;margin-bottom:0.4rem'>Loan Approved</h2>
            <p style='color:#6ee7b7;font-size:0.82rem;letter-spacing:0.07em;text-transform:uppercase'>
            Application meets eligibility criteria</p>
            <div style='margin-top:1rem;font-size:2.8rem;font-family:Playfair Display,serif;
            color:#34d399;font-weight:700'>{approved_pct:.1f}%</div>
            <p style='color:#4ade80;font-size:0.72rem;letter-spacing:0.08em;
            text-transform:uppercase;margin-top:0.2rem'>Approval confidence</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='
            background:linear-gradient(135deg,rgba(239,68,68,0.12),rgba(185,28,28,0.08));
            border:1px solid rgba(239,68,68,0.35);border-radius:20px;
            padding:2rem 2.5rem;text-align:center;
            box-shadow:0 8px 32px rgba(239,68,68,0.15);
        '>
            <div style='font-size:3.5rem;margin-bottom:0.5rem'>❌</div>
            <h2 style='font-family:Playfair Display,serif;font-size:2rem;
            color:#ef4444;margin-bottom:0.4rem'>Loan Rejected</h2>
            <p style='color:#fca5a5;font-size:0.82rem;letter-spacing:0.07em;text-transform:uppercase'>
            Application does not meet eligibility criteria</p>
            <div style='margin-top:1rem;font-size:2.8rem;font-family:Playfair Display,serif;
            color:#f87171;font-weight:700'>{rejected_pct:.1f}%</div>
            <p style='color:#f87171;font-size:0.72rem;letter-spacing:0.08em;
            text-transform:uppercase;margin-top:0.2rem'>Rejection confidence</p>
        </div>
        """, unsafe_allow_html=True)

    # Probability bar
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.68rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;font-weight:500;margin-bottom:0.5rem'>— Prediction Confidence Meter</p>", unsafe_allow_html=True)
    st.progress(float(probability[1]))

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Approval Probability", f"{approved_pct:.1f}%")
    with m2:
        st.metric("Rejection Probability", f"{rejected_pct:.1f}%")
    with m3:
        emi_val = loan_amount / loan_term if loan_term > 0 else 0
        st.metric("Monthly EMI", f"₹{emi_val*1000:,.0f}")

    # Key factors
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.68rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;font-weight:500;margin-bottom:0.75rem'>— Key Factors</p>", unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    total_inc    = applicant_income + coapplicant_income
    credit_label = "Good ✓" if credit_history == 1.0 else "Bad ✗"

    for col, icon, label, value in [
        (f1, "🏦", "Credit History", credit_label),
        (f2, "💰", "Total Income",   f"₹{total_inc:,}"),
        (f3, "🏠", "Property Area",  property_area),
        (f4, "📋", "Loan Amount",    f"₹{loan_amount}K"),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(99,102,241,0.15);
            border-radius:12px;padding:1rem;text-align:center'>
                <div style='font-size:1.4rem'>{icon}</div>
                <div style='font-size:0.62rem;color:#475569;text-transform:uppercase;
                letter-spacing:0.08em;margin:0.3rem 0'>{label}</div>
                <div style='font-size:0.88rem;color:#e2e8f0;font-weight:500'>{value}</div>
            </div>""", unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:3rem 0 1rem;font-size:0.72rem;
color:#334155;letter-spacing:0.06em'>
    Made by Anuridhi saini using Streamlit &nbsp;·&nbsp; Powered by XGBoost & scikit-learn
</div>
""", unsafe_allow_html=True)
