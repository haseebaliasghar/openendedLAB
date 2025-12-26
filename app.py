import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================================
# 1. PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Skyline Loan Portal",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 2. ADAPTIVE CSS (THEME AGNOSTIC)
# ==========================================================
# We use CSS variables (var(--...)) so colors auto-flip based on theme
st.markdown("""
    <style>
    /* Metric Cards - Adaptive Background */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--text-color);
        padding: 15px;
        border-radius: 10px;
        /* Slight transparency for the border to make it subtle */
        border-color: rgba(128, 128, 128, 0.2);
    }

    /* Headlines - Use native text color */
    h1, h2, h3 {
        color: var(--text-color) !important; 
    }
    
    /* Submit Button - Standard Blue that looks good in both modes */
    .stButton>button {
        background-color: #2563eb;
        color: white !important; /* Always white text on blue button */
        font-size: 18px;
        padding: 10px;
        border-radius: 8px;
        border: none;
        width: 100%;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 12px rgba(37,99,235,0.3);
    }
    
    /* Remove top padding to make it look more like a dashboard */
    div.block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. LOAD MODELS
# ==========================================================
@st.cache_resource
def load_artifacts():
    try:
        with open('random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('feature_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('target_encoder.pkl', 'rb') as f:
            target_encoder = pickle.load(f)
        return model, encoders, target_encoder
    except FileNotFoundError:
        return None, None, None

model, feature_encoders, target_encoder = load_artifacts()

# ==========================================================
# 4. SIDEBAR: SUMMARY & METRICS
# ==========================================================
with st.sidebar:
    st.title("🏙️ Skyline Financial")
    st.markdown("Fill out the form to check loan eligibility instantly.")
    st.markdown("---")
    
    st.markdown("### 📊 Live Summary")
    # These placeholders will be updated at the end of the script
    # to reflect the user's current inputs
    ratio_metric = st.empty()
    assets_metric = st.empty()

# ==========================================================
# 5. MAIN FORM LAYOUT
# ==========================================================
st.markdown("## 📋 Loan Application Dashboard")

# Create two main columns: Left for Inputs, Right for spacing
col_main, col_spacer = st.columns([1, 0.01]) 

with col_main:
    # --- SECTION A: CRITICAL FINANCIALS ---
    with st.container():
        st.subheader("💰 Financial Request")
        c1, c2, c3 = st.columns(3)
        with c1:
            income = st.number_input("Annual Income ($)", value=5000000, step=100000, help="Total yearly income")
        with c2:
            loan_amount = st.number_input("Loan Amount Required ($)", value=10000000, step=100000)
        with c3:
            loan_term = st.slider("Loan Term (Years)", 1, 30, 20)

    st.markdown("---")

    # --- SECTION B: PERSONAL PROFILE ---
    with st.container():
        st.subheader("👤 Personal Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            # Education
            edu_display = ["Graduate", "Not Graduate"]
            education = st.selectbox("Education Level", edu_display)
        with c2:
            # Employment
            emp_display = ["Salaried / Other", "Self-Employed"]
            employed = st.selectbox("Employment Type", emp_display)
            # Logic: Self-Employed -> Yes/No mapping
            is_self_employed_val = "Yes" if employed == "Self-Employed" else "No"
        with c3:
            dependents = st.slider("Dependents", 0, 5, 2, help="Number of financial dependents")

        st.markdown("<br>", unsafe_allow_html=True)
        # CIBIL Score Slider
        cibil = st.slider("Credit Score (CIBIL)", 300, 900, 750, help="Higher is better")
        
        # Dynamic CIBIL Feedback
        if cibil < 550:
            st.caption("🔴 :red[Poor Credit Score]")
        elif cibil < 700:
            st.caption("🟡 :orange[Average Credit Score]")
        else:
            st.caption("🟢 :green[Good Credit Score]")

    st.markdown("---")

    # --- SECTION C: ASSETS (COLLAPSIBLE) ---
    with st.expander("➕ Assets & Collateral (Click to Expand)", expanded=False):
        st.info("Enter the value of assets you currently own. Enter 0 if not applicable.")
        ac1, ac2 = st.columns(2)
        with ac1:
            residential = st.number_input("Residential Assets Value ($)", value=0, step=50000)
            commercial = st.number_input("Commercial Assets Value ($)", value=0, step=50000)
        with ac2:
            luxury = st.number_input("Luxury Assets Value ($)", value=0, step=50000)
            bank_assets = st.number_input("Bank Asset Value ($)", value=0, step=50000)

# ==========================================================
# 6. UPDATE SIDEBAR METRICS
# ==========================================================
# We calculate these now that the inputs are defined
ratio = loan_amount / (income + 1) # Avoid div by zero
total_assets = residential + commercial + luxury + bank_assets

# Update the placeholders we created earlier
ratio_metric.metric("Loan-to-Income Ratio", f"{ratio:.1f}x", delta="Lower is better" if ratio > 5 else None, delta_color="inverse")
assets_metric.metric("Total Reported Assets", f"${total_assets:,.0f}")

with st.sidebar:
    st.markdown("---")
    submit_btn = st.button("🚀 Check Eligibility Now")

# ==========================================================
# 7. PREDICTION LOGIC
# ==========================================================
if submit_btn:
    if model is None:
        st.error("⚠️ Error: Model files not found. Please upload .pkl files.")
    else:
        result_container = st.container()
        
        with st.spinner("Analyzing Financial Profile..."):
            # Prepare Data
            input_data = pd.DataFrame({
                'no_of_dependents': [dependents],
                'education': ["Graduate" if education == "Graduate" else "Not Graduate"],
                'self_employed': [is_self_employed_val],
                'income_annum': [income],
                'loan_amount': [loan_amount],
                'loan_term': [loan_term],
                'cibil_score': [cibil],
                'residential_assets_value': [residential],
                'commercial_assets_value': [commercial],
                'luxury_assets_value': [luxury],
                'bank_asset_value': [bank_assets]
            })

            try:
                # Encode Strings (with strip() for safety)
                col_edu = input_data['education'].str.strip()
                col_emp = input_data['self_employed'].str.strip()
                
                input_data['education'] = feature_encoders['education'].transform(col_edu)
                input_data['self_employed'] = feature_encoders['self_employed'].transform(col_emp)

                # Predict
                pred = model.predict(input_data)
                prob = model.predict_proba(input_data)
                status = target_encoder.inverse_transform(pred)[0]
                confidence = np.max(prob) * 100

                # --- DISPLAY RESULT ---
                # We use hardcoded colors for the result box text/bg so they 
                # ALWAYS look like this (Green/Red) regardless of theme.
                with result_container:
                    if status.strip() == "Approved":
                        st.balloons()
                        st.markdown("""
                            <div style="background-color: #dcfce7; border: 2px solid #22c55e; border-radius: 10px; padding: 20px; text-align: center; margin-top: 10px; margin-bottom: 20px;">
                                <h1 style="color: #15803d; margin: 0;">🎉 LOAN APPROVED</h1>
                                <p style="color: #166534; font-size: 18px; margin-top: 5px;">You are eligible for this loan.</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div style="background-color: #fee2e2; border: 2px solid #ef4444; border-radius: 10px; padding: 20px; text-align: center; margin-top: 10px; margin-bottom: 20px;">
                                <h1 style="color: #991b1b; margin: 0;">❌ LOAN REJECTED</h1>
                                <p style="color: #7f1d1d; font-size: 18px; margin-top: 5px;">We cannot approve this application.</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Analysis Details
                    with st.expander("View Analysis Details"):
                        st.write(f"**AI Confidence Score:** {confidence:.1f}%")
                        if cibil < 500:
                            st.write("⚠️ **Critical Factor:** Low CIBIL score.")
                        if ratio > 8:
                            st.write("⚠️ **Critical Factor:** High Loan-to-Income ratio.")

            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.info("Ensure your training data matches the input format.")
