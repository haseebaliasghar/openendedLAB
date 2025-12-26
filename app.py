import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# ==========================================================
# 1. CONFIGURATION & CUSTOM CSS (The "Pretty" Part)
# ==========================================================
st.set_page_config(
    page_title="Skyline Financial",
    page_icon="🏙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load Models (Cached)
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

# Custom CSS to override Streamlit defaults
st.markdown("""
    <style>
    /* 1. Global Background & Fonts */
    .stApp {
        background-color: #f4f6f9;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* 2. Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 3. The "Card" Container */
    div.block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .main-card {
        background-color: white;
        padding: 3rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid #e1e4e8;
    }
    
    /* 4. Custom Headers */
    h1 {
        color: #1e293b;
        font-weight: 700;
        margin-bottom: 0px;
    }
    h3 {
        color: #64748b;
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0px;
        margin-bottom: 2rem;
    }
    
    /* 5. Progress Bar Color */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #3b82f6, #06b6d4);
    }
    
    /* 6. Button Styling */
    /* Primary Button (Next) */
    div.stButton > button[kind="primary"] {
        background-color: #0f172a;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #334155;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    /* Secondary Button (Back) */
    div.stButton > button[kind="secondary"] {
        background-color: white;
        color: #64748b;
        border: 1px solid #cbd5e1;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #f1f5f9;
        color: #1e293b;
    }
    
    /* 7. Input Field Styling */
    div[data-baseweb="input"] {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# 2. SESSION STATE & NAVIGATION
# ==========================================================
if 'step' not in st.session_state:
    st.session_state.step = 1

# Initialize variables
form_keys = ['dependents', 'education', 'employed', 'income', 'bank_assets', 
             'cibil', 'residential', 'commercial', 'luxury', 'loan_amt', 'loan_term']

for key in form_keys:
    if key not in st.session_state:
        # Defaults
        if key == 'cibil': st.session_state[key] = 700
        elif key == 'loan_term': st.session_state[key] = 20
        elif key == 'income': st.session_state[key] = 5000000
        elif key == 'loan_amt': st.session_state[key] = 10000000
        elif key == 'dependents': st.session_state[key] = 0
        else: st.session_state[key] = 0

def next_step(): st.session_state.step += 1
def prev_step(): st.session_state.step -= 1
def restart(): 
    st.session_state.step = 1
    st.rerun()

# ==========================================================
# 3. UI HEADER (Custom HTML)
# ==========================================================
# We wrap everything in a container to simulate a "App Card"
with st.container():
    col_spacer_l, col_main, col_spacer_r = st.columns([1, 6, 1])
    
    with col_main:
        # -- LOGO AREA --
        st.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="font-size: 60px; line-height: 1;">🏙️</div>
                <h1 style="font-size: 2.5rem; letter-spacing: -1px;">Skyline Financial</h1>
                <p style="color: #64748b; font-size: 1rem;">AI-Powered Loan Assessment System</p>
            </div>
        """, unsafe_allow_html=True)

        # -- PROGRESS INDICATOR --
        # 5 Steps total
        progress = (st.session_state.step / 5)
        st.progress(progress)
        
        # Step label
        steps = ["Profile", "Income", "Assets", "Loan Details", "Result"]
        current_label = steps[st.session_state.step - 1] if st.session_state.step <= 5 else "Done"
        st.markdown(f"<p style='text-align: right; color: #94a3b8; font-size: 0.8rem; margin-top: -10px;'>Step {st.session_state.step}/5: {current_label}</p>", unsafe_allow_html=True)
        
        st.markdown("---")

        # ==========================================================
        # STEP 1: PERSONAL PROFILE
        # ==========================================================
        if st.session_state.step == 1:
            st.markdown("### 👤 Who are you?")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Education**")
                # Visual selection
                st.session_state.education = st.radio(
                    "Education",
                    options=[0, 1],
                    format_func=lambda x: "🎓 Graduate" if x==0 else "📘 Not Graduate",
                    label_visibility="collapsed"
                )
            with c2:
                st.markdown("**Employment**")
                st.session_state.employed = st.radio(
                    "Employed",
                    options=[0, 1],
                    format_func=lambda x: "💼 Salaried" if x==0 else "🚀 Self-Employed",
                    label_visibility="collapsed"
                )

            st.markdown("---")
            st.markdown("**Dependents**")
            st.session_state.dependents = st.slider(
                "Number of people depending on your income",
                0, 5, st.session_state.dependents
            )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Continue ➔", type="primary", use_container_width=True):
                next_step()
                st.rerun()

        # ==========================================================
        # STEP 2: FINANCIALS
        # ==========================================================
        elif st.session_state.step == 2:
            st.markdown("### 💰 Financial Strength")
            
            # Using Number Inputs with formatting for large numbers
            st.markdown("**Annual Income**")
            st.session_state.income = st.number_input(
                "Yearly Income",
                min_value=0, step=100000,
                value=st.session_state.income,
                format="%d",
                label_visibility="collapsed"
            )
            st.caption(f"Formatted: {st.session_state.income:,.0f}")

            st.markdown("**Bank Assets**")
            st.session_state.bank_assets = st.number_input(
                "Total Bank Savings/Investments",
                min_value=0, step=100000,
                value=st.session_state.bank_assets,
                format="%d",
                label_visibility="collapsed"
            )
            st.caption(f"Formatted: {st.session_state.bank_assets:,.0f}")

            st.markdown("---")
            st.markdown(f"**CIBIL Score: {st.session_state.cibil}**")
            st.session_state.cibil = st.slider(
                "Score", 300, 900, st.session_state.cibil, label_visibility="collapsed"
            )
            
            # Color-coded feedback for CIBIL
            if st.session_state.cibil >= 750:
                st.success("Excellent Score")
            elif st.session_state.cibil >= 600:
                st.warning("Average Score")
            else:
                st.error("Low Score")

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 2])
            with c1:
                if st.button("Back"): prev_step(); st.rerun()
            with c2:
                if st.button("Next Step ➔", type="primary", use_container_width=True): next_step(); st.rerun()

        # ==========================================================
        # STEP 3: ASSETS
        # ==========================================================
        elif st.session_state.step == 3:
            st.markdown("### 🏠 Assets & Collateral")
            
            st.markdown("**Residential Value**")
            st.session_state.residential = st.number_input("Residential", value=st.session_state.residential, step=100000, label_visibility="collapsed")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Commercial Value**")
                st.session_state.commercial = st.number_input("Commercial", value=st.session_state.commercial, step=100000, label_visibility="collapsed")
            with c2:
                st.markdown("**Luxury Assets**")
                st.session_state.luxury = st.number_input("Luxury", value=st.session_state.luxury, step=100000, label_visibility="collapsed")

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 2])
            with c1:
                if st.button("Back"): prev_step(); st.rerun()
            with c2:
                if st.button("Next Step ➔", type="primary", use_container_width=True): next_step(); st.rerun()

        # ==========================================================
        # STEP 4: LOAN DETAILS
        # ==========================================================
        elif st.session_state.step == 4:
            st.markdown("### 📝 Loan Request")
            
            st.markdown("**Loan Amount Required**")
            st.session_state.loan_amt = st.number_input(
                "Loan Amount", 
                value=st.session_state.loan_amt, 
                step=500000,
                format="%d",
                label_visibility="collapsed"
            )
            st.caption(f"Requesting: {st.session_state.loan_amt:,.0f}")

            st.markdown("**Repayment Term**")
            st.session_state.loan_term = st.slider(
                "Years", 1, 30, st.session_state.loan_term
            )
            st.caption(f"{st.session_state.loan_term} Years")

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 2])
            with c1:
                if st.button("Back"): prev_step(); st.rerun()
            with c2:
                if st.button("🚀 Submit Application", type="primary", use_container_width=True): next_step(); st.rerun()

        # ==========================================================
        # STEP 5: RESULT
        # ==========================================================
        elif st.session_state.step == 5:
            if model is None:
                st.error("Model not found. Please upload .pkl files.")
            else:
                with st.spinner("Processing Application..."):
                    time.sleep(1)
                    
                    # 1. Prepare Inputs (Clean string formatting included here)
                    # NOTE: We ensure strings match the clean training data (no extra spaces)
                    edu_str = "Graduate" if st.session_state.education == 0 else "Not Graduate"
                    emp_str = "No" if st.session_state.employed == 0 else "Yes" # 0=Salaried(No self emp), 1=Self Emp

                    input_data = pd.DataFrame({
                        'no_of_dependents': [st.session_state.dependents],
                        'education': [edu_str],
                        'self_employed': [emp_str],
                        'income_annum': [st.session_state.income],
                        'loan_amount': [st.session_state.loan_amt],
                        'loan_term': [st.session_state.loan_term],
                        'cibil_score': [st.session_state.cibil],
                        'residential_assets_value': [st.session_state.residential],
                        'commercial_assets_value': [st.session_state.commercial],
                        'luxury_assets_value': [st.session_state.luxury],
                        'bank_asset_value': [st.session_state.bank_assets]
                    })

                    try:
                        # 2. Encode
                        input_data['education'] = feature_encoders['education'].transform(input_data['education'])
                        input_data['self_employed'] = feature_encoders['self_employed'].transform(input_data['self_employed'])
                        
                        # 3. Predict
                        pred = model.predict(input_data)
                        prob = model.predict_proba(input_data)
                        status = target_encoder.inverse_transform(pred)[0]

                        # 4. Display Result
                        st.markdown("---")
                        if status.strip() == "Approved":
                            st.markdown("""
                                <div style="background-color: #d1fae5; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #10b981;">
                                    <h2 style="color: #065f46; margin:0;">🎉 APPROVED</h2>
                                    <p style="color: #047857;">Your loan application has been accepted.</p>
                                </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                        else:
                            st.markdown("""
                                <div style="background-color: #fee2e2; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #ef4444;">
                                    <h2 style="color: #991b1b; margin:0;">❌ REJECTED</h2>
                                    <p style="color: #b91c1c;">We cannot process your loan at this time.</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Show Confidence
                        conf = np.max(prob) * 100
                        st.metric("AI Confidence Score", f"{conf:.1f}%")

                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.info("Tip: Ensure your training data was cleaned of whitespace.")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Start New Application", type="secondary", use_container_width=True):
                restart()
