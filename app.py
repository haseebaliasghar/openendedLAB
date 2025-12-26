import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================================
# 1. PAGE CONFIG & STYLING
# ==========================================================
st.set_page_config(
    page_title="Skyline Loan Portal",
    page_icon="🏙️",
    layout="wide", # WIDE layout is better for single-page dashboards
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional "Fintech" look
st.markdown("""
    <style>
    /* Force Light Mode Styles */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    /* Input Fields */
    .stTextInput input, .stNumberInput input, .stSelectbox div, .stRadio label {
        color: #0f172a !important;
        font-weight: 500;
    }

    /* Headlines */
    h1, h2, h3 {
        color: #1e293b !important; 
    }
    
    /* The Submit Button */
    .stButton>button {
        background-color: #2563eb;
        color: white;
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
        box-shadow: 0 4px 12px rgba(37,99,235,0.2);
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# 2. LOAD MODELS
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
# 3. SIDEBAR: SUMMARY & METRICS
# ==========================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2534/2534204.png", width=60)
    st.title("Skyline Financial")
    st.markdown("Fill out the form to check your eligibility instantly.")
    st.markdown("---")
    
    # Placeholder for live metrics (will update as user types)
    st.markdown("### 📊 Application Summary")
    
    # We will display these metrics using the variables defined in the main form
    # Since Streamlit runs top-to-bottom, we usually define inputs first, 
    # but to make the sidebar dynamic, we just let the main script run and 
    # calculate metrics at the end or use session state.
    # For simplicity in this layout, we will render the form first.

# ==========================================================
# 4. MAIN FORM LAYOUT
# ==========================================================
st.markdown("## 📋 Loan Application")

# Create two main columns: Left for Inputs, Right for "Quick Stats"
col_main, col_spacer = st.columns([1, 0.05]) # Just using main width

with col_main:
    # --- SECTION A: CRITICAL FINANCIALS ---
    with st.container():
        st.subheader("💰 Financial Request")
        c1, c2, c3 = st.columns(3)
        with c1:
            income = st.number_input("Annual Income ($)", value=5000000, step=100000, help="Your total yearly income")
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
            # Map back to model expected values (0/1 logic)
            # Logic: If index 0 selected -> "No", if index 1 -> "Yes"
            is_self_employed_val = "Yes" if employed == "Self-Employed" else "No"
        with c3:
            dependents = st.slider("Dependents", 0, 5, 2, help="Number of people financially dependent on you")

        # CIBIL Score needs prominence
        st.markdown("<br>", unsafe_allow_html=True)
        cibil = st.slider("Credit Score (CIBIL)", 300, 900, 750, help="Higher is better")
        
        # CIBIL Color Indicator
        if cibil < 550:
            st.caption("🔴 Poor Credit")
        elif cibil < 700:
            st.caption("🟡 Average Credit")
        else:
            st.caption("🟢 Good Credit")

    st.markdown("---")

    # --- SECTION C: ASSETS (COLLAPSIBLE) ---
    # We hide this in an expander because it takes up a lot of space
    with st.expander("➕ Assets & Collateral Information (Click to Expand)", expanded=False):
        st.info("Enter the value of assets you currently own. Enter 0 if not applicable.")
        ac1, ac2 = st.columns(2)
        with ac1:
            residential = st.number_input("Residential Assets Value ($)", value=0, step=50000)
            commercial = st.number_input("Commercial Assets Value ($)", value=0, step=50000)
        with ac2:
            luxury = st.number_input("Luxury Assets Value ($)", value=0, step=50000)
            bank_assets = st.number_input("Bank Asset Value ($)", value=0, step=50000)

# ==========================================================
# 5. REAL-TIME METRICS (IN SIDEBAR)
# ==========================================================
# Now that variables exist, we update sidebar
with st.sidebar:
    # Simple logic to show a ratio
    ratio = loan_amount / (income + 1) # Avoid div by zero
    st.metric("Loan-to-Income Ratio", f"{ratio:.1f}x", delta="Lower is better" if ratio > 5 else None, delta_color="inverse")
    
    total_assets = residential + commercial + luxury + bank_assets
    st.metric("Total Reported Assets", f"${total_assets:,.0f}")

    st.markdown("---")
    # THE BIG BUTTON
    submit_btn = st.button("🚀 Check Eligibility Now")

# ==========================================================
# 6. PREDICTION LOGIC
# ==========================================================
if submit_btn:
    if model is None:
        st.error("Error: Model files not found.")
    else:
        # Create a container for the result to make it pop
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
                # Encode Strings
                # Note: We use .strip() to ensure no whitespace issues
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
                with result_container:
                    if status.strip() == "Approved":
                        st.markdown("""
                            <div style="background-color: #dcfce7; border: 2px solid #22c55e; border-radius: 10px; padding: 20px; text-align: center; margin-top: 20px;">
                                <h1 style="color: #15803d; margin: 0;">🎉 LOAN APPROVED</h1>
                                <p style="color: #166534; font-size: 18px;">Based on the details provided, you are eligible for this loan.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown("""
                            <div style="background-color: #fee2e2; border: 2px solid #ef4444; border-radius: 10px; padding: 20px; text-align: center; margin-top: 20px;">
                                <h1 style="color: #991b1b; margin: 0;">❌ LOAN REJECTED</h1>
                                <p style="color: #7f1d1d; font-size: 18px;">We are unable to approve this application at this time.</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Show detailed breakdown in an expander below result
                    with st.expander("View Analysis Details"):
                        st.write(f"**AI Confidence:** {confidence:.1f}%")
                        if cibil < 500:
                            st.write("- **Critical Factor:** Your CIBIL score is very low.")
                        if ratio > 8:
                            st.write("- **Critical Factor:** Loan amount is too high compared to income.")

            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.info("Ensure your model .pkl files match the input format.")
