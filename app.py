import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# ==========================================================
# CONFIGURATION & STYLING
# ==========================================================
st.set_page_config(
    page_title="Skyline Loan Approval",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a clean, professional look
st.markdown("""
    <style>
    /* Hide standard Streamlit header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Progress Bar Style */
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    
    /* Card-like container for the form */
    .form-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# SESSION STATE MANAGEMENT
# ==========================================================
if 'step' not in st.session_state:
    st.session_state.step = 1

# Initialize form data in session state if not present
form_keys = [
    'dependents', 'education', 'employed', 
    'income', 'bank_assets', 'cibil',
    'residential', 'commercial', 'luxury',
    'loan_amt', 'loan_term'
]

for key in form_keys:
    if key not in st.session_state:
        # Set sensible defaults
        if key in ['education', 'employed']:
            st.session_state[key] = 0 # Index for radio
        elif key == 'cibil':
            st.session_state[key] = 700
        elif key == 'loan_term':
            st.session_state[key] = 12
        else:
            st.session_state[key] = 0

# ==========================================================
# LOAD MODELS & UTILS
# ==========================================================
@st.cache_resource
def load_artifacts():
    try:
        # Load the saved artifacts
        # NOTE: Ensure these files are in the same directory for deployment
        with open('random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('feature_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('target_encoder.pkl', 'rb') as f:
            target_encoder = pickle.load(f)
        return model, encoders, target_encoder
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please upload .pkl files.")
        return None, None, None

model, feature_encoders, target_encoder = load_artifacts()

# Navigation Functions
def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step -= 1

def restart():
    st.session_state.step = 1
    for key in form_keys:
        if key not in ['cibil', 'loan_term']:
            st.session_state[key] = 0

# ==========================================================
# APP LAYOUT
# ==========================================================

# Header
st.title("🏦 Skyline Financial")
st.markdown("### Loan Application Portal")
st.markdown("---")

# Progress Bar
# We have 4 input steps + 1 result step
progress_value = (st.session_state.step / 5) 
st.progress(progress_value)

# ==========================================================
# WIZARD STEP 1: PERSONAL PROFILE
# ==========================================================
if st.session_state.step == 1:
    st.subheader("1. Personal Profile")
    st.info("Let's start with some basic information about you.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Education Level**")
        # Using Radio with horizontal layout instead of dropdown
        education_display = ["Graduate", "Not Graduate"]
        st.session_state.education = st.radio(
            "Select your education status:",
            options=[0, 1],
            format_func=lambda x: education_display[x],
            horizontal=True,
            index=st.session_state.education,
            label_visibility="collapsed"
        )
        
    with col2:
        st.markdown("**Employment Status**")
        employed_display = ["No", "Yes"] # Matches typical "Self_Employed" logic
        st.session_state.employed = st.radio(
            "Are you self-employed?",
            options=[0, 1],
            format_func=lambda x: "Self-Employed" if x==1 else "Salaried/Other",
            horizontal=True,
            index=st.session_state.employed,
            label_visibility="collapsed"
        )

    st.markdown("**Number of Dependents**")
    st.session_state.dependents = st.slider(
        "How many people depend on your income?",
        min_value=0, max_value=5, 
        value=st.session_state.dependents,
        help="Includes children, spouse, or elderly parents."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Next: Financial Overview >"):
        next_step()
        st.rerun()

# ==========================================================
# WIZARD STEP 2: FINANCIAL HEALTH
# ==========================================================
elif st.session_state.step == 2:
    st.subheader("2. Financial Health")
    st.info("Please provide details about your income and credit history.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Annual Income**")
        st.session_state.income = st.number_input(
            "Total yearly income (in local currency)",
            min_value=0, step=10000,
            value=st.session_state.income,
            format="%d"
        )
    
    with col2:
        st.markdown("**Bank Asset Value**")
        st.session_state.bank_assets = st.number_input(
            "Total value of savings/investments",
            min_value=0, step=10000,
            value=st.session_state.bank_assets,
            format="%d",
            help="Sum of all savings accounts, FDs, and liquid assets."
        )

    st.markdown("**Credit Score (CIBIL)**")
    st.caption("Drag the slider to your current score.")
    st.session_state.cibil = st.slider(
        "CIBIL Score",
        min_value=300, max_value=900,
        value=st.session_state.cibil,
        label_visibility="collapsed"
    )
    
    # Visual feedback for CIBIL
    if st.session_state.cibil < 500:
        st.warning("⚠️ Low Credit Score detected.")
    elif st.session_state.cibil > 750:
        st.success("✅ Excellent Credit Score!")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("< Back"):
            prev_step()
            st.rerun()
    with c2:
        if st.button("Next: Asset Evaluation >"):
            next_step()
            st.rerun()

# ==========================================================
# WIZARD STEP 3: ASSETS & COLLATERAL
# ==========================================================
elif st.session_state.step == 3:
    st.subheader("3. Asset Evaluation")
    st.info("Do you own any physical assets? (Enter 0 if not applicable)")

    st.markdown("**Residential Assets**")
    st.session_state.residential = st.number_input(
        "Value of your home/apartments",
        min_value=0, step=50000,
        value=st.session_state.residential,
        format="%d"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Commercial Assets**")
        st.session_state.commercial = st.number_input(
            "Value of shops/offices",
            min_value=0, step=50000,
            value=st.session_state.commercial,
            format="%d"
        )
    with col2:
        st.markdown("**Luxury Assets**")
        st.session_state.luxury = st.number_input(
            "Value of cars/jewelry/art",
            min_value=0, step=50000,
            value=st.session_state.luxury,
            format="%d"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("< Back"):
            prev_step()
            st.rerun()
    with c2:
        if st.button("Next: Loan Details >"):
            next_step()
            st.rerun()

# ==========================================================
# WIZARD STEP 4: LOAN REQUEST
# ==========================================================
elif st.session_state.step == 4:
    st.subheader("4. Loan Details")
    st.info("Final step: What are you looking for?")

    st.markdown("**Loan Amount Requested**")
    st.session_state.loan_amt = st.number_input(
        "Enter amount",
        min_value=0, step=100000,
        value=st.session_state.loan_amt,
        format="%d"
    )

    st.markdown("**Loan Term (Years)**")
    st.session_state.loan_term = st.slider(
        "Duration of loan",
        min_value=1, max_value=30,
        value=st.session_state.loan_term,
        help="The number of years you need to repay the loan."
    )

    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("< Back"):
            prev_step()
            st.rerun()
    with c2:
        if st.button("🚀 Submit Application", type="primary"):
            next_step()
            st.rerun()

# ==========================================================
# STEP 5: PREDICTION & RESULT
# ==========================================================
elif st.session_state.step == 5:
    st.subheader("Application Status")
    
    if model is not None:
        with st.spinner("Analyzing your financial profile..."):
            time.sleep(1.5) # Simulate processing time for UX
            
            # 1. Prepare Input Data
            # Map simple UI inputs back to model expected format
            # Education: 0->Graduate, 1->Not Graduate (Check your specific encoder mapping!)
            # Note: LabelEncoder sorts alphabetically: Graduate=0, Not Graduate=1
            edu_input = "Graduate" if st.session_state.education == 0 else "Not Graduate"
            
            # Employed: 0->No, 1->Yes
            emp_input = "No" if st.session_state.employed == 0 else "Yes"
            
            # Create DataFrame with exact column names from training
            input_df = pd.DataFrame({
                'no_of_dependents': [st.session_state.dependents],
                'education': [edu_input],
                'self_employed': [emp_input],
                'income_annum': [st.session_state.income],
                'loan_amount': [st.session_state.loan_amt],
                'loan_term': [st.session_state.loan_term],
                'cibil_score': [st.session_state.cibil],
                'residential_assets_value': [st.session_state.residential],
                'commercial_assets_value': [st.session_state.commercial],
                'luxury_assets_value': [st.session_state.luxury],
                'bank_asset_value': [st.session_state.bank_assets]
            })

            # 2. Encode Categorical Variables
            try:
                # Use the loaded encoders to transform
                input_df['education'] = feature_encoders['education'].transform(input_df['education'])
                input_df['self_employed'] = feature_encoders['self_employed'].transform(input_df['self_employed'])
                
                # 3. Predict
                prediction = model.predict(input_df)
                probability = model.predict_proba(input_df)
                
                # 4. Decode Result
                result_status = target_encoder.inverse_transform(prediction)[0]
                
                # 5. Display
                if result_status.strip() == "Approved":
                    st.balloons()
                    st.success("### 🎉 Congratulations! Your Loan is Approved.")
                    st.markdown(f"**Confidence Score:** {np.max(probability)*100:.1f}%")
                    st.markdown("Our team will contact you shortly to finalize the paperwork.")
                else:
                    st.error("### ❌ Application Rejected")
                    st.markdown(f"**Analysis:** Based on the current financial parameters, we cannot approve this loan at this time.")
                    st.markdown("*Tip: Improving your CIBIL score or reducing the loan amount may help.*")
            
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.markdown("Please check inputs and try again.")

    if st.button("Start New Application"):
        restart()
        st.rerun()
