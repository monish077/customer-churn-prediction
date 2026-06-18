import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📊 Customer Churn Prediction")
st.markdown("Production-ready ML system with proper encoding")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    path = "models/churn_model.pkl"
    
    if not os.path.exists(path):
        return None
    
    return joblib.load(path)

model = load_model()

if model is None:
    st.error("❌ Model not found. Run training first: python run.py")
    st.stop()

st.success("✅ Production ML Model Loaded")

# ---------------- INPUT UI ----------------
with st.sidebar:
    st.header("Customer Information")
    
    age = st.slider("Age", 18, 70, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.slider("Tenure (months)", 1, 72, 12)
    usage = st.slider("Usage Frequency", 1, 50, 20)
    support = st.slider("Support Calls", 0, 10, 3)
    delay = st.slider("Payment Delay", 0, 30, 5)
    sub_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Yearly"])
    spend = st.slider("Total Spend", 100, 5000, 1500)

# ---------------- FEATURE ENGINEERING ----------------
def engineer_and_encode_features(df):
    """Apply feature engineering and encode all categorical variables"""
    df = df.copy()
    
    # --- ENCODE CATEGORICAL VARIABLES ---
    # Gender: Male=0, Female=1
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    # Subscription Type: Basic=0, Standard=1, Premium=2
    sub_map = {'Basic': 0, 'Standard': 1, 'Premium': 2}
    df['Subscription Type'] = df['Subscription Type'].map(sub_map)
    
    # Contract Length: Monthly=0, Quarterly=1, Yearly=2
    contract_map = {'Monthly': 0, 'Quarterly': 1, 'Yearly': 2}
    df['Contract Length'] = df['Contract Length'].map(contract_map)
    
    # --- CREATE ENGINEERED FEATURES ---
    # Tenure Group
    df['Tenure_Group'] = pd.cut(
        df['Tenure'],
        bins=[0, 6, 12, 24, 100],
        labels=[0, 1, 2, 3]  # Encode as numbers: New=0, Regular=1, Loyal=2, VIP=3
    ).astype(int)
    
    # Avg Monthly Spend
    df['Avg_Monthly_Spend'] = df['Total Spend'] / (df['Tenure'] + 1)
    
    # Support Intensity
    df['Support_Intensity'] = df['Support Calls'] / (df['Tenure'] + 1)
    
    # Payment Reliability
    df['Payment_Reliability'] = 1 / (df['Payment Delay'] + 1)
    
    # Risk Score
    df['Risk_Score'] = (
        (df['Support Calls'] * 0.3) +
        (df['Payment Delay'] * 0.2) +
        (1 / (df['Tenure'] + 1) * 0.3) +
        (df['Usage Frequency'] * 0.2)
    )
    
    # --- REORDER COLUMNS TO MATCH MODEL EXPECTATIONS ---
    expected_order = [
        'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 
        'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend',
        'Tenure_Group', 'Avg_Monthly_Spend', 'Support_Intensity', 
        'Payment_Reliability', 'Risk_Score'
    ]
    
    df = df[expected_order]
    
    # Convert all to float to ensure compatibility
    df = df.astype(float)
    
    return df

# ---------------- INPUT DATA ----------------
# Create base dataframe with raw features
input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Tenure": tenure,
    "Usage Frequency": usage,
    "Support Calls": support,
    "Payment Delay": delay,
    "Subscription Type": sub_type,
    "Contract Length": contract,
    "Total Spend": spend
}])

# Apply feature engineering and encoding
input_df_processed = engineer_and_encode_features(input_df)

# ---------------- DISPLAY ----------------
st.subheader("Customer Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Age", age)
    st.metric("Tenure", f"{tenure} months")

with col2:
    st.metric("Contract", contract)
    st.metric("Support Calls", support)

with col3:
    st.metric("Payment Delay", f"{delay} days")
    st.metric("Total Spend", f"${spend:,}")

with col4:
    st.metric("Avg Monthly Spend", f"${input_df_processed['Avg_Monthly_Spend'].iloc[0]:.2f}")
    st.metric("Risk Score", f"{input_df_processed['Risk_Score'].iloc[0]:.2f}")

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Churn", type="primary"):
    
    try:
        # Make prediction with processed data
        prediction = model.predict(input_df_processed)[0]
        probability = model.predict_proba(input_df_processed)[0][1]
        
        st.subheader("📊 Result")
        
        col1, col2 = st.columns(2)
        
        if prediction == 1:
            col1.error("⚠️ High Churn Risk")
            col1.metric("Probability", f"{probability*100:.1f}%")
            
            col2.warning("""
            **Recommended Actions:**
            • Offer retention discount (15-20%)
            • Upgrade to Premium plan
            • Schedule customer outreach
            • Address support issues
            """)
        else:
            col1.success("✅ Low Churn Risk")
            col1.metric("Confidence", f"{(1-probability)*100:.1f}%")
            
            col2.info("""
            **Opportunities:**
            • Upsell to higher tier
            • Referral program
            • Annual subscription discount
            • Maintain engagement
            """)
        
        # Show processed feature details
        st.markdown("---")
        st.subheader("🔍 Processed Features")
        
        # Display the features being used (with decoded values for readability)
        feature_df = pd.DataFrame({
            'Feature': [
                'Age', 'Gender (0=Male,1=Female)', 'Tenure', 'Usage Frequency', 
                'Support Calls', 'Payment Delay', 'Subscription Type (0=Basic,1=Standard,2=Premium)',
                'Contract Length (0=Monthly,1=Quarterly,2=Yearly)', 'Total Spend',
                'Tenure Group (0=New,1=Regular,2=Loyal,3=VIP)', 'Avg Monthly Spend',
                'Support Intensity', 'Payment Reliability', 'Risk Score'
            ],
            'Value': input_df_processed.iloc[0].values
        })
        st.dataframe(feature_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        
        # Show debugging info
        with st.expander("🔧 Debugging Information"):
            st.write("**Processed features being sent to model:**")
            st.write(input_df_processed.columns.tolist())
            
            st.write("\n**Feature values (all numeric):**")
            st.write(input_df_processed.iloc[0].values)
            
            st.write("\n**DataFrame:**")
            st.dataframe(input_df_processed)

# ---------------- INSIGHTS ----------------
st.markdown("---")
st.subheader("📈 Business Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🚨 High Risk Indicators:**
    - Monthly contracts
    - High support calls (>5)
    - Payment delays (>10 days)
    - Low tenure (<6 months)
    - High Risk Score (>15)
    """)

with col2:
    st.markdown("""
    **✅ Retention Strategies:**
    - Upgrade to yearly plans
    - Proactive support
    - Loyalty rewards
    - Personalized offers
    - Improve payment experience
    """)

with col3:
    st.markdown("""
    **💰 Revenue Opportunities:**
    - Cross-sell Premium
    - Annual discounts
    - Referral bonuses
    - Feature upgrades
    - Engagement programs
    """)

st.caption("🏆 Production ML System - Complete Feature Engineering & Encoding")

# ---------------- SIDEBAR HELP ----------------
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📋 Feature Engineering")
    st.markdown("""
    **Categorical Encodings:**
    - Gender: Male=0, Female=1
    - Subscription: Basic=0, Standard=1, Premium=2
    - Contract: Monthly=0, Quarterly=1, Yearly=2
    - Tenure Group: New=0, Regular=1, Loyal=2, VIP=3
    
    **Engineered Features:**
    - Avg Monthly Spend
    - Support Intensity
    - Payment Reliability
    - Risk Score
    """)