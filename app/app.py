import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS FOR PROFESSIONAL DARK THEME ----------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f36 100%);
    }
    
    /* Sidebar - Dark theme */
    .stSidebar {
        background: rgba(10, 14, 26, 0.98) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    .stSidebar .st-emotion-cache-1kss9tm {
        background: rgba(10, 14, 26, 0.98) !important;
    }
    
    /* Sidebar content */
    .stSidebar .st-emotion-cache-155jwzh {
        background: rgba(10, 14, 26, 0.98) !important;
    }
    
    /* Sidebar labels - white for better contrast */
    .stSidebar .st-emotion-cache-1s2v671 {
        color: #e0e6ed !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    .stSidebar label {
        color: #e0e6ed !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar slider values */
    .stSidebar .st-emotion-cache-152mi7 {
        color: #e0e6ed !important;
        font-weight: 600 !important;
    }
    
    .stSidebar .st-emotion-cache-86wq5c p {
        color: #e0e6ed !important;
    }
    
    /* Sidebar select box */
    .stSidebar .stSelectbox div[data-baseweb="select"] {
        background: rgba(26, 31, 54, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
    }
    
    .stSidebar .stSelectbox div[data-baseweb="select"]:hover {
        border-color: rgba(102, 126, 234, 0.5) !important;
    }
    
    .stSidebar .stSelectbox .st-bx {
        color: #e0e6ed !important;
    }
    
    /* Sidebar slider track */
    .stSidebar .stSlider .st-an {
        background: rgba(255, 255, 255, 0.08) !important;
    }
    
    .stSidebar .stSlider .st-ef {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
    }
    
    /* Sidebar slider thumb */
    .stSidebar .stSlider .st-emotion-cache-11xx4re {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border: 2px solid #667eea !important;
    }
    
    /* Sidebar header */
    .stSidebar h2 {
        color: #e0e6ed !important;
        font-weight: 600 !important;
    }
    
    .stSidebar p {
        color: #8892b0 !important;
    }
    
    /* Sidebar dividers */
    .stSidebar hr {
        border-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Hide tooltip icons */
    .stTooltipIcon {
        display: none !important;
    }
    
    /* Hide collapse button */
    .stSidebar .st-emotion-cache-qmp9ai {
        display: none !important;
    }
    
    /* Sidebar scrollbar */
    .stSidebar ::-webkit-scrollbar {
        width: 4px;
    }
    
    .stSidebar ::-webkit-scrollbar-track {
        background: rgba(26, 31, 54, 0.3);
    }
    
    .stSidebar ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 4px;
    }
    
    /* Cards */
    .css-1r6slb0, .css-1v3fvcr, .css-1wrcr25 {
        background: rgba(26, 31, 54, 0.8) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e0e6ed !important;
        font-weight: 600 !important;
    }
    
    /* Metrics */
    .css-1xarl3l {
        background: rgba(26, 31, 54, 0.6) !important;
        border-radius: 10px !important;
        padding: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .css-1xarl3l label {
        color: #8892b0 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .css-1xarl3l div {
        color: #e0e6ed !important;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Success/Error/Warning boxes */
    .stAlert {
        border-radius: 10px !important;
        border: none !important;
        backdrop-filter: blur(10px);
    }
    
    .stAlert[data-baseweb="notification"] {
        background: rgba(26, 31, 54, 0.9) !important;
        border-left: 4px solid #667eea !important;
    }
    
    /* Dataframe */
    .dataframe {
        background: rgba(26, 31, 54, 0.6) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    .dataframe th {
        background: rgba(102, 126, 234, 0.2) !important;
        color: #e0e6ed !important;
    }
    
    .dataframe td {
        color: #8892b0 !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(26, 31, 54, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    /* Metric cards with glow */
    .metric-card {
        background: rgba(26, 31, 54, 0.8);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .metric-label {
        color: #8892b0;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .metric-value {
        color: #e0e6ed;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 5px;
    }
    
    /* Risk badge */
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 4px 16px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 4px 16px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.05) !important;
        margin: 2rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    path = "models/churn_pipeline.pkl"
    if not os.path.exists(path):
        return None
    return joblib.load(path)

model = load_model()

if model is None:
    st.error("❌ Model not found. Please train first.")
    st.stop()

# ---------------- HEADER ----------------
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="font-size: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        📊 Customer Churn Predictor
    </h1>
    <p style="color: #8892b0; font-size: 1.1rem; margin-top: -10px;">
        Production-ready ML system · Real-time predictions · Actionable insights
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT UI ----------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h2 style="color: #e0e6ed; font-size: 1.3rem; margin-bottom: 5px;">🎯 Customer Information</h2>
        <p style="color: #8892b0; font-size: 0.85rem;">Enter customer details for churn prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
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
def engineer_features(df):
    df = df.copy()
    
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    sub_map = {'Basic': 0, 'Standard': 1, 'Premium': 2}
    df['Subscription Type'] = df['Subscription Type'].map(sub_map)
    contract_map = {'Monthly': 0, 'Quarterly': 1, 'Yearly': 2}
    df['Contract Length'] = df['Contract Length'].map(contract_map)
    
    df['Tenure_Group'] = pd.cut(
        df['Tenure'],
        bins=[0, 6, 12, 24, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    df['Avg_Monthly_Spend'] = df['Total Spend'] / (df['Tenure'] + 1)
    df['Support_Intensity'] = df['Support Calls'] / (df['Tenure'] + 1)
    df['Payment_Reliability'] = 1 / (df['Payment Delay'] + 1)
    df['Risk_Score'] = (
        (df['Support Calls'] * 0.3) +
        (df['Payment Delay'] * 0.2) +
        (1 / (df['Tenure'] + 1) * 0.3) +
        (df['Usage Frequency'] * 0.2)
    )
    
    expected_order = [
        'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 
        'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend',
        'Tenure_Group', 'Avg_Monthly_Spend', 'Support_Intensity', 
        'Payment_Reliability', 'Risk_Score'
    ]
    
    df = df[expected_order]
    df = df.astype(float)
    return df

# ---------------- INPUT DATA ----------------
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

input_df_processed = engineer_features(input_df)

# ---------------- METRICS ROW ----------------
st.markdown("### 📈 Customer Profile")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Age</div>
        <div class="metric-value">{age}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Tenure</div>
        <div class="metric-value">{tenure} months</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Contract</div>
        <div class="metric-value">{contract}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Support Calls</div>
        <div class="metric-value">{support}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Spend</div>
        <div class="metric-value">${spend:,}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    
    with st.spinner("Analyzing customer data..."):
        try:
            prediction = model.predict(input_df_processed)[0]
            probability = model.predict_proba(input_df_processed)[0][1]
            
            # ---------------- RESULT DISPLAY ----------------
            st.markdown("### 📊 Prediction Result")
            
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                if prediction == 1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(245, 87, 108, 0.2), rgba(240, 147, 251, 0.2));
                                border-radius: 16px; padding: 30px; border: 1px solid rgba(245, 87, 108, 0.3);
                                text-align: center;">
                        <div class="risk-high">⚠️ HIGH RISK</div>
                        <div style="color: #f5576c; font-size: 3rem; font-weight: 700; margin: 20px 0;">
                            {probability*100:.1f}%
                        </div>
                        <div style="color: #8892b0; font-size: 0.9rem;">Churn Probability</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(79, 172, 254, 0.2), rgba(0, 242, 254, 0.2));
                                border-radius: 16px; padding: 30px; border: 1px solid rgba(79, 172, 254, 0.3);
                                text-align: center;">
                        <div class="risk-low">✅ LOW RISK</div>
                        <div style="color: #4facfe; font-size: 3rem; font-weight: 700; margin: 20px 0;">
                            {probability*100:.1f}%
                        </div>
                        <div style="color: #8892b0; font-size: 0.9rem;">Churn Probability</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    title = {'text': "Churn Risk Score", 'font': {'color': '#e0e6ed'}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickfont': {'color': '#8892b0'}},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(79, 172, 254, 0.2)'},
                            {'range': [30, 70], 'color': 'rgba(255, 193, 7, 0.2)'},
                            {'range': [70, 100], 'color': 'rgba(245, 87, 108, 0.2)'}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': probability * 100
                        }
                    },
                    domain = {'x': [0, 1], 'y': [0, 1]}
                ))
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#e0e6ed'},
                    height=250,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # ---------------- INSIGHTS SECTION ----------------
            st.markdown("---")
            st.markdown("### 💡 Actionable Insights")
            
            if prediction == 1:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    <div style="background: rgba(245, 87, 108, 0.1); border-radius: 12px; padding: 20px;
                                border-left: 4px solid #f5576c;">
                        <h4 style="color: #f5576c; margin: 0;">🚨 Immediate Action</h4>
                        <p style="color: #8892b0; font-size: 0.9rem; margin-top: 10px;">
                            • Offer 20% retention discount<br>
                            • Schedule priority support call<br>
                            • Upgrade to Premium at 50% off
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown("""
                    <div style="background: rgba(255, 193, 7, 0.1); border-radius: 12px; padding: 20px;
                                border-left: 4px solid #ffc107;">
                        <h4 style="color: #ffc107; margin: 0;">🎯 Engagement</h4>
                        <p style="color: #8892b0; font-size: 0.9rem; margin-top: 10px;">
                            • Send personalized offers<br>
                            • Improve onboarding experience<br>
                            • Reduce payment friction
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown("""
                    <div style="background: rgba(102, 126, 234, 0.1); border-radius: 12px; padding: 20px;
                                border-left: 4px solid #667eea;">
                        <h4 style="color: #667eea; margin: 0;">📈 Strategy</h4>
                        <p style="color: #8892b0; font-size: 0.9rem; margin-top: 10px;">
                            • Convert to yearly plan<br>
                            • Loyalty rewards program<br>
                            • Proactive support outreach
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    <div style="background: rgba(79, 172, 254, 0.1); border-radius: 12px; padding: 20px;
                                border-left: 4px solid #4facfe;">
                        <h4 style="color: #4facfe; margin: 0;">✅ Status</h4>
                        <p style="color: #8892b0; font-size: 0.9rem; margin-top: 10px;">
                            • Customer is stable<br>
                            • Low churn probability<br>
                            • Satisfied with service
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown("""
                    <div style="background: rgba(0, 242, 254, 0.1); border-radius: 12px; padding: 20px;
                                border-left: 4px solid #00f2fe;">
                        <h4 style="color: #00f2fe; margin: 0;">💰 Opportunity</h4>
                        <p style="color: #8892b0; font-size: 0.9rem; margin-top: 10px;">
                            • Upsell to Premium<br>
                            • Referral program<br>
                            • Annual subscription discount
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown("""
                    <div style="background: rgba(102, 126, 234, 0.1); border-radius: 12px; padding: 20px;
                                border-left: 4px solid #667eea;">
                        <h4 style="color: #667eea; margin: 0;">🚀 Growth</h4>
                        <p style="color: #8892b0; font-size: 0.9rem; margin-top: 10px;">
                            • Cross-sell services<br>
                            • Engage with content<br>
                            • Loyalty program benefits
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # ---------------- FEATURE IMPORTANCE ----------------
            st.markdown("---")
            with st.expander("🔍 View Detailed Feature Analysis"):
                feature_df = pd.DataFrame({
                    'Feature': [
                        'Age', 'Gender', 'Tenure', 'Usage Frequency', 
                        'Support Calls', 'Payment Delay', 'Subscription Type',
                        'Contract Length', 'Total Spend', 'Tenure Group',
                        'Avg Monthly Spend', 'Support Intensity', 
                        'Payment Reliability', 'Risk Score'
                    ],
                    'Value': [f"{x:.2f}" if isinstance(x, float) else str(x) 
                             for x in input_df_processed.iloc[0].values]
                })
                st.dataframe(feature_df, use_container_width=True, hide_index=True)
                
        except Exception as e:
            st.error(f"⚠️ Prediction Error: {str(e)}")

# ---------------- FOOTER ----------------
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <p style="color: #8892b0; font-size: 0.75rem;">🏆 Production ML System</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <p style="color: #8892b0; font-size: 0.75rem;">⚡ Real-time Predictions</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <p style="color: #8892b0; font-size: 0.75rem;">📊 v1.0.0</p>
    </div>
    """, unsafe_allow_html=True)