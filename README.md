#**📊 Customer Churn Prediction System**
A production-ready machine learning system that predicts customer churn with an interactive dashboard, providing real-time insights and actionable retention strategies.

https://static.streamlit.io/badges/streamlit_badge_black_white.svg
https://img.shields.io/github/stars/monish077/customer-churn-prediction
https://img.shields.io/badge/License-MIT-green.svg
https://img.shields.io/badge/python-3.11+-blue.svg

##**🌐 Live Demo**
Try the live application: https://customer-churn-prediction-mrfnyfzzj55die3asdzfvs.streamlit.app/

##**📊 Model Performance**
Our best-performing model (Random Forest) achieves excellent predictive accuracy:

Metric	Score
Accuracy	91.5%
Precision	92.3%
Recall	91.5%
F1 Score	90.7%
AUC-ROC	80.2%
Model Comparison
Model	Accuracy	Precision	Recall	F1 Score	AUC-ROC
Logistic Regression	91.3%	92.2%	91.3%	90.4%	79.8%
Random Forest	91.5%	92.3%	91.5%	90.7%	80.2%
XGBoost	90.9%	91.3%	90.9%	90.1%	80.0%

##**🚀 Features**

**Core Functionality**
✅ Real-time Predictions: Get instant churn probability scores for any customer

✅ Interactive Dashboard: User-friendly interface with sliders and dropdowns

✅ Feature Engineering: Automated feature engineering matching training data

✅ Business Insights: Actionable recommendations for customer retention

✅ Professional UI: Dark theme with modern glass-morphism design

**Technical Features**
✅ Production Ready: Deployable on Streamlit Cloud, Hugging Face, or Render

✅ ML Pipeline: Complete preprocessing and model pipeline

✅ Visual Analytics: Interactive gauge charts and metric displays

✅ Feature Analysis: Detailed breakdown of customer features

**🛠️ Tech Stack**
Category	Technologies
Frontend	Streamlit, Plotly
ML Framework	Scikit-learn, XGBoost
Data Processing	Pandas, NumPy
Model Serialization	Joblib
Visualization	Matplotlib, Seaborn, Plotly
Deployment	Streamlit Cloud

**📁 Project Structure**
text

customer-churn-prediction/
├── app/
│   └── app.py                 # Main Streamlit application with dark theme
├── models/
│   └── churn_pipeline.pkl     # Trained Random Forest model
├── data/
│   └── raw/                   # Dataset files
├── notebooks/                 # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/                       # Source code modules
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── runtime.txt                # Python 3.11 specification
├── packages.txt               # System dependencies
├── requirements.txt           # Python dependencies
├── setup.sh                   # Deployment setup script
├── run.py                     # Model training script
└── README.md                  # Documentation
🏃‍♂️ Local Setup
Prerequisites
Python 3.11 or higher

Git

Installation Steps
Clone the repository

bash
git clone https://github.com/monish077/customer-churn-prediction.git
cd customer-churn-prediction
Create and activate a virtual environment

bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
Install dependencies

bash
pip install -r requirements.txt
Train the model (optional)

bash
python run.py
Run the Streamlit app

bash
streamlit run app/app.py
The app will open in your browser at http://localhost:8501.

**🎯 How It Works**
Feature Engineering
The system creates several engineered features for better predictions:

Feature	Description
Tenure Group	Categorizes customer tenure (New, Regular, Loyal, VIP)
Avg Monthly Spend	Total spend divided by tenure
Support Intensity	Support calls divided by tenure
Payment Reliability	Inverse of payment delay
Risk Score	Composite risk metric combining multiple factors
Input Features
Feature	Range	Description
Age	18-70	Customer age
Gender	Male/Female	Customer gender
Tenure	1-72 months	Time with company
Usage Frequency	1-50	Monthly usage count
Support Calls	0-10	Number of support calls
Payment Delay	0-30 days	Days payment is delayed
Subscription Type	Basic/Standard/Premium	Service tier
Contract Length	Monthly/Quarterly/Yearly	Contract duration
Total Spend	100-5000	Total spending amount
📈 Business Impact
Retention Benefits
🎯 Early Warning: Identify at-risk customers before they churn

🎯 Targeted Campaigns: Focus retention efforts on high-risk customers

🎯 Cost Reduction: Reduce customer acquisition costs by retaining existing customers

Revenue Opportunities
💰 Upselling: Identify customers ready for premium upgrades

💰 Cross-selling: Recommend complementary services

💰 Loyalty Programs: Design effective loyalty incentives

Operational Efficiency
⚡ Automated Prediction: Reduce manual risk assessment time

⚡ Data-Driven Decisions: Make informed retention strategy decisions

⚡ Performance Monitoring: Track churn patterns over time

🚀 Deployment
Streamlit Cloud (Recommended)
Push code to GitHub repository

Visit share.streamlit.io

Connect your GitHub account

Deploy with:

Repository: monish077/customer-churn-prediction

Branch: main

Main file: app/app.py

Python version: 3.11

Hugging Face Spaces
Visit huggingface.co/new-space

Select "Streamlit" SDK

Choose Python 3.11

Upload your files

🤝 Contributing
We welcome contributions! Here's how you can help:

Fork the repository

Create a feature branch

bash
git checkout -b feature/amazing-feature
Make your changes

Commit and push

bash
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
Open a Pull Request

Contribution Guidelines
Follow existing code style

Add tests for new features

Update documentation

Ensure all tests pass

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author
Monish

GitHub: @monish077

Project Link: [customer-churn-prediction](https://customer-churn-prediction-mrfnyfzzj55die3asdzfvs.streamlit.app/)

🙏 Acknowledgments
Streamlit for the amazing framework

Scikit-learn for ML algorithms

XGBoost for gradient boosting

📞 Support
For issues, questions, or suggestions:

Open an issue

Fork the repository

Star the project ⭐

Made with ❤️ by Monish | Live Demo

📊 Quick Start Commands
bash
# Clone the repository
git clone https://github.com/monish077/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
🔧 Environment Variables
No environment variables are required for local development. The app uses default settings.

📊 Dataset
The model is trained on customer data with the following characteristics:

Records: 5000 customer records

Features: 9 input features

Target: Churn (Yes/No)

Data Source: Synthetic customer churn dataset

🏆 Key Achievements
✅ 91.5% accuracy in predicting customer churn

✅ Production-ready with professional UI

✅ Real-time predictions with visual analytics

✅ Actionable business insights

✅ Successful deployment on Streamlit Cloud

⭐ If you like this project, please give it a star on GitHub!
