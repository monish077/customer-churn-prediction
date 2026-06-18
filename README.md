# 📊 Customer Churn Dataset

## Dataset Overview

This dataset contains **440,882 customer records** with their respective features and churn labels. It serves as the primary resource for training machine learning models to predict customer churn.

---

## 📋 Dataset Description

### Purpose
The dataset is designed to help businesses develop accurate churn prediction models to identify customers who are most likely to churn and take proactive actions to retain them.

### Source
- **Total Records**: 440,882
- **Features**: 10 input features + 1 target variable
- **Target**: Churn label (1 = Churned, 0 = Not Churned)

---

## 🗂️ Feature Descriptions

| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| **Age** | Numerical | Customer's age | 18-70 years |
| **Gender** | Categorical | Customer's gender | Male, Female |
| **Tenure** | Numerical | Time with the company | 1-72 months |
| **Usage Frequency** | Numerical | Monthly usage frequency | 1-50 times |
| **Support Calls** | Numerical | Number of support calls made | 0-10 calls |
| **Payment Delay** | Numerical | Days payment is delayed | 0-30 days |
| **Subscription Type** | Categorical | Type of subscription | Basic, Standard, Premium |
| **Contract Length** | Categorical | Length of contract | Monthly, Quarterly, Yearly |
| **Total Spend** | Numerical | Total spending amount | $100-$5,000 |
| **Last Interaction** | Numerical | Days since last interaction | Varies |
| **Churn** | Target | Whether customer churned | 0 = No, 1 = Yes |

---

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Records** | 440,882 |
| **Total Features** | 11 (10 input + 1 target) |
| **Target Distribution** | [Add your distribution] |
| **Churn Rate** | [Add your churn rate] |

---

## 🎯 Data Usage

### Training
This dataset is used to train machine learning models for churn prediction. The models learn patterns from customer behavior and demographics to predict future churn.

### Feature Engineering
The raw features can be transformed to create more meaningful features:

| Engineered Feature | Description |
|--------------------|-------------|
| **Tenure Group** | Categorized tenure (New, Regular, Loyal, VIP) |
| **Avg Monthly Spend** | Total spend / Tenure |
| **Support Intensity** | Support calls / Tenure |
| **Payment Reliability** | 1 / (Payment Delay + 1) |
| **Risk Score** | Composite risk metric |

---

## 📁 File Structure
data/
├── raw/
│ ├── customer_churn_dataset-training-master-selected-columns.csv # Training data
│ └── [other dataset files]
├── processed/ # Processed data (if any)
└── README.md # This file

text

---

## 🔍 Sample Data

| Age | Gender | Tenure | Usage Frequency | Support Calls | Payment Delay | Subscription Type | Contract Length | Total Spend | Last Interaction | Churn |
|-----|--------|--------|-----------------|---------------|---------------|-------------------|-----------------|-------------|------------------|-------|
| 35 | Male | 12 | 20 | 3 | 5 | Basic | Monthly | 1500 | 7 | 0 |
| 42 | Female | 24 | 35 | 1 | 2 | Premium | Yearly | 3800 | 15 | 0 |
| 28 | Male | 3 | 10 | 8 | 15 | Basic | Monthly | 400 | 2 | 1 |

---

## 📈 Business Applications

This dataset enables businesses to:

- **Predict Churn**: Identify customers at risk of leaving
- **Take Action**: Proactively retain valuable customers
- **Reduce Costs**: Lower customer acquisition costs
- **Increase Revenue**: Target retention campaigns effectively
- **Improve Products**: Understand why customers leave

---

## 📝 License

This dataset is provided for educational and research purposes.

---

## 🤝 Acknowledgments

- Dataset generated for Customer Churn Prediction project
- Synthetic dataset for machine learning training

---

**📊 For more information, visit the [main repository](https://github.com/monish077/customer-churn-prediction).**
To Add This to Your Repository:
bash
cd "D:\Data science course materials\Customer Chunk Prediction"

# Create data directory if it doesn't exist
mkdir data 2>$null

# Create data/README.md with the content above
# (You can copy the content and save it)

# Add and commit
git add data/README.md
git commit -m "📊 Add dataset documentation"
git push origin main
Alternative: Add to Main README.md
If you prefer, you can add this as a section in your main README.md:

markdown
## 📊 Dataset

The training dataset contains **440,882 customer records** with features including age, gender, tenure, usage frequency, support calls, payment delay, subscription type, contract length, total spend, and last interaction. The churn label indicates whether a customer has churned (1) or not (0).

### Features

| Feature | Description |
|---------|-------------|
| Age | Customer's age (18-70) |
| Gender | Male/Female |
| Tenure | Months with company (1-72) |
| Usage Frequency | Monthly usage count (1-50) |
| Support Calls | Number of support calls (0-10) |
| Payment Delay | Days payment is delayed (0-30) |
| Subscription Type | Basic/Standard/Premium |
| Contract Length | Monthly/Quarterly/Yearly |
| Total Spend | Total spending ($100-$5,000) |
| Last Interaction | Days since last interaction |
| **Churn** | **Target: 0=No, 1=Yes** |

The dataset enables businesses to identify customers most likely to churn and take proactive retention actions.
Quick Update Commands:
bash
cd "D:\Data science course materials\Customer Chunk Prediction"

# Add dataset documentation
git add data/README.md

# Commit
git commit -m "📊 Add dataset documentation and description"

# Pull latest (if needed)
git pull origin main --rebase

# Push
git push origin main
