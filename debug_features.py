import joblib
import pandas as pd
import numpy as np

def engineer_and_encode_features(df):
    """Apply feature engineering and encode all categorical variables"""
    df = df.copy()
    
    # Encode categorical variables
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    sub_map = {'Basic': 0, 'Standard': 1, 'Premium': 2}
    df['Subscription Type'] = df['Subscription Type'].map(sub_map)
    contract_map = {'Monthly': 0, 'Quarterly': 1, 'Yearly': 2}
    df['Contract Length'] = df['Contract Length'].map(contract_map)
    
    # Create engineered features
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
    
    # Reorder columns
    expected_order = [
        'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 
        'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend',
        'Tenure_Group', 'Avg_Monthly_Spend', 'Support_Intensity', 
        'Payment_Reliability', 'Risk_Score'
    ]
    
    df = df[expected_order]
    df = df.astype(float)
    
    return df

def test_prediction():
    """Test the prediction with properly encoded data"""
    
    # Load model
    try:
        model = joblib.load("models/churn_model.pkl")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return
    
    # Create test data
    raw_data = pd.DataFrame([{
        'Age': 35,
        'Gender': 'Male',
        'Tenure': 12,
        'Usage Frequency': 20,
        'Support Calls': 3,
        'Payment Delay': 5,
        'Subscription Type': 'Basic',
        'Contract Length': 'Monthly',
        'Total Spend': 1500
    }])
    
    print("📊 Raw input data:")
    print(raw_data)
    
    # Process data
    processed_data = engineer_and_encode_features(raw_data)
    
    print("\n🔧 Processed data (all numeric):")
    print(processed_data)
    print(f"\nData types: {processed_data.dtypes}")
    
    # Make prediction
    try:
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)
        
        print("\n✅ Prediction successful!")
        print(f"  Prediction: {prediction[0]} (0=No Churn, 1=Churn)")
        print(f"  Probability of Churn: {probability[0][1]*100:.2f}%")
        print(f"  Probability of No Churn: {probability[0][0]*100:.2f}%")
        
        # Test with different scenarios
        print("\n" + "="*50)
        print("Testing multiple scenarios:")
        
        scenarios = [
            {"Age": 25, "Gender": "Female", "Tenure": 3, "Usage": 10, "Support": 8, 
             "Delay": 15, "Sub": "Basic", "Contract": "Monthly", "Spend": 500},
            {"Age": 45, "Gender": "Male", "Tenure": 36, "Usage": 35, "Support": 1, 
             "Delay": 2, "Sub": "Premium", "Contract": "Yearly", "Spend": 4000},
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            test_df = pd.DataFrame([{
                'Age': scenario['Age'],
                'Gender': scenario['Gender'],
                'Tenure': scenario['Tenure'],
                'Usage Frequency': scenario['Usage'],
                'Support Calls': scenario['Support'],
                'Payment Delay': scenario['Delay'],
                'Subscription Type': scenario['Sub'],
                'Contract Length': scenario['Contract'],
                'Total Spend': scenario['Spend']
            }])
            
            processed = engineer_and_encode_features(test_df)
            pred = model.predict(processed)[0]
            prob = model.predict_proba(processed)[0][1]
            
            status = "⚠️ HIGH RISK" if pred == 1 else "✅ LOW RISK"
            print(f"\nScenario {i}: {status}")
            print(f"  Tenure: {scenario['Tenure']} months, Contract: {scenario['Contract']}")
            print(f"  Churn Probability: {prob*100:.2f}%")
            
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")

if __name__ == "__main__":
    test_prediction()