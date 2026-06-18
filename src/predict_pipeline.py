import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

class ChurnPredictor:
    def __init__(self, model_path='models/churn_model.pkl'):
        """Initialize the predictor with the trained model"""
        self.model = joblib.load(model_path)
        
        # These are the EXACT columns the model expects
        self.feature_columns = [
            'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 
            'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend',
            'Tenure_Group', 'Avg_Monthly_Spend', 'Support_Intensity', 
            'Payment_Reliability', 'Risk_Score'
        ]
        
        # Store the training column order (critical for Random Forest)
        self.training_columns = self.feature_columns.copy()
        
        # Initialize encoders
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Try to load scaler if it exists
        scaler_path = 'models/scaler.pkl'
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                self.is_fitted = True
            except:
                pass
    
    def create_features(self, df):
        """Create engineered features - MUST match training exactly"""
        df = df.copy()
        
        # 1. Tenure groups (bins must match training)
        df['Tenure_Group'] = pd.cut(
            df['Tenure'], 
            bins=[0, 6, 24, 100], 
            labels=[0, 1, 2]
        ).astype(float)
        
        # 2. Average monthly spend
        df['Avg_Monthly_Spend'] = df['Total Spend'] / df['Tenure'].replace(0, 1)
        
        # 3. Support intensity
        df['Support_Intensity'] = df['Support Calls'] / df['Tenure'].replace(0, 1)
        
        # 4. Payment reliability
        df['Payment_Reliability'] = 1 / (df['Payment Delay'] + 1)
        
        # 5. Risk score
        contract_map = {'Monthly': 3, 'Quarterly': 2, 'Yearly': 1}
        df['Risk_Score'] = (
            df['Contract Length'].map(contract_map).fillna(2) * 0.3 +
            df['Support Calls'] * 0.2 +
            df['Payment Delay'] * 0.2 +
            (1 / df['Tenure'].replace(0, 1)) * 0.3
        )
        
        return df
    
    def preprocess(self, df):
        """Preprocess data exactly like training"""
        df = df.copy()
        
        # Handle categorical columns
        categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
        for col in categorical_cols:
            if col in df.columns:
                # Use same encoding as training
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Ensure all numeric columns are proper numbers
        numeric_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 
                       'Payment Delay', 'Total Spend']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Scale numeric features (if scaler is fitted)
        numeric_to_scale = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 
                           'Payment Delay', 'Total Spend', 'Avg_Monthly_Spend', 
                           'Support_Intensity', 'Payment_Reliability', 'Risk_Score']
        
        for col in numeric_to_scale:
            if col in df.columns:
                if self.is_fitted and col in self.scaler.mean_:
                    # Use fitted scaler
                    mean = self.scaler.mean_[self.scaler.feature_names_in_.tolist().index(col)]
                    std = self.scaler.scale_[self.scaler.feature_names_in_.tolist().index(col)]
                    df[col] = (df[col] - mean) / (std + 1e-8)
                else:
                    # Manual scaling (fallback)
                    df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        
        # Ensure ALL feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select only the features we need, in the right order
        X = df[self.feature_columns].copy()
        
        # Fill any NaN with 0
        X = X.fillna(0)
        
        return X
    
    def predict(self, input_data):
        """Make a prediction"""
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Step 1: Create engineered features
        df_with_features = self.create_features(input_data)
        
        # Step 2: Preprocess
        X = self.preprocess(df_with_features)
        
        # Step 3: Predict
        prediction = self.model.predict(X)
        probability = self.model.predict_proba(X)[0][1]
        
        return prediction[0], probability
    
    def predict_batch(self, input_data):
        """Make predictions for multiple customers"""
        if isinstance(input_data, dict):
            input_data = pd.DataFrame(input_data)
        
        df_with_features = self.create_features(input_data)
        X = self.preprocess(df_with_features)
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities