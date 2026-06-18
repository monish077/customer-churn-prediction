import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.new_features = []
    
    def create_features(self, df):
        """Create new features from existing ones"""
        df = df.copy()
        
        # Tenure groups (keep as numeric categories)
        df['Tenure_Group'] = pd.cut(df['Tenure'], 
                                    bins=[0, 6, 24, 72], 
                                    labels=[0, 1, 2])  # Numeric instead of strings
        
        # Average monthly spend
        df['Avg_Monthly_Spend'] = df['Total Spend'] / df['Tenure'].clip(lower=1)
        
        # Support intensity
        df['Support_Intensity'] = df['Support Calls'] / df['Tenure'].clip(lower=1)
        
        # Payment reliability score
        df['Payment_Reliability'] = 1 / (df['Payment Delay'] + 1)
        
        # Customer risk score
        df['Risk_Score'] = (
            (df['Contract Length'].map({'Monthly': 3, 'Quarterly': 2, 'Yearly': 1}) * 0.3) +
            (df['Support Calls'] * 0.2) +
            (df['Payment Delay'] * 0.2) +
            ((1 / df['Tenure'].clip(lower=1)) * 0.3)
        )
        
        return df