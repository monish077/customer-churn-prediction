import pandas as pd
import numpy as np
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalPredictor:
    def __init__(self, model_path='models/churn_model.pkl'):
        self.model = joblib.load(model_path)
        self.feature_columns = list(self.model.feature_names_in_)
        logger.info(f"Model expects {len(self.feature_columns)} features: {self.feature_columns}")

    def _build_feature_matrix(self, df):
        """
        Build the exact feature matrix the model expects.
        Returns a DataFrame with columns in the correct order.
        """
        df = df.copy()

        # ---- 1. Encode categoricals ----
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}).fillna(0)
        sub_map = {'Basic': 0, 'Standard': 1, 'Premium': 2}
        df['Subscription Type'] = df['Subscription Type'].map(sub_map).fillna(0)
        contract_map = {'Monthly': 0, 'Quarterly': 1, 'Yearly': 2}
        df['Contract Length'] = df['Contract Length'].map(contract_map).fillna(0)

        # ---- 2. Ensure numeric columns ----
        numeric_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls',
                        'Payment Delay', 'Total Spend']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # ---- 3. Engineered features ----
        df['Tenure_Group'] = pd.cut(df['Tenure'], bins=[-1, 6, 24, 1000], labels=[0, 1, 2]).astype(float).fillna(0)
        df['Avg_Monthly_Spend'] = df['Total Spend'] / df['Tenure'].replace(0, 1)
        df['Support_Intensity'] = df['Support Calls'] / df['Tenure'].replace(0, 1)
        df['Payment_Reliability'] = 1 / (df['Payment Delay'] + 1)
        df['Risk_Score'] = (
            (3 - df['Contract Length']) * 0.3 +
            df['Support Calls'] * 0.2 +
            df['Payment Delay'] * 0.2 +
            (1 / df['Tenure'].replace(0, 1)) * 0.3
        )

        # ---- 4. Ensure ALL expected columns exist ----
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
                logger.warning(f"Added missing column: {col}")

        # ---- 5. Select and reorder ----
        X = df[self.feature_columns].copy()
        X = X.fillna(0)

        logger.info(f"Final feature matrix shape: {X.shape}")
        logger.info(f"Columns: {list(X.columns)}")

        return X

    def predict(self, input_data):
        # Convert to DataFrame
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        elif not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame(input_data)

        logger.info(f"Raw input columns: {list(input_data.columns)}")

        # Build the feature matrix
        X = self._build_feature_matrix(input_data)

        # Final sanity check – ensure no missing columns
        missing = set(self.feature_columns) - set(X.columns)
        if missing:
            logger.error(f"Still missing columns: {missing}")
            for col in missing:
                X[col] = 0
            X = X[self.feature_columns]  # reorder

        # Predict
        pred = self.model.predict(X)
        prob = self.model.predict_proba(X)[0][1]
        return int(pred[0]), float(prob)