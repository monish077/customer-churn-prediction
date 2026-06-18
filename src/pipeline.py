from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class FeaturePipeline:

    def __init__(self):

        self.num_features = [
            "Age", "Tenure", "Usage Frequency",
            "Support Calls", "Payment Delay", "Total Spend"
        ]

        self.cat_features = [
            "Gender", "Subscription Type", "Contract Length"
        ]

        self.model = self.build_pipeline()

    def build_pipeline(self):

        numeric_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.num_features),
                ("cat", categorical_transformer, self.cat_features)
            ]
        )

        model = RandomForestClassifier(n_estimators=200, random_state=42)

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        return pipeline

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)