import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os


class ChurnModelTrainer:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }

        self.best_model = None
        self.best_model_name = None
        self.metrics = {}

    # ---------------- TRAIN ----------------
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):

        results = {}

        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        for name, model in self.models.items():
            print(f"Training {name}...")

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                results[name] = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    "AUC-ROC": roc_auc_score(y_test, y_pred)
                }

                print(f"✅ {name} done")

            except Exception as e:
                print(f"❌ {name} failed: {e}")

        # ---------------- BEST MODEL ----------------
        if results:
            best_name = max(results, key=lambda x: results[x]["F1 Score"])
            self.best_model = self.models[best_name]
            self.best_model_name = best_name
            self.metrics = results[best_name]

            print("\n🏆 BEST MODEL:", best_name)
            print("📊 METRICS:", self.metrics)

        else:
            print("⚠️ No model trained. Using RandomForest fallback.")
            self.best_model = RandomForestClassifier(random_state=42)
            self.best_model.fit(X_train, y_train)

        return results

    # ---------------- SAVE (IMPORTANT FIX) ----------------
    def save_models(self):
        os.makedirs("models", exist_ok=True)

        # 🔥 SAVE MODEL WITH CONSISTENT NAME
        model_path = "models/churn_model.pkl"
        joblib.dump(self.best_model, model_path)

        print(f"💾 Model saved → {model_path}")
        print(f"🏆 Best Model Used → {self.best_model_name}")

    # ---------------- LOAD ----------------
    def load_model(self, model_path="models/churn_model.pkl"):
        return joblib.load(model_path)