from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ChurnModelTrainer
from sklearn.model_selection import train_test_split
import pandas as pd

def main():

    print("🚀 Starting Customer Churn Prediction Pipeline")
    print("="*50)

    # ---------------- LOAD DATA ----------------
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(
        'data/raw/customer_churn_dataset-training-master-selected-columns.csv'
    )

    print("🔧 Feature Engineering...")
    engineer = FeatureEngineer()
    df = engineer.create_features(df)

    # ---------------- SPLIT FEATURES ----------------
    X, y = preprocessor.preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # ---------------- TRAIN ----------------
    trainer = ChurnModelTrainer()

    results = trainer.train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ---------------- SAVE MODEL ----------------
    trainer.save_models()

    print("\n📊 Model Performance Summary")
    print("="*50)

    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        for k, v in metrics.items():
            print(f"{k}: {v:.3f}")

    print("\n✅ Pipeline completed successfully!")
    print("👉 Run Streamlit: streamlit run app/app.py")


if __name__ == "__main__":
    main()