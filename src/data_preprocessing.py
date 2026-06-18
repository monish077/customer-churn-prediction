import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def load_data(self, filepath):
        """Load and clean the data"""
        try:
            df = pd.read_csv(filepath)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Check if data is empty
            if df.empty or df.isnull().all().all():
                print("⚠️ Data file is empty! Generating sample data...")
                df = self._generate_sample_data()
            else:
                print(f"✅ Loaded {len(df)} records from your data file")
                
                # Check for required columns
                required_cols = ['CustomerID', 'Age', 'Gender', 'Tenure', 'Usage Frequency', 
                                'Support Calls', 'Payment Delay', 'Subscription Type', 
                                'Contract Length', 'Total Spend']
                
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"⚠️ Missing columns: {missing_cols}")
                    print("📊 Your data has these columns:", df.columns.tolist())
                    print("💡 Using sample data instead...")
                    df = self._generate_sample_data()
                else:
                    print("✅ All required columns found!")
                    
                    # Check if Churn column exists
                    if 'Churn' not in df.columns:
                        print("⚠️ No 'Churn' column found. Adding placeholder for pipeline...")
                        df['Churn'] = 0
                        
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
            print("💡 Generating sample data...")
            df = self._generate_sample_data()
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("💡 Generating sample data...")
            df = self._generate_sample_data()
        
        return df
    
    def _generate_sample_data(self):
        """Generate realistic churn data for demonstration"""
        np.random.seed(42)
        n = 5000
        
        data = {
            'CustomerID': [f'CUST-{i:05d}' for i in range(n)],
            'Age': np.random.randint(18, 70, n),
            'Gender': np.random.choice(['Male', 'Female'], n),
            'Tenure': np.random.randint(1, 72, n),
            'Usage Frequency': np.random.randint(1, 50, n),
            'Support Calls': np.random.randint(0, 10, n),
            'Payment Delay': np.random.randint(0, 30, n),
            'Subscription Type': np.random.choice(['Basic', 'Standard', 'Premium'], n),
            'Contract Length': np.random.choice(['Monthly', 'Quarterly', 'Yearly'], n),
            'Total Spend': np.random.randint(100, 5000, n)
        }
        df = pd.DataFrame(data)
        
        # Add churn column based on patterns
        df['Churn'] = 0
        df.loc[(df['Contract Length'] == 'Monthly') & 
               (df['Support Calls'] > 5), 'Churn'] = 1
        df.loc[(df['Contract Length'] == 'Yearly') & 
               (df['Support Calls'] < 3), 'Churn'] = 0
        df.loc[np.random.choice(df.index, size=int(n*0.1)), 'Churn'] = 1
        
        print("📊 Sample data generated with 5000 records")
        return df
    
    def preprocess(self, df):
        """Clean and preprocess the data"""
        df = df.copy()
        
        # Handle missing values
        df = df.dropna()
        
        # Encode categorical variables BEFORE feature engineering
        categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Ensure all numeric columns are actually numeric
        numeric_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 
                       'Payment Delay', 'Total Spend']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Separate features and target
        if 'Churn' in df.columns:
            y = df['Churn'].astype(int)
            # Drop CustomerID if it exists, otherwise just drop non-feature columns
            X = df.drop(['Churn'], axis=1)
            if 'CustomerID' in X.columns:
                X = X.drop(['CustomerID'], axis=1)
        else:
            # For prediction without target
            y = None
            X = df.copy()
            if 'CustomerID' in X.columns:
                X = X.drop(['CustomerID'], axis=1)
        
        # Scale numerical features (only numeric columns)
        numeric_cols_to_scale = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 
                                'Payment Delay', 'Total Spend']
        
        # Also scale any numeric columns from feature engineering
        for col in X.columns:
            if col in numeric_cols_to_scale or pd.api.types.is_numeric_dtype(X[col]):
                if col not in ['Gender', 'Subscription Type', 'Contract Length']:
                    if col not in numeric_cols_to_scale:
                        numeric_cols_to_scale.append(col)
        
        # Remove duplicates
        numeric_cols_to_scale = list(set(numeric_cols_to_scale))
        numeric_cols_to_scale = [col for col in numeric_cols_to_scale if col in X.columns]
        
        # Scale the columns
        if numeric_cols_to_scale:
            X[numeric_cols_to_scale] = self.scaler.fit_transform(X[numeric_cols_to_scale])
        
        # Ensure all columns are numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Drop any remaining non-numeric columns
        for col in X.columns:
            if X[col].dtype == 'object':
                X = X.drop(col, axis=1)
        
        # Fill NaN values with 0
        X = X.fillna(0)
        
        return X, y
    
    def get_feature_names(self):
        return ['Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 
                'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend']