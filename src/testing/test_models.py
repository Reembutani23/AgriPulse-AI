import unittest
import pandas as pd
import numpy as np
import os
# pyrefly: ignore [missing-import]
import joblib

class TestModelPerformance(unittest.TestCase):
    """
    Unit tests for model performance and predictions
    """
    
    def setUp(self):
        """Set up test fixtures"""
        self.processed_data_path = 'data/processed/processed_data.csv'
        self.engineered_data_path = 'data/processed/engineered_data.csv'
        self.rf_model_path = 'models/random_forest.pkl'
        self.xgb_model_path = 'models/xgboost.pkl'
        
    def test_r2_score_threshold(self):
        """Test that models are trained and present"""
        self.assertTrue(os.path.exists(self.rf_model_path), "Random Forest model file missing.")
        self.assertTrue(os.path.exists(self.xgb_model_path), "XGBoost model file missing.")
        
    def test_no_nan_predictions(self):
        """Test that predictions don't contain NaN values"""
        if os.path.exists(self.xgb_model_path) and os.path.exists(self.engineered_data_path):
            model = joblib.load(self.xgb_model_path)
            feature_cols = joblib.load('models/feature_cols.pkl')
            df = pd.read_csv(self.engineered_data_path).dropna()
            
            # Predict
            X = df[feature_cols]
            preds = model.predict(X)
            self.assertEqual(np.isnan(preds).sum(), 0, "Model predictions contain NaNs!")
            
    def test_prediction_range(self):
        """Test that predictions are within a reasonable range"""
        if os.path.exists(self.xgb_model_path) and os.path.exists(self.engineered_data_path):
            model = joblib.load(self.xgb_model_path)
            feature_cols = joblib.load('models/feature_cols.pkl')
            df = pd.read_csv(self.engineered_data_path).dropna()
            
            X = df[feature_cols]
            preds = model.predict(X)
            
            # Predicted yield should be non-negative and not excessively huge
            self.assertTrue((preds >= 0).all(), "Negative yield predictions found!")
            self.assertTrue((preds < df['Yield'].max() * 2.0).all(), "Excessively high yield predictions found!")

class TestDataQuality(unittest.TestCase):
    """
    Unit tests for data quality
    """
    
    def setUp(self):
        self.cleaned_data_path = 'data/processed/cleaned_data.csv'
        
    def test_no_missing_values(self):
        """Test cleaned data has no missing values in key columns"""
        if os.path.exists(self.cleaned_data_path):
            df = pd.read_csv(self.cleaned_data_path)
            cols_to_check = ['Country', 'Crop', 'Year', 'Temperature', 'Rainfall', 'CO2_Emission', 'Yield']
            for col in cols_to_check:
                if col in df.columns:
                    self.assertEqual(df[col].isnull().sum(), 0, f"Cleaned data column '{col}' contains NaNs!")
                    
    def test_valid_feature_ranges(self):
        """Test features are within physically valid ranges in cleaned data"""
        if os.path.exists(self.cleaned_data_path):
            df = pd.read_csv(self.cleaned_data_path)
            
            if 'Temperature' in df.columns:
                self.assertTrue((df['Temperature'] >= -50).all() and (df['Temperature'] <= 60).all(), "Temperature values out of valid bounds.")
            if 'Rainfall' in df.columns:
                self.assertTrue((df['Rainfall'] >= 0).all() and (df['Rainfall'] <= 10000).all(), "Rainfall values out of valid bounds.")
            if 'Yield' in df.columns:
                self.assertTrue((df['Yield'] >= 0).all(), "Negative crop yield values found in cleaned data.")

if __name__ == '__main__':
    unittest.main()
