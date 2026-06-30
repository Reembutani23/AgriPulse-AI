import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import logging
import os
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Comprehensive model training with multiple algorithms
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'Yield', test_size: float = 0.2):
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.models = {}
        self.results = {}
        
        self._prepare_data()
    
    def _prepare_data(self):
        """
        Prepare data for modeling
        """
        logger.info("Preparing data for modeling...")
        
        # Select features (exclude non-numeric and target)
        exclude_cols = [self.target_col, 'Country', 'Crop', 'Year']
        self.feature_cols = [col for col in self.df.columns 
                            if col not in exclude_cols 
                            and self.df[col].dtype in ['float64', 'int64', 'int32', 'float32']]
        
        # Remove NaN values introduced by lagged/rolling features
        self.df_clean = self.df.dropna()
        
        self.X = self.df_clean[self.feature_cols]
        self.y = self.df_clean[self.target_col]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=42
        )
        
        logger.info(f"Training set: {self.X_train.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
        logger.info(f"Features: {len(self.feature_cols)}")
    
    def train_random_forest(self, n_estimators: int = 200, max_depth: int = 20,
                          cv: int = 5) -> Dict:
        """
        Train Random Forest model
        
        Master Prompt Context:
        - Handles non-linear relationships well
        - Good for feature importance analysis
        - Robust to outliers
        """
        logger.info("Training Random Forest model...")
        
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        # Train on training set
        rf_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_train_pred = rf_model.predict(self.X_train)
        y_test_pred = rf_model.predict(self.X_test)
        
        # Evaluate
        results = self._evaluate_model(y_test_pred, 'Random Forest')
        results['model'] = rf_model
        results['train_r2'] = r2_score(self.y_train, y_train_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, self.X_train, self.y_train,
                                    cv=cv, scoring='r2', n_jobs=-1)
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
        
        self.models['Random Forest'] = rf_model
        self.results['Random Forest'] = results
        
        logger.info(f"RF Train R²: {results['train_r2']:.4f}")
        logger.info(f"RF Test R²: {results['r2']:.4f}")
        logger.info(f"RF CV R² (mean ± std): {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        
        return results
    
    def train_xgboost(self, n_estimators: int = 150, max_depth: int = 10,
                     learning_rate: float = 0.1, cv: int = 5) -> Dict:
        """
        Train XGBoost model
        
        Master Prompt Context:
        - Gradient boosting often outperforms Random Forest
        - Better for complex non-linear relationships
        - More prone to overfitting (requires careful tuning)
        """
        logger.info("Training XGBoost model...")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # Train
        xgb_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = xgb_model.predict(self.X_train)
        y_test_pred = xgb_model.predict(self.X_test)
        
        # Evaluate
        results = self._evaluate_model(y_test_pred, 'XGBoost')
        results['model'] = xgb_model
        results['train_r2'] = r2_score(self.y_train, y_train_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(xgb_model, self.X_train, self.y_train,
                                    cv=cv, scoring='r2', n_jobs=-1)
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
        
        self.models['XGBoost'] = xgb_model
        self.results['XGBoost'] = results
        
        logger.info(f"XGB Train R²: {results['train_r2']:.4f}")
        logger.info(f"XGB Test R²: {results['r2']:.4f}")
        logger.info(f"XGB CV R² (mean ± std): {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        
        return results
    
    def train_gradient_boosting(self, n_estimators: int = 200, max_depth: int = 5,
                               learning_rate: float = 0.1, cv: int = 5) -> Dict:
        """
        Train Sklearn Gradient Boosting model
        """
        logger.info("Training Gradient Boosting model...")
        
        gb_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42
        )
        
        # Train
        gb_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = gb_model.predict(self.X_train)
        y_test_pred = gb_model.predict(self.X_test)
        
        # Evaluate
        results = self._evaluate_model(y_test_pred, 'Gradient Boosting')
        results['model'] = gb_model
        results['train_r2'] = r2_score(self.y_train, y_train_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(gb_model, self.X_train, self.y_train,
                                    cv=cv, scoring='r2', n_jobs=-1)
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
        
        self.models['Gradient Boosting'] = gb_model
        self.results['Gradient Boosting'] = results
        
        logger.info(f"GB Train R²: {results['train_r2']:.4f}")
        logger.info(f"GB Test R²: {results['r2']:.4f}")
        logger.info(f"GB CV R² (mean ± std): {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        
        return results
    
    def _evaluate_model(self, y_pred: np.ndarray, model_name: str) -> Dict:
        """
        Evaluate model performance
        """
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((self.y_test - y_pred) / (self.y_test + 1e-5))) * 100
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  MSE: {mse:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R² Score: {r2:.4f}")
        logger.info(f"  MAPE: {mape:.4f}%")
        
        return results
    
    def ensemble_predictions(self) -> Tuple[np.ndarray, Dict]:
        """
        Create ensemble predictions using multiple models
        """
        logger.info("Creating ensemble predictions...")
        
        ensemble_pred = np.zeros_like(self.y_test, dtype=float)
        weights = {}
        
        # Weight models by their R² scores (clipping at 0 to avoid giving negative weight)
        total_r2 = sum([max(0, results['r2']) for results in self.results.values()])
        if total_r2 == 0:
            total_r2 = 1.0
            
        for model_name, model in self.models.items():
            weight = max(0, self.results[model_name]['r2']) / total_r2
            pred = model.predict(self.X_test)
            ensemble_pred += weight * pred
            weights[model_name] = weight
            logger.info(f"{model_name} ensemble weight: {weight:.4f}")
        
        # Evaluate ensemble
        ensemble_results = self._evaluate_model(ensemble_pred, "Ensemble")
        ensemble_results['weights'] = weights
        
        return ensemble_pred, ensemble_results
    
    def save_models(self, path: str = 'models/'):
        """
        Save trained models
        """
        os.makedirs(path, exist_ok=True)
        for model_name, model in self.models.items():
            model_path = os.path.join(path, f"{model_name.lower().replace(' ', '_')}.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
            
        # Save feature list for explaining model later
        feature_path = os.path.join(path, "feature_cols.pkl")
        joblib.dump(self.feature_cols, feature_path)
        logger.info(f"Saved feature columns list to {feature_path}")
    
    def get_training_report(self) -> pd.DataFrame:
        """
        Generate training report
        """
        report_data = []
        
        for model_name, results in self.results.items():
            report_data.append({
                'Model': model_name,
                'Train R²': results.get('train_r2', 'N/A'),
                'Test R²': results['r2'],
                'CV R² (Mean)': results.get('cv_mean', 'N/A'),
                'CV R² (Std)': results.get('cv_std', 'N/A'),
                'RMSE': results['rmse'],
                'MAE': results['mae'],
                'MAPE': results['mape']
            })
        
        report_df = pd.DataFrame(report_data)
        
        logger.info("\n" + "="*80)
        logger.info("MODEL TRAINING REPORT")
        logger.info("="*80)
        logger.info(report_df.to_string(index=False))
        logger.info("="*80)
        
        return report_df

if __name__ == "__main__":
    import sys
    data_path = 'data/processed/engineered_data.csv'
    if not os.path.exists(data_path):
        logger.error(f"Engineered data not found at {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    
    # Train models
    trainer = ModelTrainer(df)
    
    # Train individual models
    trainer.train_random_forest(n_estimators=200, max_depth=20)
    trainer.train_xgboost(n_estimators=150, max_depth=10)
    trainer.train_gradient_boosting(n_estimators=200, max_depth=5)
    
    # Ensemble
    ensemble_pred, ensemble_results = trainer.ensemble_predictions()
    
    # Save models
    trainer.save_models()
    
    # Report
    report = trainer.get_training_report()
    os.makedirs('output', exist_ok=True)
    report.to_csv('output/training_report.csv', index=False)
    
    print("Model training complete!")
