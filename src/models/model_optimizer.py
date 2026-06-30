import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# pyrefly: ignore [missing-import]
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """
    Hyperparameter tuning and model optimization
    """
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def optimize_random_forest(self, cv: int = 5) -> Tuple[dict, RandomForestRegressor]:
        """
        Grid search for Random Forest hyperparameters
        
        Master Prompt Context:
        - n_estimators: More trees generally better (100-300)
        - max_depth: Deeper = better fit but risk of overfitting (10-25)
        - min_samples_split: Higher = less overfitting (2-10)
        - min_samples_leaf: Higher = less overfitting (1-5)
        """
        logger.info("Optimizing Random Forest hyperparameters...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, 
            cv=cv, 
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        logger.info(f"Best RF parameters: {grid_search.best_params_}")
        logger.info(f"Best CV R²: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(self.X_test)
        test_r2 = r2_score(self.y_test, y_pred)
        logger.info(f"Test R²: {test_r2:.4f}")
        
        results = {
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'test_r2': test_r2
        }
        
        return results, best_rf
    
    def optimize_xgboost(self, cv: int = 5, n_iter: int = 10) -> Tuple[dict, xgb.XGBRegressor]:
        """
        Randomized search for XGBoost hyperparameters
        
        Master Prompt Context:
        - n_estimators: 50-300
        - max_depth: 3-15
        - learning_rate: 0.01-0.3 (lower = slower but better)
        - subsample: 0.5-1.0 (fraction of samples per tree)
        - colsample_bytree: 0.5-1.0 (fraction of features per tree)
        """
        logger.info("Optimizing XGBoost hyperparameters...")
        
        param_dist = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        random_search = RandomizedSearchCV(
            xgb_model, param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        logger.info(f"Best XGB parameters: {random_search.best_params_}")
        logger.info(f"Best CV R²: {random_search.best_score_:.4f}")
        
        # Evaluate on test set
        best_xgb = random_search.best_estimator_
        y_pred = best_xgb.predict(self.X_test)
        test_r2 = r2_score(self.y_test, y_pred)
        logger.info(f"Test R²: {test_r2:.4f}")
        
        results = {
            'best_params': random_search.best_params_,
            'cv_score': random_search.best_score_,
            'test_r2': test_r2
        }
        
        return results, best_xgb
    
    def plot_learning_curves(self, model, model_name: str):
        """
        Plot learning curves to diagnose bias/variance
        
        Master Prompt Context:
        - If train and validation curves converge at low score: High bias
        - If large gap between curves: High variance (overfitting)
        """
        logger.info(f"Generating learning curves for {model_name}...")
        
        # pyrefly: ignore [bad-unpacking]
        train_sizes, train_scores, val_scores = learning_curve(
            model, self.X_train, self.y_train,
            cv=5,
            scoring='r2',
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training Score', color='blue', marker='o')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
        plt.plot(train_sizes, val_mean, label='Validation Score', color='red', marker='o')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('R² Score')
        plt.title(f'{model_name} Learning Curves')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs('output', exist_ok=True)
        save_path = f'output/learning_curves_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Learning curves saved to {save_path}")
    
    def plot_residuals(self, y_pred, y_test, model_name: str):
        """
        Plot residual analysis
        """
        residuals = y_test.values - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'{model_name}: Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1].hist(residuals, bins=50, edgecolor='black')
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{model_name}: Residuals Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs('output', exist_ok=True)
        save_path = f'output/residuals_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Residual plots saved to {save_path}")

if __name__ == "__main__":
    # Load data
    from src.models.model_trainer import ModelTrainer
    df = pd.read_csv('data/processed/engineered_data.csv')
    
    trainer = ModelTrainer(df)
    optimizer = ModelOptimizer(
        trainer.X_train, trainer.X_test,
        trainer.y_train, trainer.y_test
    )
    
    # Optimize models
    rf_results, best_rf = optimizer.optimize_random_forest()
    xgb_results, best_xgb = optimizer.optimize_xgboost()
    
    # Visualize
    optimizer.plot_learning_curves(best_rf, "Random Forest")
    optimizer.plot_learning_curves(best_xgb, "XGBoost")
    
    y_pred_rf = best_rf.predict(trainer.X_test)
    optimizer.plot_residuals(y_pred_rf, trainer.y_test, "Random Forest")
    
    y_pred_xgb = best_xgb.predict(trainer.X_test)
    optimizer.plot_residuals(y_pred_xgb, trainer.y_test, "XGBoost")
    print("Optimization and validation plotting complete!")
