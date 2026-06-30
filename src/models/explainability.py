import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import os
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplainabilityAnalyzer:
    """
    Model explainability using SHAP
    """
    
    def __init__(self, model, X_test, feature_names):
        self.model = model
        self.X_test = X_test
        self.feature_names = feature_names
    
    def shap_analysis(self, num_samples: int = 100):
        """
        SHAP analysis for feature attribution
        
        Master Prompt Context:
        - SHAP (SHapley Additive exPlanations)
        - Shows how each feature contributes to predictions
        - Provides both global and local explanations
        """
        logger.info("Performing SHAP analysis...")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        
        # Compute shap values
        X_sample = self.X_test.iloc[:num_samples]
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot (beeswarm summary)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample,
                         feature_names=self.feature_names, show=False)
        plt.tight_layout()
        os.makedirs('output', exist_ok=True)
        save_path_summary = 'output/shap_summary.png'
        plt.savefig(save_path_summary, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP summary beeswarm plot to {save_path_summary}")
        
        # Bar plot (feature importance)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample,
                         plot_type="bar", feature_names=self.feature_names, show=False)
        plt.tight_layout()
        save_path_bar = 'output/shap_importance.png'
        plt.savefig(save_path_bar, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP importance bar plot to {save_path_bar}")
        
        return shap_values, explainer
    
    def feature_importance_ranking(self, shap_values) -> pd.DataFrame:
        """
        Rank features by SHAP importance
        """
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP_Importance': feature_importance
        }).sort_values('SHAP_Importance', ascending=False)
        
        logger.info("\nTop Features by SHAP Importance:")
        logger.info(importance_df.head(10).to_string(index=False))
        
        return importance_df

if __name__ == "__main__":
    import sys
    data_path = 'data/processed/engineered_data.csv'
    model_path = 'models/xgboost.pkl'
    cols_path = 'models/feature_cols.pkl'
    
    if not (os.path.exists(data_path) and os.path.exists(model_path) and os.path.exists(cols_path)):
        logger.error("Required data or model files not found. Run training script first.")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)
    
    # Prepare data (filter out target variable and non-numeric columns)
    X_test = df[feature_cols].dropna()
    
    # SHAP analysis
    analyzer = ExplainabilityAnalyzer(model, X_test, feature_cols)
    shap_values, explainer = analyzer.shap_analysis()
    importance = analyzer.feature_importance_ranking(shap_values)
    importance.to_csv('output/shap_importance.csv', index=False)
    print("Explainability analysis complete!")
