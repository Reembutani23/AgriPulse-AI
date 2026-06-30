import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# pyrefly: ignore [missing-import]
import joblib
import logging
import os
import json
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FuturePredictor:
    """
    Predict future climate and crop yields.

    Master Prompt Context:
    - Use historical trends to project future scenarios
    - Model climate variables with trend analysis
    - Combine with ML models for yield prediction
    - Provide uncertainty estimates
    """

    CLIMATE_FEATURES = ['Temperature', 'Rainfall', 'CO2_Emission', 'Humidity']

    def __init__(self, df: pd.DataFrame, model_path: str = 'models/'):
        self.df = df
        self.model_path = model_path
        self.trend_models: Dict = {}
        self._ml_models: Dict = {}
        self._feature_cols: Optional[List[str]] = None
        self._scaling_meta: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Trend modelling
    # ------------------------------------------------------------------

    def trend_analysis(self, feature: str, order: int = 2) -> Optional[LinearRegression]:
        """Fit a polynomial trend to a historical feature series."""
        logger.info(f"Analyzing trend for '{feature}' (degree={order})...")
        X = self.df['Year'].values.reshape(-1, 1)
        y = self.df[feature].values
        mask = ~np.isnan(y)
        X_clean, y_clean = X[mask], y[mask]
        if len(X_clean) == 0:
            logger.warning(f"No valid data points for '{feature}' — skipping.")
            return None
        poly = PolynomialFeatures(degree=order)
        X_poly = poly.fit_transform(X_clean)
        model = LinearRegression()
        model.fit(X_poly, y_clean)
        r2 = model.score(X_poly, y_clean)
        self.trend_models[feature] = {'model': model, 'poly': poly, 'r2': r2}
        logger.info(f"  '{feature}' trend R²: {r2:.4f}")
        return model

    def _fit_all_climate_trends(self, order: int = 2) -> None:
        for col in self.CLIMATE_FEATURES:
            if col in self.df.columns:
                self.trend_analysis(col, order=order)

    def predict_future_climate(self, years: List[int]) -> pd.DataFrame:
        """Project climate variables forward for each requested year."""
        if not self.trend_models:
            raise RuntimeError(
                "No trend models fitted. Call trend_analysis() or "
                "climate_scenario_analysis() first."
            )
        logger.info(f"Projecting climate for years: {years}")
        future_data: Dict = {'Year': years}
        X_future = np.array(years).reshape(-1, 1)
        for feature, info in self.trend_models.items():
            X_poly = info['poly'].transform(X_future)
            future_data[feature] = info['model'].predict(X_poly)
        return pd.DataFrame(future_data)

    # ------------------------------------------------------------------
    # Scenario analysis
    # ------------------------------------------------------------------

    def climate_scenario_analysis(
        self, future_years: Optional[List[int]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Build three IPCC-inspired climate scenarios."""
        if future_years is None:
            future_years = [2026, 2030, 2040, 2050]
        self._fit_all_climate_trends()
        if not self.trend_models:
            logger.error("No climate features could be modelled.")
            return {}
        baseline = self.predict_future_climate(future_years)
        warming = baseline.copy()
        if 'Temperature' in warming.columns:
            warming['Temperature'] *= 1.3
        if 'CO2_Emission' in warming.columns:
            warming['CO2_Emission'] *= 1.15
        if 'Rainfall' in warming.columns:
            warming['Rainfall'] *= 0.95
        adaptation = baseline.copy()
        if 'Temperature' in adaptation.columns:
            adaptation['Temperature'] *= 0.7
        if 'CO2_Emission' in adaptation.columns:
            adaptation['CO2_Emission'] *= 0.85
        return {'baseline': baseline, 'warming': warming, 'adaptation': adaptation}

    # ------------------------------------------------------------------
    # ML-based yield prediction
    # ------------------------------------------------------------------

    def _load_ml_models(self) -> bool:
        if self._ml_models:
            return True
        candidates = {
            'XGBoost': 'xgboost.pkl',
            'Random Forest': 'random_forest.pkl',
            'Gradient Boosting': 'gradient_boosting.pkl',
        }
        feature_cols_path = os.path.join(self.model_path, 'feature_cols.pkl')
        scaling_meta_path = os.path.join(self.model_path, 'scaling_metadata.json')
        for name, filename in candidates.items():
            full_path = os.path.join(self.model_path, filename)
            if os.path.exists(full_path):
                self._ml_models[name] = joblib.load(full_path)
                logger.info(f"Loaded model: {name}")
        if os.path.exists(feature_cols_path):
            self._feature_cols = joblib.load(feature_cols_path)
            logger.info(f"Loaded {len(self._feature_cols)} feature columns.")
        if os.path.exists(scaling_meta_path):
            with open(scaling_meta_path, 'r') as f:
                self._scaling_meta = json.load(f)
        if not self._ml_models:
            logger.warning(f"No ML model files found in '{self.model_path}'.")
            return False
        return True

    def _compute_historical_stats(self, country: str) -> Dict:
        stats: Dict = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            stats[col] = float(self.df[col].mean())
        stats['baseline_year'] = int(self.df['Year'].min())
        stats['co2_mean'] = float(self.df['CO2_Emission'].mean()) \
            if 'CO2_Emission' in self.df.columns else 378.0
        if 'Crop' in self.df.columns and 'Yield' in self.df.columns:
            mask = (
                self.df['Country'].str.lower() == country.lower()
                if 'Country' in self.df.columns
                else pd.Series([True] * len(self.df))
            )
            sub = self.df[mask] if mask.any() else self.df
            stats['country_yield_baseline'] = sub.groupby('Crop')['Yield'].mean().to_dict()
        return stats

    def _build_feature_row(self, climate_row, crop, country, country_baseline, hist) -> Optional[pd.Series]:
        if self._feature_cols is None:
            return None
        row: Dict = {}
        hist = hist or {}
        for col in self._feature_cols:
            if col in climate_row.index:
                row[col] = climate_row[col]
            elif col.endswith('_outlier'):
                row[col] = False
            elif col.startswith('Crop_'):
                parts = col.split('_')
                if len(parts) == 3 and parts[1] == crop.lower():
                    base_var = parts[2]
                    row[col] = climate_row.get(base_var, hist.get(col, 0.0))
                else:
                    row[col] = 0.0
            elif col == 'Country_Yield_Baseline':
                row[col] = country_baseline if country_baseline is not None else hist.get(col, 60.0)
            elif col == 'Country_Yield_Trend':
                row[col] = hist.get(col, 0.0)
            elif col == 'Years_Since_Baseline':
                row[col] = climate_row['Year'] - hist.get('baseline_year', 2000)
            elif col == 'Decade':
                row[col] = (climate_row['Year'] // 10) * 10
            elif col == 'Climate_Stress_Index':
                T = climate_row.get('Temperature', hist.get('Temperature', 15.0))
                R = climate_row.get('Rainfall', hist.get('Rainfall', 580.0))
                row[col] = hist.get('Climate_Stress_Index', (T - 15) / 5 - (R - 580) / 200)
            elif col == 'Temp_Rain_Ratio':
                T = climate_row.get('Temperature', hist.get('Temperature', 15.0))
                R = climate_row.get('Rainfall', hist.get('Rainfall', 580.0))
                row[col] = T / (R + 1e-5)
            elif col == 'Growing_Season_Quality':
                row[col] = hist.get(col, 0.8)
            elif col == 'Extreme_Weather_Risk':
                row[col] = hist.get(col, 0.25)
            elif col == 'CO2_Fertilization_Effect':
                co2 = climate_row.get('CO2_Emission', hist.get('CO2_Emission', 378.0))
                row[col] = co2 / 300.0
            elif col == 'CO2_Anomaly':
                co2 = climate_row.get('CO2_Emission', hist.get('CO2_Emission', 378.0))
                row[col] = co2 - hist.get('co2_mean', 378.0)
            elif col == 'Climate_CO2_Interaction':
                T = climate_row.get('Temperature', hist.get('Temperature', 15.0))
                co2 = climate_row.get('CO2_Emission', hist.get('CO2_Emission', 378.0))
                row[col] = T * co2 / 5000.0
            elif col == 'Climate_Change_Index':
                row[col] = hist.get(col, 0.15)
            else:
                row[col] = hist.get(col, 0.0)
        return pd.Series(row)

    def predict_future_yields(
        self,
        scenarios: Dict[str, pd.DataFrame],
        crops: Optional[List[str]] = None,
        country: str = 'Global',
        model_name: str = 'XGBoost',
    ) -> Dict[str, pd.DataFrame]:
        """Apply trained ML models to projected climate scenarios to forecast yield."""
        if crops is None:
            crops = ['wheat', 'rice', 'maize', 'soybean']
        if not self._load_ml_models():
            logger.error("No ML models available.")
            return {}
        if model_name not in self._ml_models:
            model_name = next(iter(self._ml_models))
        model = self._ml_models[model_name]
        hist_stats = self._compute_historical_stats(country)
        results: Dict[str, pd.DataFrame] = {}
        for scenario_name, climate_df in scenarios.items():
            records = []
            for _, climate_row in climate_df.iterrows():
                for crop in crops:
                    baseline = hist_stats.get('country_yield_baseline', {}).get(crop)
                    feature_row = self._build_feature_row(
                        climate_row, crop, country, baseline, hist_stats
                    )
                    if feature_row is None:
                        continue
                    X = pd.DataFrame([feature_row], columns=self._feature_cols)
                    try:
                        pred = float(model.predict(X)[0])
                    except Exception as exc:
                        logger.warning(f"Prediction failed: {exc}")
                        pred = np.nan
                    records.append({
                        'Year': int(climate_row['Year']),
                        'Crop': crop,
                        'Predicted_Yield': pred,
                        'Temperature': climate_row.get('Temperature', np.nan),
                        'Rainfall': climate_row.get('Rainfall', np.nan),
                        'CO2_Emission': climate_row.get('CO2_Emission', np.nan),
                    })
            if records:
                results[scenario_name] = pd.DataFrame(records)
                logger.info(f"Scenario '{scenario_name}': {len(records)} predictions.")
        return results

    # ------------------------------------------------------------------
    # Uncertainty estimation
    # ------------------------------------------------------------------

    def uncertainty_estimates(
        self,
        years: List[int],
        feature: str = 'Temperature',
        n_bootstrap: int = 200,
        ci: float = 0.90,
    ) -> pd.DataFrame:
        """Bootstrap confidence intervals for a trend projection."""
        if feature not in self.trend_models:
            self.trend_analysis(feature)
        if feature not in self.trend_models:
            logger.error(f"Could not fit trend for '{feature}'.")
            return pd.DataFrame()
        info = self.trend_models[feature]
        poly: PolynomialFeatures = info['poly']
        base_model: LinearRegression = info['model']
        X_hist = self.df['Year'].values.reshape(-1, 1)
        y_hist = self.df[feature].values
        mask = ~np.isnan(y_hist)
        X_clean, y_clean = X_hist[mask], y_hist[mask]
        residuals = y_clean - base_model.predict(poly.transform(X_clean))
        X_future = np.array(years).reshape(-1, 1)
        X_poly_future = poly.transform(X_future)
        point_pred = base_model.predict(X_poly_future)
        alpha = (1 - ci) / 2
        boot_preds = np.zeros((n_bootstrap, len(years)))
        rng = np.random.default_rng(42)
        for i in range(n_bootstrap):
            sampled = rng.choice(residuals, size=len(years), replace=True)
            boot_preds[i] = point_pred + sampled
        lower = np.quantile(boot_preds, alpha, axis=0)
        upper = np.quantile(boot_preds, 1 - alpha, axis=0)
        result = pd.DataFrame({
            'Year': years,
            'mean_prediction': point_pred,
            f'lower_{int(ci * 100)}ci': lower,
            f'upper_{int(ci * 100)}ci': upper,
        })
        logger.info(f"Uncertainty for '{feature}' ({int(ci*100)}% CI):\n{result.to_string(index=False)}")
        return result

    def yield_uncertainty_estimates(
        self,
        scenario_yields: pd.DataFrame,
        n_bootstrap: int = 200,
        ci: float = 0.90,
    ) -> pd.DataFrame:
        """Confidence intervals for yield predictions using inter-model spread."""
        if not self._load_ml_models() or len(self._ml_models) < 2:
            logger.warning("Need ≥2 models for yield uncertainty.")
            return scenario_yields
        alpha = (1 - ci) / 2
        rng = np.random.default_rng(42)
        augmented = scenario_yields.copy()
        lowers, uppers = [], []
        hist_stats = self._compute_historical_stats('Global')
        for _, row in scenario_yields.iterrows():
            climate_row = pd.Series({
                'Year': row['Year'],
                'Temperature': row.get('Temperature', np.nan),
                'Rainfall': row.get('Rainfall', np.nan),
                'CO2_Emission': row.get('CO2_Emission', np.nan),
            })
            crop = row.get('Crop', 'wheat')
            baseline = hist_stats.get('country_yield_baseline', {}).get(crop)
            feature_row = self._build_feature_row(
                climate_row, crop, 'Global', baseline, hist_stats
            )
            if feature_row is None:
                lowers.append(np.nan); uppers.append(np.nan)
                continue
            X = pd.DataFrame([feature_row], columns=self._feature_cols)
            model_preds = []
            for m in self._ml_models.values():
                try:
                    model_preds.append(float(m.predict(X)[0]))
                except Exception:
                    pass
            if not model_preds:
                lowers.append(np.nan); uppers.append(np.nan)
                continue
            model_preds = np.array(model_preds)
            boot = rng.normal(loc=model_preds.mean(), scale=max(model_preds.std(), 0.1),
                              size=n_bootstrap)
            lowers.append(float(np.quantile(boot, alpha)))
            uppers.append(float(np.quantile(boot, 1 - alpha)))
        augmented[f'lower_{int(ci * 100)}ci'] = lowers
        augmented[f'upper_{int(ci * 100)}ci'] = uppers
        return augmented


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    data_path = 'data/processed/processed_data.csv'
    if not os.path.exists(data_path):
        logger.error(f"Data not found at '{data_path}'")
        sys.exit(1)
    df = pd.read_csv(data_path)
    predictor = FuturePredictor(df, model_path='models/')
    scenarios = predictor.climate_scenario_analysis()
    os.makedirs('output', exist_ok=True)
    for name, sdf in scenarios.items():
        sdf.to_csv(f'output/future_scenario_{name}.csv', index=False)
        logger.info(f"Saved scenario '{name}'")
    yield_forecasts = predictor.predict_future_yields(scenarios)
    for name, ydf in yield_forecasts.items():
        ydf.to_csv(f'output/future_yield_{name}.csv', index=False)
    unc = predictor.uncertainty_estimates([2026, 2030, 2040, 2050], feature='Temperature', ci=0.90)
    if not unc.empty:
        unc.to_csv('output/temperature_uncertainty.csv', index=False)
    print("\nFuture prediction pipeline complete.")
