import pandas as pd
import numpy as np
from typing import List, Tuple
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Advanced feature engineering for climate-agriculture prediction
    
    Master Prompt Context:
    Domain-specific features based on agricultural science:
    1. Climate stress indices (combination of temperature, rainfall, CO2)
    2. Seasonal patterns (growing season indicators)
    3. Historical trends (momentum indicators)
    4. Geographical features (crop-specific climate zones)
    5. Interaction terms (complex climate scenarios)
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_list = []
    
    def climate_stress_index(self) -> pd.DataFrame:
        """
        Create composite climate stress index
        
        Combines multiple climate variables into single stress indicator:
        Stress = w1*Temperature_Anomaly + w2*Rainfall_Deficit + w3*CO2_Effect
        
        Where weights are based on crop-specific sensitivity
        """
        logger.info("Creating climate stress index...")
        
        # Normalize variables
        temp_std = self.df['Temperature'].std()
        temp_normalized = (self.df['Temperature'] - self.df['Temperature'].mean()) / (temp_std if temp_std > 0 else 1.0)
        
        rain_std = self.df['Rainfall'].std()
        rain_normalized = (self.df['Rainfall'] - self.df['Rainfall'].mean()) / (rain_std if rain_std > 0 else 1.0)
        
        co2_std = self.df['CO2_Emission'].std()
        co2_normalized = (self.df['CO2_Emission'] - self.df['CO2_Emission'].mean()) / (co2_std if co2_std > 0 else 1.0)
        
        # Weights based on agricultural impact research
        self.df['Climate_Stress_Index'] = (
            0.45 * temp_normalized +  # Temperature is most important
            0.30 * (-rain_normalized) +  # Rainfall deficit (negative is stress)
            0.25 * co2_normalized  # CO2 effect
        )
        
        self.feature_list.append('Climate_Stress_Index')
        logger.info("Created Climate_Stress_Index")
        
        return self.df
    
    def temperature_rainfall_ratio(self) -> pd.DataFrame:
        """
        Create temperature-rainfall ratio
        
        Indicates aridity: high ratio means hot and dry
        Risk of drought stress
        """
        logger.info("Creating temperature-rainfall ratio...")
        
        # Avoid division by zero
        self.df['Temp_Rain_Ratio'] = (
            self.df['Temperature'] / (self.df['Rainfall'] + 1)
        )
        
        self.feature_list.append('Temp_Rain_Ratio')
        logger.info("Created Temp_Rain_Ratio")
        
        return self.df
    
    def growing_season_indicator(self) -> pd.DataFrame:
        """
        Create growing season quality indicator
        
        Based on temperature and rainfall patterns during growing season
        Optimal growing conditions: 
        - Temperature 15-25°C (crop-specific)
        - Adequate rainfall 400-700mm
        """
        logger.info("Creating growing season indicator...")
        
        # Ideal temperature range (15-25°C)
        ideal_temp_score = 1 - np.abs(self.df['Temperature'] - 20) / 20
        ideal_temp_score = np.clip(ideal_temp_score, 0, 1)
        
        # Ideal rainfall range (400-700mm)
        ideal_rain_score = np.where(
            (self.df['Rainfall'] >= 400) & (self.df['Rainfall'] <= 700),
            1.0,
            1 - np.abs(self.df['Rainfall'] - 550) / 550
        )
        ideal_rain_score = np.clip(ideal_rain_score, 0, 1)
        
        # Combine scores
        self.df['Growing_Season_Quality'] = (
            0.6 * ideal_temp_score + 0.4 * ideal_rain_score
        )
        
        self.feature_list.append('Growing_Season_Quality')
        logger.info("Created Growing_Season_Quality")
        
        return self.df
    
    def extreme_weather_risk(self) -> pd.DataFrame:
        """
        Create extreme weather risk indicator
        
        Combines temperature extremes and rainfall variability
        """
        logger.info("Creating extreme weather risk indicator...")
        
        # Temperature extremes (deviation from optimal)
        temp_risk = np.abs(self.df['Temperature'] - 20) / 20
        
        # Rainfall extremes (too much or too little)
        rain_optimal = 550  # mm
        rain_risk = np.abs(self.df['Rainfall'] - rain_optimal) / rain_optimal
        
        # Combine with weights
        self.df['Extreme_Weather_Risk'] = (
            0.6 * np.clip(temp_risk, 0, 1) +
            0.4 * np.clip(rain_risk, 0, 1)
        )
        
        self.feature_list.append('Extreme_Weather_Risk')
        logger.info("Created Extreme_Weather_Risk")
        
        return self.df
    
    def lagged_features(self, lags: List[int] = [1, 2]) -> pd.DataFrame:
        """
        Create lagged features for temporal patterns
        
        Captures carry-over effects from previous years
        """
        logger.info(f"Creating lagged features with lags: {lags}...")
        
        # Group by country and crop for proper lagging
        for lag in lags:
            self.df[f'Yield_Lag_{lag}'] = self.df.groupby(['Country', 'Crop'])['Yield'].shift(lag)
            self.df[f'Temp_Lag_{lag}'] = self.df.groupby(['Country', 'Crop'])['Temperature'].shift(lag)
            self.df[f'Rain_Lag_{lag}'] = self.df.groupby(['Country', 'Crop'])['Rainfall'].shift(lag)
            
            self.feature_list.extend([f'Yield_Lag_{lag}', f'Temp_Lag_{lag}', f'Rain_Lag_{lag}'])
        
        logger.info(f"Created {len(lags) * 3} lagged features")
        
        return self.df
    
    def rolling_statistics(self, window: int = 3) -> pd.DataFrame:
        """
        Create rolling window statistics
        
        Captures recent trends and volatility
        """
        logger.info(f"Creating rolling statistics with window={window}...")
        
        # Rolling mean
        rolling_mean = self.df.groupby(['Country', 'Crop'])['Yield'].rolling(window, min_periods=1).mean()
        rolling_mean.index = rolling_mean.index.get_level_values(2)
        self.df[f'Yield_Rolling_Mean_{window}'] = rolling_mean
        
        # Rolling std (volatility)
        rolling_temp_std = self.df.groupby(['Country', 'Crop'])['Temperature'].rolling(window, min_periods=1).std()
        rolling_temp_std.index = rolling_temp_std.index.get_level_values(2)
        self.df[f'Temp_Rolling_Std_{window}'] = rolling_temp_std.fillna(0)
        
        rolling_rain_std = self.df.groupby(['Country', 'Crop'])['Rainfall'].rolling(window, min_periods=1).std()
        rolling_rain_std.index = rolling_rain_std.index.get_level_values(2)
        self.df[f'Rain_Rolling_Std_{window}'] = rolling_rain_std.fillna(0)
        
        # Fill standard deviation NaNs with 0 (which happens for window=1 or start of series)
        self.df[f'Temp_Rolling_Std_{window}'] = self.df[f'Temp_Rolling_Std_{window}'].fillna(0)
        self.df[f'Rain_Rolling_Std_{window}'] = self.df[f'Rain_Rolling_Std_{window}'].fillna(0)
        
        self.feature_list.extend([
            f'Yield_Rolling_Mean_{window}',
            f'Temp_Rolling_Std_{window}',
            f'Rain_Rolling_Std_{window}'
        ])
        
        logger.info(f"Created rolling statistics")
        
        return self.df
    
    def crop_specific_features(self) -> pd.DataFrame:
        """
        Create crop-specific indicator variables
        
        Different crops have different climate sensitivities
        """
        logger.info("Creating crop-specific features...")
        
        # One-hot encode crops
        crop_dummies = pd.get_dummies(self.df['Crop'], prefix='Crop')
        
        # Crop-climate interaction features
        for col in crop_dummies.columns:
            # Ensure boolean dummies are treated as numbers
            dummy_num = crop_dummies[col].astype(int)
            self.df[f'{col}_Temp'] = dummy_num * self.df['Temperature']
            self.df[f'{col}_Rain'] = dummy_num * self.df['Rainfall']
            self.feature_list.extend([f'{col}_Temp', f'{col}_Rain'])
        
        logger.info(f"Created crop-specific features")
        
        return self.df
    
    def geographical_features(self) -> pd.DataFrame:
        """
        Create geographical and regional features
        
        Different regions have different baseline productivities
        """
        logger.info("Creating geographical features...")
        
        # Country-level average yield (baseline productivity)
        country_baseline = self.df.groupby('Country')['Yield'].mean()
        self.df['Country_Yield_Baseline'] = self.df['Country'].map(country_baseline)
        
        # Deviation from country baseline
        self.df['Yield_vs_Baseline'] = self.df['Yield'] - self.df['Country_Yield_Baseline']
        
        # Country yield momentum (trend) with safety checks
        def get_trend(x):
            val = x.dropna()
            if len(val) < 2:
                return 0.0
            try:
                return np.polyfit(range(len(val)), val, 1)[0]
            except Exception:
                return 0.0

        # Avoid warnings and handle group sizes safely
        country_trend = self.df.groupby('Country')['Yield'].apply(get_trend)
        self.df['Country_Yield_Trend'] = self.df['Country'].map(country_trend)
        
        self.feature_list.extend([
            'Country_Yield_Baseline',
            'Yield_vs_Baseline',
            'Country_Yield_Trend'
        ])
        
        logger.info("Created geographical features")
        
        return self.df
    
    def temporal_features(self) -> pd.DataFrame:
        """
        Create temporal features
        
        Capture long-term climate trends
        """
        logger.info("Creating temporal features...")
        
        # Years since baseline (for trend analysis)
        min_year = self.df['Year'].min()
        self.df['Years_Since_Baseline'] = self.df['Year'] - min_year
        
        # Decade (for decadal climate patterns)
        self.df['Decade'] = (self.df['Year'] // 10) * 10
        
        # Climate change indicator (increasing trend)
        self.df['Climate_Change_Index'] = self.df['Years_Since_Baseline'] * 0.03  # Approximate warming rate
        
        self.feature_list.extend([
            'Years_Since_Baseline',
            'Decade',
            'Climate_Change_Index'
        ])
        
        logger.info("Created temporal features")
        
        return self.df
    
    def co2_and_adaptation(self) -> pd.DataFrame:
        """
        Create CO2 and adaptation-related features
        """
        logger.info("Creating CO2 and adaptation features...")
        
        # CO2 anomaly (deviation from historical trend)
        co2_mean = self.df['CO2_Emission'].mean()
        self.df['CO2_Anomaly'] = self.df['CO2_Emission'] - co2_mean
        
        # CO2 effect on yield (CO2 fertilization)
        self.df['CO2_Fertilization_Effect'] = np.log(self.df['CO2_Emission'] / 280) * 5  # Simplified
        
        # Climate-CO2 interaction
        self.df['Climate_CO2_Interaction'] = self.df['Climate_Stress_Index'] * self.df['CO2_Anomaly']
        
        self.feature_list.extend([
            'CO2_Anomaly',
            'CO2_Fertilization_Effect',
            'Climate_CO2_Interaction'
        ])
        
        logger.info("Created CO2 and adaptation features")
        
        return self.df
    
    def get_feature_importance_baseline(self) -> pd.DataFrame:
        """
        Calculate correlation-based feature importance baseline
        """
        logger.info("Calculating feature importance baseline...")
        
        numeric_features = self.df[self.feature_list].select_dtypes(include=[np.number])
        correlations = numeric_features.corrwith(self.df['Yield']).sort_values(ascending=False)
        
        importance_df = pd.DataFrame({
            'Feature': correlations.index,
            'Correlation_with_Yield': correlations.values,
            'Abs_Correlation': np.abs(correlations.values)
        }).sort_values('Abs_Correlation', ascending=False)
        
        logger.info("\nTop 10 Features by Correlation:\n" + str(importance_df.head(10)))
        
        return importance_df
    
    def engineer_all_features(self) -> Tuple[pd.DataFrame, List, pd.DataFrame]:
        """
        Execute full feature engineering pipeline
        """
        logger.info("Starting feature engineering pipeline...")
        
        self.climate_stress_index()
        self.temperature_rainfall_ratio()
        self.growing_season_indicator()
        self.extreme_weather_risk()
        self.lagged_features(lags=[1, 2])
        self.rolling_statistics(window=3)
        self.crop_specific_features()
        self.geographical_features()
        self.temporal_features()
        self.co2_and_adaptation()
        
        logger.info(f"Feature engineering complete! Created {len(self.feature_list)} new features")
        
        # Get importance ranking
        importance = self.get_feature_importance_baseline()
        
        return self.df, self.feature_list, importance

if __name__ == "__main__":
    import sys
    data_path = 'data/processed/cleaned_data.csv'
    if not os.path.exists(data_path):
        logger.error(f"Cleaned data not found at {data_path}. Please run data clean script first.")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    
    # Engineer features
    engineer = FeatureEngineer(df)
    df_engineered, features, importance = sorted_results = engineer.engineer_all_features()
    
    # Save engineered data
    os.makedirs('data/processed', exist_ok=True)
    df_engineered.to_csv('data/processed/engineered_data.csv', index=False)
    
    os.makedirs('output', exist_ok=True)
    importance.to_csv('output/feature_importance_baseline.csv', index=False)
    
    print("Feature engineering complete!")
    print(f"Total features created: {len(features)}")
    print("\nTop 10 Most Important Features:")
    print(importance.head(10))
