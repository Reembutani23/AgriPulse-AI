import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EDAAnalyzer:
    """
    Comprehensive exploratory data analysis for climate-food dataset.
    This class provides tools to generate a comprehensive dataset overview,
    statistical summaries, and various publication-quality visualizations
    to understand relationships, trends, and anomalies in agricultural and climate data.
    """
    
    def __init__(self, df: pd.DataFrame, output_dir: str = 'output'):
        """
        Initialize the EDA Analyzer.
        
        Args:
            df (pd.DataFrame): The dataset to analyze.
            output_dir (str): Directory to save output plots and reports.
        """
        try:
            self.df = df
            self.output_dir = output_dir
            self.numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.categorical_cols = df.select_dtypes(include='object').columns
            
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Set style
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (14, 8)
            logger.info("EDAAnalyzer initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing EDAAnalyzer: {str(e)}")
            raise
    
    def dataset_overview(self) -> Dict:
        """
        Generate comprehensive dataset overview.
        
        Analyzes shape, dtypes, missing values, and identifies data quality issues.
        Implications for modeling: Helps determine necessary preprocessing steps 
        like imputation and data type conversions.
        """
        logger.info("Generating dataset overview...")
        try:
            overview = {
                'shape': self.df.shape,
                'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
                'dtypes': self.df.dtypes.to_dict(),
                'missing_values': self.df.isnull().sum().to_dict(),
                'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
                'duplicate_rows': self.df.duplicated().sum(),
            }
            logger.info(f"Dataset overview: {overview['shape']}")
            return overview
        except Exception as e:
            logger.error(f"Error generating dataset overview: {str(e)}")
            raise
    
    def statistical_summary(self) -> pd.DataFrame:
        """
        Generate statistical summary for numeric columns.
        
        Implications for modeling: Helps identify the scale of different features,
        potential skewness (needing transformation), and overall data distribution.
        """
        logger.info("Generating statistical summary...")
        try:
            stats = self.df[self.numeric_cols].describe()
            
            # Add custom statistics
            stats.loc['skewness'] = self.df[self.numeric_cols].skew()
            stats.loc['kurtosis'] = self.df[self.numeric_cols].kurtosis()
            stats.loc['cv'] = self.df[self.numeric_cols].std() / self.df[self.numeric_cols].mean()
            
            return stats
        except Exception as e:
            logger.error(f"Error generating statistical summary: {str(e)}")
            raise
    
    def temperature_vs_yield_analysis(self):
        """
        Analyze relationship between temperature and crop yield.
        
        Implications for modeling: Reveals if temperature has a linear or non-linear 
        effect on yield, informing whether polynomial features or non-linear models 
        (like trees) might be beneficial.
        """
        logger.info("Generating Temperature vs Yield analysis...")
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Temperature vs Yield (All Crops)',
                              'Temperature vs Yield by Crop',
                              'Correlation Heatmap',
                              'Distribution Comparison')
            )
            
            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=self.df['Temperature'],
                    y=self.df['Yield'],
                    mode='markers',
                    marker=dict(size=4, opacity=0.6, color=self.df['Temperature'], colorscale='Viridis'),
                    name='Data Points'
                ),
                row=1, col=1
            )
            
            # By crop type
            if 'Crop' in self.df.columns:
                for crop in self.df['Crop'].unique():
                    crop_data = self.df[self.df['Crop'] == crop]
                    fig.add_trace(
                        go.Scatter(
                            x=crop_data['Temperature'],
                            y=crop_data['Yield'],
                            mode='markers',
                            name=crop,
                            marker=dict(size=6, opacity=0.7)
                        ),
                        row=1, col=2
                    )
            
            fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
            fig.update_yaxes(title_text="Yield", row=1, col=1)
            
            # Note: Assuming this is run in an interactive environment. For scripts, saving might be preferred.
            # fig.write_html(os.path.join(self.output_dir, 'temp_vs_yield.html'))
            
            # Compute correlation
            if 'Temperature' in self.df.columns and 'Yield' in self.df.columns:
                corr = self.df['Temperature'].corr(self.df['Yield'])
                logger.info(f"Temperature-Yield Correlation: {corr:.4f}")
            
            return fig
        except Exception as e:
            logger.error(f"Error in temperature vs yield analysis: {str(e)}")
            raise
    
    def rainfall_vs_yield_analysis(self):
        """
        Analyze relationship between rainfall and crop yield.
        
        Implications for modeling: Helps understand the impact of precipitation.
        If relationship is weak overall but strong for specific crops, interactions
        between Rainfall and Crop type will be important features.
        """
        logger.info("Generating Rainfall vs Yield analysis...")
        try:
            if not all(col in self.df.columns for col in ['Rainfall', 'Yield', 'Crop', 'Country', 'Year']):
                logger.warning("Missing columns for full rainfall vs yield analysis. Generating simplified plot.")
                fig = px.scatter(
                    self.df, x='Rainfall', y='Yield',
                    title='Rainfall vs Crop Yield',
                    labels={'Rainfall': 'Rainfall (mm)', 'Yield': 'Crop Yield (kg/ha)'}
                )
            else:
                fig = px.scatter(
                    self.df,
                    x='Rainfall',
                    y='Yield',
                    color='Crop',
                    size='Rainfall',
                    hover_data=['Country', 'Year'],
                    title='Rainfall vs Crop Yield by Type',
                    labels={'Rainfall': 'Rainfall (mm)', 'Yield': 'Crop Yield (kg/ha)'}
                )
            
            if 'Rainfall' in self.df.columns and 'Yield' in self.df.columns:
                corr = self.df['Rainfall'].corr(self.df['Yield'])
                logger.info(f"Rainfall-Yield Correlation: {corr:.4f}")
            
            return fig
        except Exception as e:
            logger.error(f"Error in rainfall vs yield analysis: {str(e)}")
            raise
    
    def correlation_matrix(self):
        """
        Generate correlation heatmap for all numeric features.
        
        Implications for modeling: Identifies strongest linear predictors for yield.
        Detects multicollinearity (e.g., highly correlated climate variables) which 
        might require dimensionality reduction (PCA) or regularization (Ridge/Lasso).
        """
        logger.info("Generating correlation matrix...")
        try:
            plt.figure(figsize=(12, 8))
            
            corr_matrix = self.df[self.numeric_cols].corr()
            
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={'label': 'Correlation Coefficient'}
            )
            
            plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            output_file = os.path.join(self.output_dir, 'correlation_heatmap.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved correlation heatmap to {output_file}")
            plt.close() # Close to free memory
            
            return corr_matrix
        except Exception as e:
            logger.error(f"Error generating correlation matrix: {str(e)}")
            raise
    
    def time_series_trends(self):
        """
        Analyze temporal trends in climate variables.
        
        Implications for modeling: Reveals non-stationary behavior, overall climate 
        change trends, and target variable drifts. May require time-based train/test splits.
        """
        logger.info("Generating time series trends...")
        try:
            if 'Year' not in self.df.columns:
                logger.warning("'Year' column not found, skipping time series trends.")
                return None
                
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Temperature Trend', 'Rainfall Trend',
                              'CO2 Emission Trend', 'Yield Trend by Crop')
            )
            
            # Handle potential missing columns gracefully
            cols_to_agg = {'Temperature': 'mean', 'Rainfall': 'mean', 'Yield': 'mean'}
            if 'CO2_Emission' in self.df.columns:
                cols_to_agg['CO2_Emission'] = 'mean'
                
            yearly_avg = self.df.groupby('Year').agg({k: v for k, v in cols_to_agg.items() if k in self.df.columns}).reset_index()
            
            # Temperature trend
            if 'Temperature' in yearly_avg.columns:
                fig.add_trace(
                    go.Scatter(x=yearly_avg['Year'], y=yearly_avg['Temperature'],
                              mode='lines+markers', name='Temperature'),
                    row=1, col=1
                )
            
            # Rainfall trend
            if 'Rainfall' in yearly_avg.columns:
                fig.add_trace(
                    go.Scatter(x=yearly_avg['Year'], y=yearly_avg['Rainfall'],
                              mode='lines+markers', name='Rainfall'),
                    row=1, col=2
                )
            
            # CO2 trend
            if 'CO2_Emission' in yearly_avg.columns:
                fig.add_trace(
                    go.Scatter(x=yearly_avg['Year'], y=yearly_avg['CO2_Emission'],
                              mode='lines+markers', name='CO2'),
                    row=2, col=1
                )
            
            # Yield by crop
            if 'Crop' in self.df.columns and 'Yield' in self.df.columns:
                for crop in self.df['Crop'].unique():
                    crop_yearly = self.df[self.df['Crop'] == crop].groupby('Year')['Yield'].mean()
                    fig.add_trace(
                        go.Scatter(x=crop_yearly.index, y=crop_yearly.values,
                                  mode='lines', name=f'Yield-{crop}'),
                        row=2, col=2
                    )
            
            fig.update_xaxes(title_text="Year")
            fig.update_yaxes(title_text="Value")
            fig.update_layout(height=800)
            
            return fig
        except Exception as e:
            logger.error(f"Error generating time series trends: {str(e)}")
            raise
    
    def distribution_analysis(self):
        """
        Analyze feature distributions.
        
        Implications for modeling: Helps decide if log transformations are needed 
        for skewed data (like rainfall or emissions) or if robust scaling is required.
        """
        logger.info("Generating distribution analysis...")
        try:
            features = ['Temperature', 'Rainfall', 'CO2_Emission', 'Humidity', 'Yield']
            # Only use features present in the dataset
            features = [f for f in features if f in self.df.columns]
            
            if not features:
                logger.warning("None of the specified features found for distribution analysis.")
                return
                
            n_features = len(features)
            cols = 3
            rows = (n_features + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
            axes = axes.flatten() if n_features > 1 else [axes]
            
            for idx, feature in enumerate(features):
                ax = axes[idx]
                
                # Histogram with KDE
                self.df[feature].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_title(f'{feature} Distribution', fontweight='bold')
                ax.set_xlabel(feature)
                ax.set_ylabel('Frequency')
                
                # Add statistics
                mean = self.df[feature].mean()
                median = self.df[feature].median()
                ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
                ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
                ax.legend()
            
            # Hide empty subplots
            for idx in range(n_features, len(axes)):
                fig.delaxes(axes[idx])
                
            plt.tight_layout()
            output_file = os.path.join(self.output_dir, 'distributions.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved distributions to {output_file}")
            plt.close()
        except Exception as e:
            logger.error(f"Error generating distribution analysis: {str(e)}")
            raise
    
    def outlier_detection(self) -> Dict:
        """
        Detect outliers using IQR method.
        
        Implications for modeling: Identifies extreme values that could negatively 
        impact distance-based models or linear regression. Helps decide if winsorization, 
        trimming, or robust loss functions (like Huber loss) are needed.
        """
        logger.info("Detecting outliers...")
        try:
            outliers = {}
            
            for col in self.numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    outliers[col] = {
                        'count': int(outlier_count),
                        'percentage': float((outlier_count / len(self.df)) * 100),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
            
            logger.info(f"Found {len(outliers)} columns with outliers")
            return outliers
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            raise
    
    def crop_specific_analysis(self):
        """
        Analyze patterns specific to each crop.
        
        Implications for modeling: Confirms whether different crops have significantly
        different baseline yields and sensitivities, suggesting the potential need for
        crop-specific models or mixed-effects modeling.
        """
        logger.info("Generating crop-specific analysis...")
        try:
            if 'Crop' not in self.df.columns or 'Yield' not in self.df.columns:
                logger.warning("Missing 'Crop' or 'Yield' columns. Skipping crop specific analysis.")
                return None
                
            fig = px.box(
                self.df,
                x='Crop',
                y='Yield',
                color='Crop',
                title='Yield Distribution by Crop Type'
            )
            
            # Statistical comparison
            agg_dict = {'Yield': ['mean', 'std', 'min', 'max']}
            for col in ['Temperature', 'Rainfall', 'CO2_Emission']:
                if col in self.df.columns:
                    agg_dict[col] = 'mean'
                    
            crop_stats = self.df.groupby('Crop').agg(agg_dict)
            logger.info(f"Crop-specific statistics:\n{crop_stats}")
            
            return crop_stats, fig
        except Exception as e:
            logger.error(f"Error in crop-specific analysis: {str(e)}")
            raise
    
    def generate_eda_report(self, output_path='output/eda_report.html'):
        """
        Generate comprehensive HTML EDA report.
        """
        logger.info("Generating comprehensive EDA report...")
        try:
            # Note: For a fully integrated HTML report, libraries like pandas-profiling
            # (now ydata-profiling) or sweetviz are ideal. Here we could use plotly's 
            # write_html or similar mechanisms.
            # As a placeholder, we log the action:
            logger.info(f"EDA report generated and intended to be saved to {output_path}")
            # Real implementation might involve building an HTML string or using a reporting tool.
        except Exception as e:
            logger.error(f"Error generating EDA report: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Provide path to sample data
        data_path = 'data/raw/crop_data.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            # Run EDA
            analyzer = EDAAnalyzer(df)
            
            print("\n=== DATASET OVERVIEW ===")
            overview = analyzer.dataset_overview()
            for key, value in overview.items():
                print(f"{key}: {value}")
            
            print("\n=== STATISTICAL SUMMARY ===")
            print(analyzer.statistical_summary())
            
            print("\n=== CORRELATION MATRIX ===")
            corr_matrix = analyzer.correlation_matrix()
            
            print("\n=== OUTLIER ANALYSIS ===")
            outliers = analyzer.outlier_detection()
            print(f"Outliers found: {outliers}")
            
            # Generate visualizations
            # Note: In a script running via CLI, we might want to save figs instead of just showing them.
            # To view them, run this script interactively or use fig.show() in a Jupyter notebook.
            analyzer.temperature_vs_yield_analysis()
            analyzer.rainfall_vs_yield_analysis()
            analyzer.time_series_trends()
            analyzer.distribution_analysis()
            analyzer.crop_specific_analysis()
        else:
            logger.warning(f"Data file {data_path} not found. Please ensure the data exists to run the example.")
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
