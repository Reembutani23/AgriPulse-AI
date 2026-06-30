"""
Phase 4 – Data Preprocessing & Cleaning
========================================
Production-grade cleaning, validation, and feature-scaling pipeline for
the AgriPulse AI climate–agriculture dataset.

Capabilities
------------
* Multi-strategy missing-value imputation (smart/group-mean, forward-fill, KNN)
* Duplicate removal (exact + conceptual Country-Year-Crop keys)
* Physical-constraint validation with configurable bounds
* Outlier detection: IQR, Z-score, Isolation Forest — flags rather than
  removes by default so domain experts can review
* Data-quality scorecard per column and overall
* Cross-source consistency checks when multiple data sources exist
* Feature scaling (Standard / Robust / MinMax) with serialised scaler metadata
* Rich logging and structured cleaning reports (dict / JSON / HTML)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain-specific physical bounds
# ---------------------------------------------------------------------------

PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Temperature": (-50.0, 60.0),       # °C
    "Rainfall": (0.0, 10_000.0),        # mm
    "CO2_Emission": (280.0, 600.0),     # ppm
    "Humidity": (0.0, 100.0),           # %
    "Yield": (0.0, float("inf")),       # non-negative
    "Area": (0.0, float("inf")),        # non-negative
    "Production": (0.0, float("inf")),  # non-negative
    "Extreme_Weather_Events": (0.0, float("inf")),  # non-negative integer
}


# ═══════════════════════════════════════════════════════════════════════════
# DataCleaner
# ═══════════════════════════════════════════════════════════════════════════

class DataCleaner:
    """
    Comprehensive data cleaning and preprocessing pipeline.

    Usage::

        cleaner = DataCleaner(df)
        df_clean = cleaner.get_cleaned_data()
        report  = cleaner.generate_cleaning_report()
    """

    # Default key columns used to detect conceptual duplicates
    KEY_COLUMNS: List[str] = ["Country", "Year", "Crop"]

    # Columns that are identifiers or targets — never treated as regular
    # numeric features during outlier detection / scaling
    EXCLUDE_FROM_OUTLIERS: List[str] = ["Year", "Yield"]

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        physical_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_log: List[str] = []
        self.quality_scorecard: Dict[str, Any] = {}
        self.bounds = physical_bounds or PHYSICAL_BOUNDS

    # ------------------------------------------------------------------
    # 1. Missing-value handling
    # ------------------------------------------------------------------

    def handle_missing_values(self, strategy: str = "smart") -> pd.DataFrame:
        """
        Handle missing values using *strategy*:

        * ``'smart'``  — domain-aware group imputation (default)
        * ``'knn'``    — KNN imputation for numeric columns
        * ``'forward_fill'`` — forward + backward fill per country (time-series)
        * ``'mean'`` / ``'median'`` — global statistical imputation
        * ``'drop'``   — drop any row with *any* missing value
        """
        logger.info("Missing values BEFORE cleaning:\n%s", self.df.isnull().sum())

        # ── Stage 1: Drop columns with >50 % missing ──
        missing_pct = self.df.isnull().mean() * 100
        cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
        if cols_to_drop:
            logger.info("Dropping columns with >50%% missing: %s", cols_to_drop)
            self.df.drop(columns=cols_to_drop, inplace=True)
            self.cleaning_log.append(
                f"Dropped {len(cols_to_drop)} columns (>50% missing): {cols_to_drop}"
            )

        # ── Stage 2: Strategy-specific imputation ──
        if strategy == "smart":
            self._impute_smart()
        elif strategy == "knn":
            self._impute_knn()
        elif strategy == "forward_fill":
            self._impute_ffill()
        elif strategy in ("mean", "median"):
            self._impute_stat(strategy)
        elif strategy == "drop":
            before = len(self.df)
            self.df.dropna(inplace=True)
            self.cleaning_log.append(
                f"Dropped {before - len(self.df)} rows with any missing value"
            )
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy!r}")

        # ── Stage 3: Always drop rows where the target is missing ──
        if "Yield" in self.df.columns:
            before = len(self.df)
            self.df.dropna(subset=["Yield"], inplace=True)
            dropped = before - len(self.df)
            if dropped:
                self.cleaning_log.append(
                    f"Dropped {dropped} rows with missing target (Yield)"
                )

        logger.info("Missing values AFTER cleaning:\n%s", self.df.isnull().sum())
        return self.df

    # ---- private imputation helpers ----

    def _impute_smart(self) -> None:
        """Domain-aware imputation: group means → global mean fallback."""
        group_cols = [c for c in ("Country", "Year") if c in self.df.columns]

        for col, group in [
            ("Temperature", ["Country", "Year"]),
            ("Rainfall", ["Country"]),
            ("Humidity", ["Country"]),
        ]:
            if col not in self.df.columns or self.df[col].isnull().sum() == 0:
                continue
            available_groups = [g for g in group if g in self.df.columns]
            if available_groups:
                self.df[col] = self.df.groupby(available_groups)[col].transform(
                    lambda s: s.fillna(s.mean())
                )
            # Global fallback
            self.df[col] = self.df[col].fillna(self.df[col].mean())
            self.cleaning_log.append(
                f"Imputed {col} via group mean ({available_groups}) + global fallback"
            )

        # CO₂: forward-fill per country (represents a trend)
        if "CO2_Emission" in self.df.columns and self.df["CO2_Emission"].isnull().sum():
            if "Country" in self.df.columns:
                self.df["CO2_Emission"] = (
                    self.df.groupby("Country")["CO2_Emission"].ffill().bfill()
                )
            else:
                self.df["CO2_Emission"] = self.df["CO2_Emission"].ffill().bfill()
            self.cleaning_log.append("Forward/backward filled CO2_Emission")

        # Remaining numeric: global mean
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            n_miss = self.df[col].isnull().sum()
            if n_miss > 0:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
                self.cleaning_log.append(
                    f"Imputed {col} (global mean) — {n_miss} values"
                )

    def _impute_knn(self, n_neighbors: int = 5) -> None:
        """KNN imputation for all numeric columns with missing data."""
        try:
            from sklearn.impute import KNNImputer
        except ImportError:
            logger.warning("KNNImputer unavailable — falling back to smart strategy")
            return self._impute_smart()

        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return

        n_missing_before = self.df[num_cols].isnull().sum().sum()
        if n_missing_before == 0:
            return

        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        self.df[num_cols] = imputer.fit_transform(self.df[num_cols])
        self.cleaning_log.append(
            f"KNN imputation (k={n_neighbors}): filled {n_missing_before} values "
            f"across {len(num_cols)} numeric columns"
        )

    def _impute_ffill(self) -> None:
        """Time-series style forward + backward fill per country."""
        group_col = "Country" if "Country" in self.df.columns else None
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        for col in num_cols:
            n_miss = self.df[col].isnull().sum()
            if n_miss == 0:
                continue
            if group_col:
                self.df[col] = self.df.groupby(group_col)[col].ffill().bfill()
            else:
                self.df[col] = self.df[col].ffill().bfill()
            # Any stragglers filled with global mean
            remaining = self.df[col].isnull().sum()
            if remaining:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            self.cleaning_log.append(f"Forward/backward filled {col} — {n_miss} values")

    def _impute_stat(self, stat: str) -> None:
        """Simple mean or median imputation for all numeric columns."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            n_miss = self.df[col].isnull().sum()
            if n_miss == 0:
                continue
            fill_value = self.df[col].mean() if stat == "mean" else self.df[col].median()
            self.df[col] = self.df[col].fillna(fill_value)
            self.cleaning_log.append(f"Imputed {col} ({stat}={fill_value:.4f}) — {n_miss} values")

    # ------------------------------------------------------------------
    # 2. Duplicate removal
    # ------------------------------------------------------------------

    def handle_duplicates(self) -> pd.DataFrame:
        """Remove exact duplicates and conceptual duplicates (same key tuple)."""
        initial_rows = len(self.df)

        # Exact duplicates
        n_exact = self.df.duplicated().sum()
        if n_exact:
            self.df.drop_duplicates(inplace=True)
            self.cleaning_log.append(f"Removed {n_exact} exact duplicate rows")

        # Conceptual duplicates (Country-Year-Crop)
        key_cols = [c for c in self.KEY_COLUMNS if c in self.df.columns]
        if key_cols:
            n_concept = self.df.duplicated(subset=key_cols, keep="first").sum()
            if n_concept:
                self.df.drop_duplicates(subset=key_cols, keep="first", inplace=True)
                self.cleaning_log.append(
                    f"Removed {n_concept} conceptual duplicates (keys: {key_cols})"
                )

        total_removed = initial_rows - len(self.df)
        logger.info("Duplicate removal: %d rows removed", total_removed)
        return self.df

    # ------------------------------------------------------------------
    # 3. Physical-constraint validation
    # ------------------------------------------------------------------

    def remove_invalid_values(self) -> pd.DataFrame:
        """
        Drop rows where any value falls outside its physical bounds.

        Bounds are defined in ``self.bounds`` (defaults to PHYSICAL_BOUNDS).
        """
        logger.info("Validating physical constraints …")
        initial_rows = len(self.df)

        for col, (lo, hi) in self.bounds.items():
            if col not in self.df.columns:
                continue
            mask = (self.df[col] >= lo) & (self.df[col] <= hi)
            n_invalid = (~mask).sum()
            if n_invalid:
                self.df = self.df.loc[mask]
                logger.info(
                    "%s: removed %d rows outside [%.1f, %.1f]",
                    col, n_invalid, lo, hi,
                )

        total_removed = initial_rows - len(self.df)
        self.cleaning_log.append(
            f"Physical-constraint validation: removed {total_removed} rows"
        )
        logger.info("Total rows removed by constraint validation: %d", total_removed)
        return self.df

    # ------------------------------------------------------------------
    # 4. Outlier detection (flag, don't remove)
    # ------------------------------------------------------------------

    def handle_outliers(
        self,
        method: str = "iqr",
        threshold: float = 1.5,
        contamination: float = 0.05,
    ) -> pd.DataFrame:
        """
        Flag outliers using the chosen *method*:

        * ``'iqr'``              — inter-quartile range (default)
        * ``'zscore'``           — absolute Z-score > *threshold* (default 3)
        * ``'isolation_forest'`` — sklearn IsolationForest

        Outlier columns are boolean flags named ``<col>_outlier``.
        A summary column ``outlier_score`` counts how many features are flagged.
        """
        logger.info("Outlier detection: method=%s …", method)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [
            c for c in numeric_cols
            if c not in self.EXCLUDE_FROM_OUTLIERS and not c.endswith("_outlier")
        ]

        if method == "iqr":
            self._outliers_iqr(cols, threshold)
        elif method == "zscore":
            # For Z-score, a typical threshold is 3 rather than 1.5
            z_thresh = threshold if threshold > 2 else 3.0
            self._outliers_zscore(cols, z_thresh)
        elif method == "isolation_forest":
            self._outliers_isolation_forest(cols, contamination)
        else:
            raise ValueError(f"Unknown outlier method: {method!r}")

        # Summary score: count of flagged features per row
        outlier_flags = [c for c in self.df.columns if c.endswith("_outlier")]
        if outlier_flags:
            self.df["outlier_score"] = self.df[outlier_flags].sum(axis=1).astype(int)

        return self.df

    def _outliers_iqr(self, cols: List[str], threshold: float) -> None:
        for col in cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
            mask = (self.df[col] < lo) | (self.df[col] > hi)
            if mask.sum():
                self.df[f"{col}_outlier"] = mask
                logger.info("%s (IQR): flagged %d outliers", col, mask.sum())

    def _outliers_zscore(self, cols: List[str], threshold: float) -> None:
        for col in cols:
            mean = self.df[col].mean()
            std = self.df[col].std()
            if std == 0:
                continue
            z = ((self.df[col] - mean) / std).abs()
            mask = z > threshold
            if mask.sum():
                self.df[f"{col}_outlier"] = mask
                logger.info(
                    "%s (Z-score>%.1f): flagged %d outliers",
                    col, threshold, mask.sum(),
                )

    def _outliers_isolation_forest(
        self, cols: List[str], contamination: float
    ) -> None:
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.warning("IsolationForest unavailable — falling back to IQR")
            return self._outliers_iqr(cols, 1.5)

        if not cols:
            return

        iso = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        preds = iso.fit_predict(self.df[cols].fillna(0))
        mask = preds == -1
        n_flagged = mask.sum()
        if n_flagged:
            self.df["isolation_forest_outlier"] = mask
            logger.info(
                "Isolation Forest: flagged %d outliers (contamination=%.2f)",
                n_flagged, contamination,
            )
            self.cleaning_log.append(
                f"Isolation Forest flagged {n_flagged} outliers"
            )

    # ------------------------------------------------------------------
    # 5. Data-type validation
    # ------------------------------------------------------------------

    def data_type_validation(self) -> pd.DataFrame:
        """
        Coerce columns to appropriate data types.
        """
        logger.info("Validating data types …")

        type_mapping: Dict[str, str] = {
            "Year": "int32",
            "Country": "category",
            "Crop": "category",
        }
        for col, dtype in type_mapping.items():
            if col in self.df.columns:
                try:
                    self.df[col] = self.df[col].astype(dtype)
                    logger.info("Converted %s → %s", col, dtype)
                except Exception as exc:
                    logger.warning("Could not convert %s: %s", col, exc)

        numeric_cols = [
            "Temperature", "Rainfall", "CO2_Emission", "Humidity",
            "Yield", "Area", "Production",
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        return self.df

    # ------------------------------------------------------------------
    # 6. Cross-source consistency checks
    # ------------------------------------------------------------------

    def cross_validate_sources(
        self,
        source_col: str = "Data_Source",
    ) -> Dict[str, Any]:
        """
        When the dataset merges multiple data sources, compare overlap and
        consistency for shared key tuples.

        Returns a summary dict (also appended to the cleaning log).
        """
        if source_col not in self.df.columns:
            logger.info("No '%s' column — skipping cross-source validation", source_col)
            return {}

        sources = self.df[source_col].unique().tolist()
        if len(sources) < 2:
            logger.info("Only one data source present — nothing to cross-validate")
            return {"sources": sources, "status": "single_source"}

        key_cols = [c for c in self.KEY_COLUMNS if c in self.df.columns]
        report: Dict[str, Any] = {"sources": sources, "comparisons": []}

        for i, src_a in enumerate(sources):
            for src_b in sources[i + 1:]:
                df_a = self.df[self.df[source_col] == src_a]
                df_b = self.df[self.df[source_col] == src_b]

                if key_cols:
                    keys_a = set(df_a[key_cols].apply(tuple, axis=1))
                    keys_b = set(df_b[key_cols].apply(tuple, axis=1))
                    overlap = len(keys_a & keys_b)
                else:
                    overlap = 0

                comparison = {
                    "source_a": src_a,
                    "source_b": src_b,
                    "rows_a": len(df_a),
                    "rows_b": len(df_b),
                    "key_overlap": overlap,
                }
                report["comparisons"].append(comparison)
                logger.info(
                    "Cross-source %s vs %s: %d / %d overlapping keys",
                    src_a, src_b, overlap, max(len(keys_a), len(keys_b), 1),
                )

        self.cleaning_log.append(
            f"Cross-source validation: {len(report['comparisons'])} pair(s) checked"
        )
        return report

    # ------------------------------------------------------------------
    # 7. Data-quality scorecard
    # ------------------------------------------------------------------

    def build_quality_scorecard(self) -> Dict[str, Any]:
        """
        Produce a per-column and overall quality scorecard.

        Metrics per numeric column:
          completeness, uniqueness, validity (within physical bounds),
          mean, std, min, max, skew

        Overall score is the average of column-level scores.
        """
        logger.info("Building data-quality scorecard …")
        n_rows = len(self.df)
        column_scores: Dict[str, Dict[str, Any]] = {}

        for col in self.df.columns:
            if col.endswith("_outlier") or col == "outlier_score":
                continue

            completeness = 1.0 - (self.df[col].isnull().sum() / max(n_rows, 1))
            uniqueness = self.df[col].nunique() / max(n_rows, 1)

            entry: Dict[str, Any] = {
                "completeness": round(completeness, 4),
                "uniqueness": round(uniqueness, 4),
            }

            if pd.api.types.is_numeric_dtype(self.df[col]):
                entry.update({
                    "mean": round(float(self.df[col].mean()), 4),
                    "std": round(float(self.df[col].std()), 4),
                    "min": round(float(self.df[col].min()), 4),
                    "max": round(float(self.df[col].max()), 4),
                    "skew": round(float(self.df[col].skew()), 4),
                })

                # Validity: fraction within physical bounds
                if col in self.bounds:
                    lo, hi = self.bounds[col]
                    valid = ((self.df[col] >= lo) & (self.df[col] <= hi)).sum()
                    entry["validity"] = round(valid / max(n_rows, 1), 4)
                else:
                    entry["validity"] = 1.0
            else:
                entry["validity"] = 1.0  # categorical — always valid

            # Composite column score (simple weighted average)
            entry["score"] = round(
                0.5 * entry["completeness"]
                + 0.3 * entry["validity"]
                + 0.2 * min(entry["uniqueness"] * 10, 1.0),  # cap at 1
                4,
            )
            column_scores[col] = entry

        overall = (
            np.mean([v["score"] for v in column_scores.values()])
            if column_scores
            else 0.0
        )

        self.quality_scorecard = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_rows": n_rows,
            "n_columns": len(self.df.columns),
            "overall_quality_score": round(float(overall), 4),
            "columns": column_scores,
        }
        logger.info("Overall quality score: %.2f / 1.00", overall)
        return self.quality_scorecard

    # ------------------------------------------------------------------
    # 8. Reporting
    # ------------------------------------------------------------------

    def generate_cleaning_report(self) -> Dict[str, Any]:
        """
        Generate a structured cleaning report (dict).
        """
        report: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_shape": list(self.original_shape),
            "final_shape": list(self.df.shape),
            "rows_removed": self.original_shape[0] - self.df.shape[0],
            "columns_removed": self.original_shape[1] - self.df.shape[1],
            "pct_data_retained": round(
                len(self.df) / max(self.original_shape[0], 1) * 100, 2
            ),
            "cleaning_steps": self.cleaning_log,
            "remaining_missing": self.df.isnull().sum().to_dict(),
            "quality_scorecard": self.quality_scorecard or "not computed",
        }

        logger.info("═══ CLEANING REPORT ═══")
        for key, value in report.items():
            if key != "quality_scorecard":
                logger.info("  %s: %s", key, value)

        return report

    def save_report(
        self,
        report: Dict[str, Any],
        path: Union[str, Path] = "output/cleaning_report.json",
    ) -> Path:
        """Serialise the cleaning report to JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("Cleaning report saved → %s", out)
        return out

    # ------------------------------------------------------------------
    # 9. Pipeline orchestration
    # ------------------------------------------------------------------

    def get_cleaned_data(
        self,
        imputation_strategy: str = "smart",
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        run_cross_validation: bool = True,
        build_scorecard: bool = True,
    ) -> pd.DataFrame:
        """
        Execute the full cleaning pipeline and return the cleaned DataFrame.

        Parameters
        ----------
        imputation_strategy : str
            One of ``'smart'``, ``'knn'``, ``'forward_fill'``,
            ``'mean'``, ``'median'``, ``'drop'``.
        outlier_method : str
            One of ``'iqr'``, ``'zscore'``, ``'isolation_forest'``.
        outlier_threshold : float
            Sensitivity for IQR (default 1.5) or Z-score (default 3.0).
        run_cross_validation : bool
            Whether to run cross-source consistency checks.
        build_scorecard : bool
            Whether to build a quality scorecard at the end.
        """
        logger.info("╔═══ Starting data-cleaning pipeline ═══╗")

        self.data_type_validation()
        self.handle_missing_values(strategy=imputation_strategy)
        self.handle_duplicates()
        self.remove_invalid_values()
        self.handle_outliers(method=outlier_method, threshold=outlier_threshold)

        if run_cross_validation:
            self.cross_validate_sources()

        if build_scorecard:
            self.build_quality_scorecard()

        logger.info("╚═══ Data-cleaning pipeline complete ═══╝")
        return self.df


# ═══════════════════════════════════════════════════════════════════════════
# DataPreprocessor — feature scaling with metadata preservation
# ═══════════════════════════════════════════════════════════════════════════

class DataPreprocessor:
    """
    Feature scaling and preprocessing.

    Preserves scaler metadata (fitted params, column names) for
    reproducibility in production inference pipelines.
    """

    # Columns that should never be scaled
    NEVER_SCALE: List[str] = ["Year", "Yield"]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.scalers: Dict[str, Any] = {}
        self.scaling_metadata: Dict[str, Any] = {}

    def scale_features(self, method: str = "standard") -> pd.DataFrame:
        """
        Scale numeric features using *method*:

        * ``'standard'`` — Z-score (μ=0, σ=1)
        * ``'robust'``   — median/IQR-based (less sensitive to outliers)
        * ``'minmax'``   — scale to [0, 1]

        Stores fitted scaler and column names in ``self.scalers`` and
        ``self.scaling_metadata``.
        """
        logger.info("Scaling features: method=%s …", method)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_scale = [
            c for c in numeric_cols
            if c not in self.NEVER_SCALE and not c.endswith("_outlier")
            and c != "outlier_score"
        ]

        if not cols_to_scale:
            logger.warning("No numeric columns found to scale.")
            return self.df

        scaler: Union[StandardScaler, RobustScaler, MinMaxScaler]
        if method == "standard":
            scaler = StandardScaler()
        elif method == "robust":
            scaler = RobustScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method!r}")

        self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])
        self.scalers["features"] = scaler

        # Preserve metadata for reproducibility
        self.scaling_metadata = {
            "method": method,
            "scaler_class": type(scaler).__name__,
            "columns_scaled": cols_to_scale,
            "n_features": len(cols_to_scale),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if hasattr(scaler, "mean_"):
            self.scaling_metadata["feature_means"] = dict(
                zip(cols_to_scale, scaler.mean_.tolist())
            )
        if hasattr(scaler, "scale_"):
            self.scaling_metadata["feature_scales"] = dict(
                zip(cols_to_scale, scaler.scale_.tolist())
            )
        if hasattr(scaler, "center_"):
            self.scaling_metadata["feature_centers"] = dict(
                zip(cols_to_scale, scaler.center_.tolist())
            )
        if hasattr(scaler, "data_min_"):
            self.scaling_metadata["feature_mins"] = dict(
                zip(cols_to_scale, scaler.data_min_.tolist())
            )
        if hasattr(scaler, "data_max_"):
            self.scaling_metadata["feature_maxs"] = dict(
                zip(cols_to_scale, scaler.data_max_.tolist())
            )

        logger.info("Scaled %d columns: %s", len(cols_to_scale), cols_to_scale)
        return self.df

    def save_scalers(self, path: str = "models/scalers.pkl") -> Path:
        """Persist fitted scalers via joblib for production inference."""
        # pyrefly: ignore [missing-import]
        import joblib

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scalers": self.scalers,
            "metadata": self.scaling_metadata,
        }
        joblib.dump(payload, out)
        logger.info("Scalers + metadata saved → %s", out)
        return out

    def save_scaling_metadata(
        self, path: str = "models/scaling_metadata.json"
    ) -> Path:
        """Persist scaling metadata as human-readable JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self.scaling_metadata, fh, indent=2, default=str)
        logger.info("Scaling metadata saved → %s", out)
        return out


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    data_path = "data/raw/crop_data.csv"
    if not os.path.exists(data_path):
        logger.error("Raw crop data not found at %s", data_path)
        sys.exit(1)

    df = pd.read_csv(data_path)
    logger.info("Loaded raw data: %s", df.shape)

    # ── Clean ──
    cleaner = DataCleaner(df)
    df_cleaned = cleaner.get_cleaned_data(
        imputation_strategy="smart",
        outlier_method="iqr",
    )
    report = cleaner.generate_cleaning_report()
    cleaner.save_report(report)

    # ── Save cleaned data ──
    os.makedirs("data/processed", exist_ok=True)
    df_cleaned.to_csv("data/processed/cleaned_data.csv", index=False)
    logger.info("Cleaned data saved → data/processed/cleaned_data.csv")

    # ── Scale features ──
    preprocessor = DataPreprocessor(df_cleaned)
    df_processed = preprocessor.scale_features(method="standard")
    preprocessor.save_scalers()
    preprocessor.save_scaling_metadata()

    df_processed.to_csv("data/processed/processed_data.csv", index=False)
    logger.info("Processed data saved → data/processed/processed_data.csv")

    print("\n[OK] Phase 4 pipeline complete!")
    print(f"   Raw:       {df.shape}")
    print(f"   Cleaned:   {df_cleaned.shape}")
    print(f"   Processed: {df_processed.shape}")
    print(f"   Quality:   {report.get('pct_data_retained', '?')}% retained")
