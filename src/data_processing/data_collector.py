from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
# pyrefly: ignore [missing-import]
from urllib3.util.retry import Retry

from src.data_processing.create_sample_data import (
    create_synthetic_climate_overlay,
    create_synthetic_dataset,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceDefinition:
    """Metadata for a supported upstream data source."""

    name: str
    description: str
    official_url: str
    access_url: str
    data_format: str
    coverage: str
    granularity: str
    merge_keys: Tuple[str, ...]
    local_candidates: Tuple[str, ...]
    notes: str = ""


class DataCollector:
    """
    Collect, standardize, validate, and integrate climate-agriculture datasets.

    The collector is designed to work in two modes:
    1. Production/local-export mode: users place official exports in `data/raw`.
    2. Development/demo mode: the pipeline falls back to synthetic data so the
       rest of the project can run without live network access.
    """

    DEFAULT_WORLD_BANK_INDICATORS = {
        "SP.POP.TOTL": "Population",
        "NY.GDP.PCAP.CD": "GDP_Per_Capita",
        "EN.ATM.CO2E.PC": "CO2_Emissions_Per_Capita",
        "AG.LND.AGRI.ZS": "Agricultural_Land_Pct",
    }

    REQUIRED_MODEL_COLUMNS = [
        "Country",
        "Year",
        "Crop",
        "Temperature",
        "Rainfall",
        "CO2_Emission",
        "Humidity",
        "Yield",
    ]

    FINAL_COLUMN_ORDER = [
        "Country",
        "Country_Code",
        "Year",
        "Crop",
        "Production",
        "Area",
        "Yield",
        "Temperature",
        "Rainfall",
        "CO2_Emission",
        "Humidity",
        "Extreme_Weather_Events",
        "Global_Temperature_Anomaly",
        "Population",
        "GDP_Per_Capita",
        "CO2_Emissions_Per_Capita",
        "Agricultural_Land_Pct",
        "USGS_Soil_Moisture",
        "USGS_Streamflow_Index",
        "Base_Data_Source",
        "Synthetic_Data_Used",
        "Data_Sources_Used",
    ]

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        cache_dir: str = "data/cache",
        metadata_dir: str = "data/metadata",
        enable_remote_fetch: bool = True,
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.cache_dir = Path(cache_dir)
        self.metadata_dir = Path(metadata_dir)
        self.enable_remote_fetch = enable_remote_fetch
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.cache_dir,
            self.metadata_dir,
            self.raw_dir / "versions",
            self.metadata_dir / "versions",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.session = self._build_retry_session()
        self.sources = self._build_source_catalog()
        self.lineage_log: List[Dict[str, object]] = []

    def _build_retry_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=self.max_retries,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(
            {"User-Agent": "AgriPulse-AI/1.0 (data integration pipeline)"}
        )
        return session

    def _build_source_catalog(self) -> Dict[str, SourceDefinition]:
        return {
            "fao": SourceDefinition(
                name="FAOSTAT",
                description="Crop production, harvested area, and yield exports.",
                official_url="https://www.fao.org/faostat/en/#data/QCL",
                access_url="https://www.fao.org/faostat/en/#data/QCL",
                data_format="Manual CSV export",
                coverage="1961-present, depending on commodity",
                granularity="Country-Year-Crop",
                merge_keys=("Country", "Year", "Crop"),
                local_candidates=(
                    "fao_crop_data.csv",
                    "faostat_crop_production.csv",
                    "fao_production.csv",
                ),
                notes="Best handled as an official browser export dropped into data/raw.",
            ),
            "world_bank": SourceDefinition(
                name="World Bank Indicators API",
                description="Country-year macro, emissions, and agricultural indicators.",
                official_url=(
                    "https://datahelpdesk.worldbank.org/knowledgebase/articles/"
                    "898581-api-basic-call-structures"
                ),
                access_url="https://api.worldbank.org/v2/country/all/indicator/{indicator}",
                data_format="REST JSON",
                coverage="1960-present, by indicator",
                granularity="Country-Year",
                merge_keys=("Country", "Year"),
                local_candidates=(
                    "world_bank_context.csv",
                    "world_bank_indicators.csv",
                ),
            ),
            "nasa": SourceDefinition(
                name="NASA GISS GISTEMP",
                description="Global annual temperature anomaly series.",
                official_url="https://data.giss.nasa.gov/gistemp/",
                access_url="https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
                data_format="CSV",
                coverage="1880-present",
                granularity="Year",
                merge_keys=("Year",),
                local_candidates=("nasa_gistemp.csv", "nasa_temperature.csv"),
            ),
            "noaa": SourceDefinition(
                name="NOAA GHCN-Monthly",
                description="Monthly precipitation observations that can be aggregated to country-year.",
                official_url=(
                    "https://www.ncei.noaa.gov/products/land-based-station/"
                    "global-historical-climatology-network-monthly"
                ),
                access_url="https://www.ncei.noaa.gov/data/ghcnm/v4/precipitation/access/",
                data_format="CSV or text export",
                coverage="1700s-present, station-dependent",
                granularity="Station-Month or Country-Year",
                merge_keys=("Country", "Year"),
                local_candidates=(
                    "noaa_precipitation.csv",
                    "ghcnm_precipitation.csv",
                    "country_year_precipitation.csv",
                ),
                notes=(
                    "For project use, store a prepared country-year precipitation extract in data/raw."
                ),
            ),
            "usgs": SourceDefinition(
                name="USGS Water Services",
                description="Optional hydrology and environmental enrichment.",
                official_url="https://waterservices.usgs.gov/",
                access_url="https://waterservices.usgs.gov/nwis",
                data_format="REST JSON/WaterML or local CSV export",
                coverage="US sites, time-series dependent",
                granularity="Site-Date or Region-Year",
                merge_keys=("Country", "Year"),
                local_candidates=(
                    "usgs_environmental_data.csv",
                    "usgs_water_data.csv",
                ),
                notes=(
                    "USGS is optional in the unified dataset because its native granularity is site-based."
                ),
            ),
        }

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _register_lineage(
        self,
        source_key: str,
        df: pd.DataFrame,
        *,
        status: str,
        detail: str,
        path: Optional[Path] = None,
    ) -> None:
        source = self.sources[source_key]
        lineage_entry = {
            "source_key": source_key,
            "source_name": source.name,
            "status": status,
            "detail": detail,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "coverage": {
                "countries": int(df["Country"].nunique()) if "Country" in df.columns else None,
                "crops": int(df["Crop"].nunique()) if "Crop" in df.columns else None,
                "year_min": int(df["Year"].min()) if "Year" in df.columns and not df.empty else None,
                "year_max": int(df["Year"].max()) if "Year" in df.columns and not df.empty else None,
            },
            "official_url": source.official_url,
            "access_url": source.access_url,
            "path": str(path) if path else None,
            "captured_at_utc": self._timestamp(),
        }
        self.lineage_log.append(lineage_entry)

    def _save_cache(self, filename: str, df: pd.DataFrame) -> None:
        cache_path = self.cache_dir / filename
        df.to_csv(cache_path, index=False)
        logger.info("Cached dataset at %s", cache_path)

    def _load_cache(self, filename: str) -> Optional[pd.DataFrame]:
        cache_path = self.cache_dir / filename
        if cache_path.exists():
            logger.info("Using cached dataset %s", cache_path)
            return pd.read_csv(cache_path)
        return None

    def _load_local_dataset(self, source_key: str) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
        for candidate in self.sources[source_key].local_candidates:
            path = self.raw_dir / candidate
            if path.exists():
                logger.info("Loading local %s export from %s", source_key, path)
                return pd.read_csv(path), path
        return None, None

    def _get_json(self, url: str, params: Optional[Dict[str, object]] = None) -> Optional[object]:
        if not self.enable_remote_fetch:
            return None
        try:
            response = self.session.get(url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning("Remote JSON fetch failed for %s: %s", url, exc)
            return None

    def _get_text(self, url: str) -> Optional[str]:
        if not self.enable_remote_fetch:
            return None
        try:
            response = self.session.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            logger.warning("Remote text fetch failed for %s: %s", url, exc)
            return None

    @staticmethod
    def _standardize_name(name: str) -> str:
        return (
            str(name)
            .strip()
            .replace("%", "pct")
            .replace("/", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
            .replace("__", "_")
            .lower()
        )

    def _find_first_column(
        self,
        df: pd.DataFrame,
        candidates: Sequence[str],
    ) -> Optional[str]:
        standardized = {self._standardize_name(column): column for column in df.columns}
        for candidate in candidates:
            actual = standardized.get(self._standardize_name(candidate))
            if actual:
                return actual
        return None

    @staticmethod
    def _as_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    @staticmethod
    def _normalize_country(series: pd.Series) -> pd.Series:
        replacements = {
            "United States": "USA",
            "United States of America": "USA",
            "Russian Federation": "Russia",
        }
        return series.astype(str).str.strip().replace(replacements)

    @staticmethod
    def _normalize_crop(series: pd.Series, crop_types: Optional[Iterable[str]]) -> pd.Series:
        crop_aliases = {
            "wheat": "wheat",
            "rice": "rice",
            "rice, paddy": "rice",
            "maize": "maize",
            "corn": "maize",
            "soybean": "soybean",
            "soybeans": "soybean",
        }
        normalized = series.astype(str).str.strip().str.lower()
        mapped = normalized.map(lambda value: crop_aliases.get(value, value))

        if crop_types:
            allowed = {item.lower() for item in crop_types}
            mapped = mapped.where(mapped.isin(allowed))

        return mapped

    def list_data_sources(self) -> pd.DataFrame:
        """Return the configured source catalog as a DataFrame."""
        return pd.DataFrame([asdict(source) for source in self.sources.values()])

    def collect_fao_data(self, crop_types: Optional[List[str]] = None) -> pd.DataFrame:
        logger.info("Collecting FAOSTAT crop production data...")
        local_df, local_path = self._load_local_dataset("fao")

        if local_df is None:
            empty = pd.DataFrame(columns=["Country", "Year", "Crop", "Production", "Area", "Yield"])
            self._register_lineage(
                "fao",
                empty,
                status="missing_local_export",
                detail="Expected a manually exported FAOSTAT CSV in data/raw.",
            )
            return empty

        standardized = self._standardize_fao_frame(local_df, crop_types=crop_types)
        self._register_lineage(
            "fao",
            standardized,
            status="local_file",
            detail="Loaded local FAOSTAT export.",
            path=local_path,
        )
        return standardized

    def _standardize_fao_frame(
        self,
        df: pd.DataFrame,
        crop_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        df = df.copy()
        country_col = self._find_first_column(df, ["Country", "Area", "Area Name"])
        crop_col = self._find_first_column(df, ["Crop", "Item", "Item Name"])
        year_col = self._find_first_column(df, ["Year"])
        value_col = self._find_first_column(df, ["Value"])
        element_col = self._find_first_column(df, ["Element", "Measure"])

        if not all([country_col, crop_col, year_col]):
            raise ValueError("FAO export is missing one of Country/Area, Crop/Item, or Year.")

        if value_col and element_col:
            index_cols = [country_col, crop_col, year_col]
            pivot = df.pivot_table(
                index=index_cols,
                columns=element_col,
                values=value_col,
                aggfunc="sum",
            ).reset_index()
            pivot.columns = [str(column) for column in pivot.columns]
            df = pivot

        rename_map = {
            country_col: "Country",
            crop_col: "Crop",
            year_col: "Year",
        }
        for column in df.columns:
            if column in rename_map:
                continue
            normalized = self._standardize_name(column)
            if normalized in {"production", "production_quantity"}:
                rename_map[column] = "Production"
            elif normalized in {"area_harvested", "harvested_area", "area"}:
                rename_map[column] = "Area"
            elif normalized.startswith("yield"):
                rename_map[column] = "Yield"

        standardized = df.rename(columns=rename_map)
        standardized["Country"] = self._normalize_country(standardized["Country"])
        standardized["Crop"] = self._normalize_crop(standardized["Crop"], crop_types)
        standardized["Year"] = self._as_numeric(standardized["Year"])
        standardized = standardized.dropna(subset=["Country", "Crop", "Year"])
        standardized["Year"] = standardized["Year"].astype(int)

        for column in ("Production", "Area", "Yield"):
            if column in standardized.columns:
                standardized[column] = self._as_numeric(standardized[column])

        if "Yield" not in standardized.columns and {"Production", "Area"}.issubset(standardized.columns):
            standardized["Yield"] = standardized["Production"] / standardized["Area"].replace({0: pd.NA})

        columns = ["Country", "Year", "Crop", "Production", "Area", "Yield"]
        standardized = standardized.reindex(columns=columns)
        standardized = standardized.dropna(subset=["Crop"])
        standardized = standardized.groupby(["Country", "Year", "Crop"], as_index=False).sum(numeric_only=True)
        return standardized.sort_values(["Year", "Country", "Crop"]).reset_index(drop=True)

    def collect_world_bank_context(
        self,
        indicators: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        logger.info("Collecting World Bank country-year indicators...")
        indicators = indicators or self.DEFAULT_WORLD_BANK_INDICATORS

        local_df, local_path = self._load_local_dataset("world_bank")
        if local_df is not None:
            standardized = self._standardize_world_bank_frame(local_df)
            self._register_lineage(
                "world_bank",
                standardized,
                status="local_file",
                detail="Loaded local World Bank indicator export.",
                path=local_path,
            )
            return standardized

        cached = self._load_cache("world_bank_context.csv")
        if cached is not None:
            standardized = self._standardize_world_bank_frame(cached)
            self._register_lineage(
                "world_bank",
                standardized,
                status="cache",
                detail="Loaded cached World Bank response.",
                path=self.cache_dir / "world_bank_context.csv",
            )
            return standardized

        frames = []
        for indicator, label in indicators.items():
            url = self.sources["world_bank"].access_url.format(indicator=indicator)
            payload = self._get_json(
                url,
                params={
                    "format": "json",
                    "per_page": 20000,
                    "source": 2,
                },
            )
            if not isinstance(payload, list) or len(payload) < 2:
                continue

            rows = []
            for item in payload[1]:
                if not isinstance(item, dict):
                    continue
                region = (item.get("region") or {}).get("value")
                if region == "Aggregates":
                    continue
                year = pd.to_numeric(item.get("date"), errors="coerce")
                rows.append(
                    {
                        "Country": (item.get("country") or {}).get("value"),
                        "Country_Code": item.get("countryiso3code"),
                        "Year": year,
                        label: item.get("value"),
                    }
                )

            indicator_df = pd.DataFrame(rows)
            if indicator_df.empty:
                continue
            indicator_df["Year"] = self._as_numeric(indicator_df["Year"])
            indicator_df[label] = self._as_numeric(indicator_df[label])
            indicator_df = indicator_df.dropna(subset=["Country", "Year"])
            indicator_df["Year"] = indicator_df["Year"].astype(int)
            frames.append(indicator_df)

        if not frames:
            empty = pd.DataFrame(columns=["Country", "Country_Code", "Year"])
            self._register_lineage(
                "world_bank",
                empty,
                status="unavailable",
                detail="No local export, cache, or live API response was available.",
            )
            return empty

        merged = frames[0]
        for next_frame in frames[1:]:
            merged = merged.merge(next_frame, on=["Country", "Country_Code", "Year"], how="outer")

        standardized = self._standardize_world_bank_frame(merged)
        self._save_cache("world_bank_context.csv", standardized)
        self._register_lineage(
            "world_bank",
            standardized,
            status="remote",
            detail="Downloaded from World Bank Indicators API.",
            path=self.cache_dir / "world_bank_context.csv",
        )
        return standardized

    def _standardize_world_bank_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rename_map = {}
        for column in df.columns:
            normalized = self._standardize_name(column)
            if normalized == "country":
                rename_map[column] = "Country"
            elif normalized in {"country_code", "countryiso3code"}:
                rename_map[column] = "Country_Code"
            elif normalized == "year":
                rename_map[column] = "Year"
        df = df.rename(columns=rename_map)

        if "Country" in df.columns:
            df["Country"] = self._normalize_country(df["Country"])
        if "Year" in df.columns:
            df["Year"] = self._as_numeric(df["Year"])
            df = df.dropna(subset=["Year"])
            df["Year"] = df["Year"].astype(int)

        for column in df.columns:
            if column not in {"Country", "Country_Code", "Year"}:
                df[column] = self._as_numeric(df[column])

        return df.sort_values(["Year", "Country"]).reset_index(drop=True)

    def collect_nasa_temperature(self) -> pd.DataFrame:
        logger.info("Collecting NASA GISS global temperature anomalies...")
        _, local_path = self._load_local_dataset("nasa")

        if local_path is not None:
            standardized = self._parse_nasa_gistemp_csv(local_path.read_text(encoding="utf-8"))
            self._register_lineage(
                "nasa",
                standardized,
                status="local_file",
                detail="Loaded local NASA GISS export.",
                path=local_path,
            )
            return standardized

        cached = self._load_cache("nasa_temperature.csv")
        if cached is not None:
            standardized = self._standardize_nasa_frame(cached)
            self._register_lineage(
                "nasa",
                standardized,
                status="cache",
                detail="Loaded cached NASA GISS response.",
                path=self.cache_dir / "nasa_temperature.csv",
            )
            return standardized

        text = self._get_text(self.sources["nasa"].access_url)
        if not text:
            empty = pd.DataFrame(columns=["Year", "Global_Temperature_Anomaly"])
            self._register_lineage(
                "nasa",
                empty,
                status="unavailable",
                detail="No local export, cache, or live NASA download was available.",
            )
            return empty

        standardized = self._parse_nasa_gistemp_csv(text)
        self._save_cache("nasa_temperature.csv", standardized)
        self._register_lineage(
            "nasa",
            standardized,
            status="remote",
            detail="Downloaded from NASA GISS GISTEMP.",
            path=self.cache_dir / "nasa_temperature.csv",
        )
        return standardized

    def _standardize_nasa_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if {"Year", "Global_Temperature_Anomaly"}.issubset(df.columns):
            standardized = df.loc[:, ["Year", "Global_Temperature_Anomaly"]].copy()
            standardized["Year"] = self._as_numeric(standardized["Year"])
            standardized["Global_Temperature_Anomaly"] = self._as_numeric(
                standardized["Global_Temperature_Anomaly"]
            )
            standardized = standardized.dropna().reset_index(drop=True)
            standardized["Year"] = standardized["Year"].astype(int)
            return standardized.sort_values("Year").reset_index(drop=True)
        return self._parse_nasa_gistemp_csv(df.to_csv(index=False))

    def _parse_nasa_gistemp_csv(self, text: str) -> pd.DataFrame:
        lines = [line for line in text.splitlines() if line.strip()]
        header_idx = next(
            (
                idx
                for idx, line in enumerate(lines)
                if line.lower().startswith("year")
            ),
            None,
        )
        if header_idx is None:
            raise ValueError("Could not locate the NASA GISS header row.")

        csv_block = "\n".join(lines[header_idx:])
        df = pd.read_csv(StringIO(csv_block))
        df.columns = [str(column).strip() for column in df.columns]

        annual_col = None
        for candidate in ("J-D", "J-D ", "D-N", "Annual", "Annual Mean"):
            if candidate in df.columns:
                annual_col = candidate
                break

        if annual_col is None:
            monthly_cols = [column for column in df.columns if column[:3].title() in {
                "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
            }]
            if not monthly_cols:
                raise ValueError("NASA GISS file did not contain an annual or monthly anomaly column.")
            for column in monthly_cols:
                df[column] = self._as_numeric(df[column].replace("***", pd.NA))
            df["Global_Temperature_Anomaly"] = df[monthly_cols].mean(axis=1)
        else:
            df["Global_Temperature_Anomaly"] = self._as_numeric(
                df[annual_col].replace("***", pd.NA)
            )

        df["Year"] = self._as_numeric(df["Year"])
        standardized = df.loc[:, ["Year", "Global_Temperature_Anomaly"]].dropna()
        standardized["Year"] = standardized["Year"].astype(int)
        return standardized.sort_values("Year").reset_index(drop=True)

    def collect_noaa_precipitation(self) -> pd.DataFrame:
        logger.info("Collecting NOAA precipitation data...")
        local_df, local_path = self._load_local_dataset("noaa")

        if local_df is None:
            empty = pd.DataFrame(columns=["Country", "Year", "Rainfall"])
            self._register_lineage(
                "noaa",
                empty,
                status="missing_local_export",
                detail="Expected a prepared NOAA country-year precipitation extract in data/raw.",
            )
            return empty

        standardized = self._standardize_noaa_frame(local_df)
        self._register_lineage(
            "noaa",
            standardized,
            status="local_file",
            detail="Loaded local NOAA precipitation export.",
            path=local_path,
        )
        return standardized

    def _standardize_noaa_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rename_map = {}
        for column in df.columns:
            normalized = self._standardize_name(column)
            if normalized == "country":
                rename_map[column] = "Country"
            elif normalized in {"country_code", "iso3"}:
                rename_map[column] = "Country_Code"
            elif normalized == "year":
                rename_map[column] = "Year"
            elif normalized in {"rainfall", "precipitation", "precipitation_mm"}:
                rename_map[column] = "Rainfall"
        df = df.rename(columns=rename_map)

        if "Rainfall" not in df.columns:
            monthly_cols = [column for column in df.columns if self._standardize_name(column) in {
                "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
            }]
            if monthly_cols:
                for column in monthly_cols:
                    df[column] = self._as_numeric(df[column])
                df["Rainfall"] = df[monthly_cols].sum(axis=1, min_count=1)

        if not {"Country", "Year", "Rainfall"}.issubset(df.columns):
            raise ValueError("NOAA export must contain Country, Year, and Rainfall-compatible columns.")

        df["Country"] = self._normalize_country(df["Country"])
        df["Year"] = self._as_numeric(df["Year"])
        df["Rainfall"] = self._as_numeric(df["Rainfall"])
        df = df.dropna(subset=["Country", "Year", "Rainfall"])
        df["Year"] = df["Year"].astype(int)

        if "Country_Code" in df.columns:
            grouped = df.groupby(["Country", "Country_Code", "Year"], as_index=False)["Rainfall"].mean()
        else:
            grouped = df.groupby(["Country", "Year"], as_index=False)["Rainfall"].mean()
        return grouped.sort_values(["Year", "Country"]).reset_index(drop=True)

    def collect_usgs_environmental(self) -> pd.DataFrame:
        logger.info("Collecting optional USGS environmental data...")
        local_df, local_path = self._load_local_dataset("usgs")

        if local_df is None:
            empty = pd.DataFrame(columns=["Country", "Year", "USGS_Soil_Moisture", "USGS_Streamflow_Index"])
            self._register_lineage(
                "usgs",
                empty,
                status="optional_missing",
                detail="USGS enrichment is optional and no local export was found.",
            )
            return empty

        standardized = self._standardize_usgs_frame(local_df)
        self._register_lineage(
            "usgs",
            standardized,
            status="local_file",
            detail="Loaded local USGS environmental export.",
            path=local_path,
        )
        return standardized

    def _standardize_usgs_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rename_map = {}
        for column in df.columns:
            normalized = self._standardize_name(column)
            if normalized == "country":
                rename_map[column] = "Country"
            elif normalized in {"country_code", "iso3"}:
                rename_map[column] = "Country_Code"
            elif normalized == "year":
                rename_map[column] = "Year"
            elif normalized in {"soil_moisture", "usgs_soil_moisture"}:
                rename_map[column] = "USGS_Soil_Moisture"
            elif normalized in {"streamflow_index", "streamflow", "usgs_streamflow_index"}:
                rename_map[column] = "USGS_Streamflow_Index"
        df = df.rename(columns=rename_map)

        if not {"Country", "Year"}.issubset(df.columns):
            raise ValueError("USGS export must contain Country and Year fields.")

        df["Country"] = self._normalize_country(df["Country"])
        df["Year"] = self._as_numeric(df["Year"])
        df = df.dropna(subset=["Country", "Year"])
        df["Year"] = df["Year"].astype(int)

        for column in ("USGS_Soil_Moisture", "USGS_Streamflow_Index"):
            if column in df.columns:
                df[column] = self._as_numeric(df[column])

        keep_cols = [column for column in [
            "Country",
            "Country_Code",
            "Year",
            "USGS_Soil_Moisture",
            "USGS_Streamflow_Index",
        ] if column in df.columns]

        grouped_cols = [column for column in keep_cols if column not in {"Country", "Country_Code", "Year"}]
        grouping_keys = [column for column in ("Country", "Country_Code", "Year") if column in keep_cols]
        if not grouped_cols:
            grouped = df.loc[:, grouping_keys].drop_duplicates().reset_index(drop=True)
        else:
            grouped = df.loc[:, keep_cols].groupby(grouping_keys, as_index=False)[grouped_cols].mean()
        return grouped.sort_values(["Year", "Country"]).reset_index(drop=True)

    def _ensure_required_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required_keys = df.loc[:, ["Country", "Crop", "Year"]].drop_duplicates()
        overlay = create_synthetic_climate_overlay(required_keys)
        merged = df.merge(
            overlay,
            on=["Country", "Crop", "Year"],
            how="left",
            suffixes=("", "_synthetic"),
        )

        synthetic_columns = [
            "Temperature",
            "Rainfall",
            "CO2_Emission",
            "Humidity",
            "Yield",
            "Area",
            "Production",
            "Extreme_Weather_Events",
        ]
        needed_mask = pd.Series(False, index=merged.index)
        for column in synthetic_columns:
            synthetic_col = f"{column}_synthetic"
            if column not in merged.columns:
                needed_mask = needed_mask | merged[synthetic_col].notna()
            else:
                needed_mask = needed_mask | merged[column].isna()
            if column not in merged.columns:
                merged[column] = merged[synthetic_col]
            else:
                merged[column] = merged[column].fillna(merged[synthetic_col])

        if "Synthetic_Data_Used" not in merged.columns:
            merged["Synthetic_Data_Used"] = False
        merged["Synthetic_Data_Used"] = (
            merged["Synthetic_Data_Used"].fillna(False).astype(bool)
            | needed_mask
        )

        drop_cols = [column for column in merged.columns if column.endswith("_synthetic")]
        return merged.drop(columns=drop_cols)

    def _build_data_sources_used(self, df: pd.DataFrame) -> pd.Series:
        source_labels = []
        for _, row in df.iterrows():
            labels = []
            if row.get("Base_Data_Source") == "FAO":
                labels.append("FAO")
            if not pd.isna(row.get("Population")) or not pd.isna(row.get("GDP_Per_Capita")):
                labels.append("WorldBank")
            if not pd.isna(row.get("Global_Temperature_Anomaly")):
                labels.append("NASA")
            if not pd.isna(row.get("Rainfall")):
                labels.append("NOAA/Synthetic")
            if not pd.isna(row.get("USGS_Soil_Moisture")) or not pd.isna(row.get("USGS_Streamflow_Index")):
                labels.append("USGS")
            if bool(row.get("Synthetic_Data_Used")):
                labels.append("Synthetic")
            source_labels.append("|".join(sorted(set(labels))) if labels else "Unknown")
        return pd.Series(source_labels, index=df.index)

    def validate_integrated_dataset(self, df: pd.DataFrame) -> Dict[str, object]:
        logger.info("Validating integrated dataset...")
        duplicate_keys = 0
        if {"Country", "Year", "Crop"}.issubset(df.columns):
            duplicate_keys = int(df.duplicated(subset=["Country", "Year", "Crop"]).sum())

        if df.empty:
            missing_rates = {column: 100.0 for column in df.columns}
            completeness = 0.0
            uniqueness = 100.0
            valid_years = 0.0
        else:
            missing_rates = {
                column: round(float(df[column].isna().mean() * 100), 2)
                for column in df.columns
            }
            completeness = 100.0 - (
                sum(df[column].isna().mean() for column in self.REQUIRED_MODEL_COLUMNS if column in df.columns)
                / len(self.REQUIRED_MODEL_COLUMNS)
                * 100.0
            )
            uniqueness = (1.0 - (duplicate_keys / len(df))) * 100.0
            valid_years = (
                df["Year"].between(1880, datetime.now().year).mean() * 100.0
                if "Year" in df.columns
                else 0.0
            )
        quality_score = round((completeness + uniqueness + valid_years) / 3.0, 2)

        report = {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "countries": int(df["Country"].nunique()) if "Country" in df.columns else 0,
            "crops": int(df["Crop"].nunique()) if "Crop" in df.columns else 0,
            "year_min": int(df["Year"].min()) if "Year" in df.columns and not df.empty else None,
            "year_max": int(df["Year"].max()) if "Year" in df.columns and not df.empty else None,
            "duplicate_country_year_crop_rows": duplicate_keys,
            "missing_percent_by_column": missing_rates,
            "completeness_score": round(completeness, 2),
            "uniqueness_score": round(uniqueness, 2),
            "temporal_validity_score": round(valid_years, 2),
            "quality_score": quality_score,
        }
        return report

    def _data_dictionary(self) -> List[Dict[str, str]]:
        return [
            {"column": "Country", "type": "string", "description": "Standardized country name used as a merge key."},
            {"column": "Country_Code", "type": "string", "description": "Optional ISO-like country code from supporting sources."},
            {"column": "Year", "type": "integer", "description": "Observation year."},
            {"column": "Crop", "type": "string", "description": "Canonical crop label (wheat, rice, maize, soybean)."},
            {"column": "Production", "type": "float", "description": "Crop production volume, typically tonnes."},
            {"column": "Area", "type": "float", "description": "Harvested area, typically hectares."},
            {"column": "Yield", "type": "float", "description": "Crop yield per hectare."},
            {"column": "Temperature", "type": "float", "description": "Average yearly temperature in degrees Celsius."},
            {"column": "Rainfall", "type": "float", "description": "Annual rainfall or precipitation total in millimeters."},
            {"column": "CO2_Emission", "type": "float", "description": "Pipeline-ready CO2 feature used for modeling."},
            {"column": "Humidity", "type": "float", "description": "Average yearly humidity percentage."},
            {"column": "Extreme_Weather_Events", "type": "integer", "description": "Count-like indicator for extreme events."},
            {"column": "Global_Temperature_Anomaly", "type": "float", "description": "NASA GISS global annual anomaly series."},
            {"column": "Population", "type": "float", "description": "World Bank total population indicator."},
            {"column": "GDP_Per_Capita", "type": "float", "description": "World Bank GDP per capita, current US dollars."},
            {"column": "CO2_Emissions_Per_Capita", "type": "float", "description": "World Bank CO2 emissions metric."},
            {"column": "Agricultural_Land_Pct", "type": "float", "description": "World Bank agricultural land share."},
            {"column": "USGS_Soil_Moisture", "type": "float", "description": "Optional USGS enrichment metric."},
            {"column": "USGS_Streamflow_Index", "type": "float", "description": "Optional USGS hydrology enrichment metric."},
            {"column": "Base_Data_Source", "type": "string", "description": "Primary row origin before climate/context enrichment."},
            {"column": "Synthetic_Data_Used", "type": "boolean", "description": "True when the row required synthetic fallback values."},
            {"column": "Data_Sources_Used", "type": "string", "description": "Pipe-delimited list of source systems contributing to the row."},
        ]

    def _write_metadata_artifacts(
        self,
        final_df: pd.DataFrame,
        quality_report: Dict[str, object],
        version_tag: str,
    ) -> None:
        catalog_path = self.metadata_dir / "data_sources.json"
        dictionary_path = self.metadata_dir / "data_dictionary.json"
        lineage_path = self.metadata_dir / "data_lineage.json"
        quality_path = self.metadata_dir / "quality_report.json"

        with catalog_path.open("w", encoding="utf-8") as handle:
            json.dump([asdict(source) for source in self.sources.values()], handle, indent=2)
        with dictionary_path.open("w", encoding="utf-8") as handle:
            json.dump(self._data_dictionary(), handle, indent=2)
        with lineage_path.open("w", encoding="utf-8") as handle:
            json.dump(self.lineage_log, handle, indent=2)
        with quality_path.open("w", encoding="utf-8") as handle:
            json.dump(quality_report, handle, indent=2)

        version_dir = self.metadata_dir / "versions"
        for filename in ("data_sources.json", "data_dictionary.json", "data_lineage.json", "quality_report.json"):
            source_path = self.metadata_dir / filename
            target_path = version_dir / f"{version_tag}_{filename}"
            target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")

        schema_snapshot = self.metadata_dir / "schema_snapshot.csv"
        pd.DataFrame(
            {"column": final_df.columns, "dtype": [str(dtype) for dtype in final_df.dtypes]}
        ).to_csv(schema_snapshot, index=False)

    def merge_all_sources(
        self,
        crop_types: Optional[List[str]] = None,
        use_synthetic_fallback: bool = True,
    ) -> pd.DataFrame:
        logger.info("Merging all configured sources...")
        self.lineage_log = []

        fao_data = self.collect_fao_data(crop_types=crop_types or ["wheat", "rice", "maize", "soybean"])
        world_bank = self.collect_world_bank_context()
        nasa = self.collect_nasa_temperature()
        noaa = self.collect_noaa_precipitation()
        usgs = self.collect_usgs_environmental()

        if fao_data.empty and use_synthetic_fallback:
            logger.info("No FAO export found. Falling back to synthetic base dataset.")
            merged = create_synthetic_dataset(5000)
            merged["Base_Data_Source"] = "Synthetic"
        else:
            merged = fao_data.copy()
            merged["Base_Data_Source"] = "FAO"

        if not world_bank.empty:
            join_keys = ["Country", "Year"]
            if "Country_Code" in world_bank.columns and "Country_Code" in merged.columns:
                join_keys = ["Country", "Country_Code", "Year"]
            merged = merged.merge(world_bank, on=join_keys, how="left")

        if not nasa.empty:
            merged = merged.merge(nasa, on=["Year"], how="left")

        if not noaa.empty:
            join_keys = ["Country", "Year"]
            noaa_join = [key for key in join_keys if key in noaa.columns]
            merged = merged.merge(noaa, on=noaa_join, how="left", suffixes=("", "_noaa"))
            if "Rainfall_noaa" in merged.columns:
                if "Rainfall" in merged.columns:
                    merged["Rainfall"] = merged["Rainfall"].fillna(merged["Rainfall_noaa"])
                else:
                    merged["Rainfall"] = merged["Rainfall_noaa"]
                merged = merged.drop(columns=["Rainfall_noaa"])

        if not usgs.empty:
            join_keys = [key for key in ("Country", "Year") if key in usgs.columns]
            merged = merged.merge(usgs, on=join_keys, how="left")

        if use_synthetic_fallback:
            merged = self._ensure_required_features(merged)

        if "Yield" not in merged.columns and {"Production", "Area"}.issubset(merged.columns):
            merged["Yield"] = merged["Production"] / merged["Area"].replace({0: pd.NA})

        if "Country_Code" not in merged.columns:
            merged["Country_Code"] = pd.NA

        merged["Data_Sources_Used"] = self._build_data_sources_used(merged)
        merged = merged.drop_duplicates(subset=["Country", "Year", "Crop"], keep="first")
        merged = merged.sort_values(["Year", "Country", "Crop"]).reset_index(drop=True)

        for column in self.FINAL_COLUMN_ORDER:
            if column not in merged.columns:
                merged[column] = pd.NA

        final_df = merged.loc[:, self.FINAL_COLUMN_ORDER]
        quality_report = self.validate_integrated_dataset(final_df)
        version_tag = self._timestamp()

        integrated_path = self.raw_dir / "integrated_data.csv"
        versioned_path = self.raw_dir / "versions" / f"integrated_data_{version_tag}.csv"
        final_df.to_csv(integrated_path, index=False)
        final_df.to_csv(versioned_path, index=False)
        logger.info("Saved integrated dataset to %s", integrated_path)
        logger.info("Saved versioned dataset snapshot to %s", versioned_path)

        self._write_metadata_artifacts(final_df, quality_report, version_tag)
        return final_df


if __name__ == "__main__":
    collector = DataCollector()
    integrated = collector.merge_all_sources()
    print(f"Data collection complete. Shape: {integrated.shape}")
