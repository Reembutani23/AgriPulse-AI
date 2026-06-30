import logging
import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COUNTRY_PROFILES = {
    "India": {"base_temp": 24.0, "base_rain": 1080.0, "base_humidity": 72.0, "area_scale": 1.10, "adaptation": 0.45},
    "China": {"base_temp": 13.0, "base_rain": 645.0, "base_humidity": 66.0, "area_scale": 1.20, "adaptation": 0.60},
    "USA": {"base_temp": 12.5, "base_rain": 715.0, "base_humidity": 61.0, "area_scale": 1.30, "adaptation": 0.70},
    "Brazil": {"base_temp": 25.5, "base_rain": 1760.0, "base_humidity": 79.0, "area_scale": 1.15, "adaptation": 0.55},
    "Russia": {"base_temp": 3.0, "base_rain": 460.0, "base_humidity": 68.0, "area_scale": 0.95, "adaptation": 0.50},
    "Pakistan": {"base_temp": 22.0, "base_rain": 495.0, "base_humidity": 54.0, "area_scale": 0.88, "adaptation": 0.38},
    "Bangladesh": {"base_temp": 25.0, "base_rain": 2220.0, "base_humidity": 82.0, "area_scale": 0.72, "adaptation": 0.35},
    "Indonesia": {"base_temp": 26.0, "base_rain": 2600.0, "base_humidity": 84.0, "area_scale": 0.84, "adaptation": 0.48},
    "Nigeria": {"base_temp": 27.5, "base_rain": 1120.0, "base_humidity": 70.0, "area_scale": 0.90, "adaptation": 0.30},
    "Mexico": {"base_temp": 21.0, "base_rain": 760.0, "base_humidity": 62.0, "area_scale": 0.82, "adaptation": 0.52},
}

CROP_PROFILES = {
    "wheat": {"optimal_temp": 18.0, "optimal_rain": 520.0, "base_yield": 3.6, "base_area": 420000.0, "heat_sensitivity": 0.085},
    "rice": {"optimal_temp": 24.0, "optimal_rain": 1250.0, "base_yield": 4.8, "base_area": 380000.0, "heat_sensitivity": 0.065},
    "maize": {"optimal_temp": 21.0, "optimal_rain": 760.0, "base_yield": 5.3, "base_area": 360000.0, "heat_sensitivity": 0.080},
    "soybean": {"optimal_temp": 22.0, "optimal_rain": 690.0, "base_yield": 2.9, "base_area": 310000.0, "heat_sensitivity": 0.070},
}

DEFAULT_COUNTRY_PROFILE = {
    "base_temp": 19.5,
    "base_rain": 980.0,
    "base_humidity": 68.0,
    "area_scale": 0.95,
    "adaptation": 0.45,
}

DEFAULT_CROP_PROFILE = {
    "optimal_temp": 21.0,
    "optimal_rain": 800.0,
    "base_yield": 3.8,
    "base_area": 340000.0,
    "heat_sensitivity": 0.075,
}


def _build_country_crop_year_frame(years: Optional[Iterable[int]] = None) -> pd.DataFrame:
    """Create a full country-crop-year grid."""
    if years is None:
        years = range(2000, 2026)

    records = [
        (country, crop, int(year))
        for country in COUNTRY_PROFILES
        for crop in CROP_PROFILES
        for year in years
    ]
    return pd.DataFrame(records, columns=["Country", "Crop", "Year"])


def _clean_key_frame(keys_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"Country", "Crop", "Year"}
    missing = required_cols.difference(keys_df.columns)
    if missing:
        raise ValueError(f"Missing required key columns: {sorted(missing)}")

    cleaned = keys_df.loc[:, ["Country", "Crop", "Year"]].copy()
    cleaned["Country"] = cleaned["Country"].astype(str).str.strip()
    cleaned["Crop"] = cleaned["Crop"].astype(str).str.strip().str.lower()
    cleaned["Year"] = pd.to_numeric(cleaned["Year"], errors="coerce")
    cleaned = cleaned.dropna(subset=["Country", "Crop", "Year"])
    cleaned["Year"] = cleaned["Year"].astype(int)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned


def create_synthetic_climate_overlay(
    keys_df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create climate and production features for a supplied country-year-crop key grid.

    The output mirrors the production pipeline schema so it can be used as an
    offline fallback or as a standalone demo dataset.
    """
    df = _clean_key_frame(keys_df)
    rng = np.random.default_rng(random_state)

    country_meta = df["Country"].map(
        lambda country: COUNTRY_PROFILES.get(country, DEFAULT_COUNTRY_PROFILE)
    )
    crop_meta = df["Crop"].map(lambda crop: CROP_PROFILES.get(crop, DEFAULT_CROP_PROFILE))
    year_offset = df["Year"] - df["Year"].min()

    base_temp = country_meta.map(lambda meta: meta["base_temp"]).astype(float)
    base_rain = country_meta.map(lambda meta: meta["base_rain"]).astype(float)
    base_humidity = country_meta.map(lambda meta: meta["base_humidity"]).astype(float)
    area_scale = country_meta.map(lambda meta: meta["area_scale"]).astype(float)
    adaptation = country_meta.map(lambda meta: meta["adaptation"]).astype(float)

    optimal_temp = crop_meta.map(lambda meta: meta["optimal_temp"]).astype(float)
    optimal_rain = crop_meta.map(lambda meta: meta["optimal_rain"]).astype(float)
    base_yield = crop_meta.map(lambda meta: meta["base_yield"]).astype(float)
    base_area = crop_meta.map(lambda meta: meta["base_area"]).astype(float)
    heat_sensitivity = crop_meta.map(lambda meta: meta["heat_sensitivity"]).astype(float)

    climate_warming = 0.03 * year_offset
    temp_noise = rng.normal(0, 1.7, len(df))
    rain_noise = rng.normal(0, 110, len(df))
    humidity_noise = rng.normal(0, 5.5, len(df))
    area_noise = rng.lognormal(mean=0.0, sigma=0.25, size=len(df))

    temperature = base_temp + climate_warming + temp_noise
    rainfall = np.clip(base_rain - (1.6 * year_offset) + rain_noise, 80.0, None)
    co2_emission = 350.0 + (2.5 * year_offset) + rng.normal(0, 4.5, len(df))
    humidity = np.clip(
        base_humidity
        + humidity_noise
        - 0.18 * (temperature - base_temp)
        + 0.006 * (rainfall - base_rain),
        25.0,
        98.0,
    )

    heat_stress = np.abs(temperature - optimal_temp)
    rainfall_gap = np.abs(rainfall - optimal_rain) / np.maximum(optimal_rain, 1.0)
    tech_gain = 0.012 * year_offset

    yield_tonnes_per_ha = (
        base_yield
        * (1.0 + tech_gain + (0.10 * adaptation))
        * (1.0 - (heat_stress * heat_sensitivity))
        * (1.0 - (0.35 * rainfall_gap))
    )
    yield_tonnes_per_ha += rng.normal(0, 0.35, len(df))
    yield_tonnes_per_ha = np.clip(yield_tonnes_per_ha, 0.6, None)

    area = np.clip(base_area * area_scale * area_noise, 60000.0, None)
    production = area * yield_tonnes_per_ha

    stress_signal = (
        np.clip((heat_stress - 4.0) / 8.0, 0.0, None)
        + np.clip(rainfall_gap - 0.15, 0.0, None)
    )
    extreme_weather_events = rng.poisson(lam=np.clip(1.2 + stress_signal, 0.5, 5.5))

    result = df.copy()
    result["Temperature"] = np.round(temperature, 3)
    result["Rainfall"] = np.round(rainfall, 3)
    result["CO2_Emission"] = np.round(co2_emission, 3)
    result["Humidity"] = np.round(humidity, 3)
    result["Yield"] = np.round(yield_tonnes_per_ha, 3)
    result["Area"] = np.round(area, 2)
    result["Production"] = np.round(production, 2)
    result["Extreme_Weather_Events"] = extreme_weather_events.astype(int)
    result["Synthetic_Data_Used"] = True
    result["Data_Source"] = "Synthetic"

    return result


def create_synthetic_dataset(
    n_samples: int = 5000,
    random_state: int = 42,
    years: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """
    Create a production-compatible synthetic climate-agriculture dataset.

    Args:
        n_samples: Target number of rows. Samples are drawn from the full
            country-crop-year grid with replacement when necessary.
        random_state: Seed used for reproducible sampling and value generation.
        years: Optional iterable of years to include in the base grid.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    full_grid = _build_country_crop_year_frame(years=years)
    if n_samples >= len(full_grid):
        sampled = full_grid.sample(
            n=len(full_grid),
            replace=False,
            random_state=random_state,
        ).reset_index(drop=True)
        extra = full_grid.sample(
            n=n_samples - len(full_grid),
            replace=True,
            random_state=random_state + 7,
        ).reset_index(drop=True)
        sampled = pd.concat([sampled, extra], ignore_index=True)
    else:
        sampled = full_grid.sample(
            n=n_samples,
            replace=False,
            random_state=random_state,
        ).reset_index(drop=True)

    dataset = create_synthetic_climate_overlay(sampled, random_state=random_state)
    dataset = dataset.sort_values(["Year", "Country", "Crop"]).reset_index(drop=True)

    logger.info("Created synthetic dataset with shape %s", dataset.shape)
    return dataset


if __name__ == "__main__":
    df = create_synthetic_dataset(5000)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/crop_data.csv", index=False)
    print(f"Sample dataset created: {df.shape}")
    print(df.head())
