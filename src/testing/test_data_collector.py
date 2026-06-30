import os
import tempfile
import unittest

import pandas as pd

from src.data_processing.create_sample_data import (
    create_synthetic_climate_overlay,
    create_synthetic_dataset,
)
from src.data_processing.data_collector import DataCollector


class TestSyntheticDataGeneration(unittest.TestCase):
    def test_create_synthetic_dataset_has_required_columns(self):
        df = create_synthetic_dataset(n_samples=128, random_state=7)
        required = {
            "Country",
            "Crop",
            "Year",
            "Temperature",
            "Rainfall",
            "CO2_Emission",
            "Humidity",
            "Yield",
            "Production",
            "Area",
            "Extreme_Weather_Events",
            "Synthetic_Data_Used",
        }
        self.assertTrue(required.issubset(df.columns))
        self.assertEqual(len(df), 128)

    def test_create_synthetic_overlay_preserves_keys(self):
        keys = pd.DataFrame(
            {
                "Country": ["India", "Kenya"],
                "Crop": ["wheat", "maize"],
                "Year": [2012, 2015],
            }
        )
        overlay = create_synthetic_climate_overlay(keys, random_state=11)
        self.assertEqual(len(overlay), 2)
        self.assertEqual(
            overlay.loc[:, ["Country", "Crop", "Year"]].to_dict(orient="records"),
            keys.to_dict(orient="records"),
        )


class TestDataCollector(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_dir = os.path.join(self.temp_dir.name, "raw")
        self.processed_dir = os.path.join(self.temp_dir.name, "processed")
        self.cache_dir = os.path.join(self.temp_dir.name, "cache")
        self.metadata_dir = os.path.join(self.temp_dir.name, "metadata")
        self.collector = DataCollector(
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
            cache_dir=self.cache_dir,
            metadata_dir=self.metadata_dir,
            enable_remote_fetch=False,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_source_catalog_lists_five_sources(self):
        catalog = self.collector.list_data_sources()
        self.assertGreaterEqual(len(catalog), 5)
        self.assertIn("FAOSTAT", set(catalog["name"]))
        self.assertIn("NASA GISS GISTEMP", set(catalog["name"]))

    def test_fao_standardization_from_element_value_export(self):
        sample_fao = pd.DataFrame(
            [
                {"Area": "India", "Item": "Wheat", "Year": 2019, "Element": "Production", "Value": 1000},
                {"Area": "India", "Item": "Wheat", "Year": 2019, "Element": "Area harvested", "Value": 200},
                {"Area": "India", "Item": "Rice, paddy", "Year": 2019, "Element": "Production", "Value": 1400},
                {"Area": "India", "Item": "Rice, paddy", "Year": 2019, "Element": "Area harvested", "Value": 280},
            ]
        )
        os.makedirs(self.raw_dir, exist_ok=True)
        sample_fao.to_csv(os.path.join(self.raw_dir, "fao_crop_data.csv"), index=False)

        standardized = self.collector.collect_fao_data(crop_types=["wheat", "rice"])
        self.assertEqual(len(standardized), 2)
        wheat_row = standardized[standardized["Crop"] == "wheat"].iloc[0]
        self.assertEqual(wheat_row["Production"], 1000)
        self.assertEqual(wheat_row["Area"], 200)
        self.assertAlmostEqual(wheat_row["Yield"], 5.0)

    def test_merge_all_sources_offline_generates_outputs(self):
        integrated = self.collector.merge_all_sources(use_synthetic_fallback=True)
        self.assertFalse(integrated.empty)
        self.assertTrue(
            set(DataCollector.REQUIRED_MODEL_COLUMNS).issubset(integrated.columns)
        )
        self.assertTrue(os.path.exists(os.path.join(self.raw_dir, "integrated_data.csv")))
        self.assertTrue(
            os.path.exists(os.path.join(self.metadata_dir, "quality_report.json"))
        )


if __name__ == "__main__":
    unittest.main()
