# Data Collection Pipeline

This project's Phase 2 pipeline is implemented in `src/data_processing/data_collector.py` and `src/data_processing/create_sample_data.py`. The collector is designed to support both real source exports and an offline synthetic fallback so the downstream modeling workflow remains runnable.

## Supported Sources

| Source | Purpose | Access Pattern | Expected Local File |
| --- | --- | --- | --- |
| FAOSTAT | Crop production, harvested area, yield | Manual CSV export from FAOSTAT browser | `data/raw/fao_crop_data.csv` |
| World Bank Indicators API | Population, GDP per capita, CO2 per capita, agricultural land | REST JSON or saved CSV | `data/raw/world_bank_context.csv` |
| NASA GISS GISTEMP | Global annual temperature anomalies | Direct CSV download or saved CSV | `data/raw/nasa_gistemp.csv` |
| NOAA GHCN-Monthly | Annualized precipitation | Prepared country-year CSV derived from NOAA exports | `data/raw/noaa_precipitation.csv` |
| USGS Water Services | Optional hydrology enrichment | Local CSV export | `data/raw/usgs_environmental_data.csv` |

## Integration Rules

- Base grain: `Country + Year + Crop`
- FAOSTAT is the primary production source.
- World Bank is merged on `Country + Year`.
- NASA is merged on `Year`.
- NOAA is merged on `Country + Year`.
- USGS is optional and merged on `Country + Year` when available.
- Missing model-critical features are filled with a transparent synthetic fallback.

## Generated Outputs

- `data/raw/integrated_data.csv`
- `data/raw/versions/integrated_data_<timestamp>.csv`
- `data/metadata/data_sources.json`
- `data/metadata/data_dictionary.json`
- `data/metadata/data_lineage.json`
- `data/metadata/quality_report.json`
- `data/metadata/schema_snapshot.csv`

## Update Workflow

1. Refresh the official source exports in `data/raw/`.
2. Run `python src/data_processing/data_collector.py`.
3. Review `data/metadata/quality_report.json`.
4. Confirm whether `Synthetic_Data_Used` appears only where expected.
5. Keep the timestamped snapshot in `data/raw/versions/` as the immutable dataset version for that run.

## Versioning Strategy

- Every integrated build writes a timestamped dataset snapshot.
- Metadata artifacts are copied into `data/metadata/versions/` with the same timestamp prefix.
- `data_lineage.json` records whether each source came from a local export, cache, remote API, or synthetic fallback.
- `Base_Data_Source` shows whether a row started from FAO or from the synthetic base dataset.

## Notes

- NOAA and USGS data are intentionally treated as optional local enrichments because their native formats are not already aligned to `Country + Year + Crop`.
- The synthetic generator is meant for development continuity, demos, and partial backfills. It should not silently replace a planned production feed without review.
