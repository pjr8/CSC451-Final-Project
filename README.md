# CSC451 Final Project

Local temperature forecasting with a small PyTorch neural network. The project will predict the temperature three hours ahead from the previous 24 hourly temperatures and compare it with persistence and linear regression.

## Raw data samples

The table below displays selected columns from **20 original NOAA GHCNh records** for **Hartford Bradley International Airport, Connecticut (USW00014740)**. The original file preserves all **329 columns**, including NOAA metadata and quality flags.

- [Full raw sample file](data/raw_samples_20.psv)
- [Download the raw sample file](https://raw.githubusercontent.com/pjr8/CSC451-Final-Project/main/data/raw_samples_20.psv)
- [Source details and file checksums](data/SOURCE.txt)

| Row | Timestamp (UTC) | Temperature (°C) | Quality code | Report type | Source code |
|---:|---|---:|---|---|---|
| 1 | 2023-01-01 00:34:00 | 10.0 | 5 | FM16 | 343 |
| 2 | 2023-01-01 00:51:00 | 10.0 | 5 | FM15 | 343 |
| 3 | 2023-01-01 00:59:00 | 10.0 | 5 | FM16 | 343 |
| 4 | 2023-01-01 01:20:00 | 10.0 | 5 | FM16 | 343 |
| 5 | 2023-01-01 01:34:00 | 10.0 | 5 | FM16 | 343 |
| 6 | 2023-01-01 01:51:00 | 9.4 | 5 | FM15 | 343 |
| 7 | 2023-01-01 02:51:00 | 9.4 | 5 | FM15 | 343 |
| 8 | 2023-01-01 03:46:00 | 9.0 | 5 | FM16 | 343 |
| 9 | 2023-01-01 03:51:00 | 9.4 | 5 | FM15 | 343 |
| 10 | 2023-01-01 04:01:00 | 10.0 | 5 | FM16 | 343 |
| 11 | 2023-01-01 04:48:00 | 9.0 | 5 | FM16 | 343 |
| 12 | 2023-01-01 04:51:00 | 10.0 | 5 | FM15 | 343 |
| 13 | 2023-01-01 05:05:00 | 10.0 | 5 | FM16 | 343 |
| 14 | 2023-01-01 05:12:00 | 10.0 | 5 | FM16 | 343 |
| 15 | 2023-01-01 05:45:00 | 10.0 | 5 | FM16 | 343 |
| 16 | 2023-01-01 05:51:00 | 10.0 | 5 | FM15 | 343 |
| 17 | 2023-01-01 06:00:00 | 10.0 | 1 | FM12 | 223 |
| 18 | 2023-01-01 06:44:00 | 10.0 | 5 | FM16 | 343 |
| 19 | 2023-01-01 06:51:00 | 10.0 | 5 | FM15 | 343 |
| 20 | 2023-01-01 07:17:00 | 10.6 | 5 | FM16 | 343 |

These are the first 20 observations in the station's 2023 source file. Multiple observations can occur within an hour. They are raw source samples, not 20 complete forecasting examples. The project will clean the observations and aggregate them into hourly inputs before building training examples.

## Reading the full samples

`data/raw_samples_20.psv` is a UTF-8 text file. A pipe (`|`) separates columns, the first line contains column names, and the next 20 lines contain the original records. Open it in a text editor or import it into a spreadsheet using `|` as the delimiter. Blank fields are preserved.

The preview above displays `DATE`, `temperature`, `temperature_Quality_Code`, `temperature_Report_Type`, and `temperature_Source_Code`. Temperature is in Celsius and timestamps are UTC. NOAA defines the quality, report, and source codes in its documentation.

## Source and license

- **Dataset:** NOAA Global Historical Climatology Network hourly (GHCNh).
- **Original file:** [GHCNh_USW00014740_2023.psv](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year/2023/psv/GHCNh_USW00014740_2023.psv).
- **Downloaded:** November 4, 2026.
- **Selection:** First 20 data records, with the complete original header. No values, flags, record order, or line endings were changed in the raw sample file.
- **Documentation:** [NOAA GHCNh documentation](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf).
- **Data license:** CC0-1.0, as stated in [NOAA's dataset metadata](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01688).
- **Dataset citation:** Menne et al. (2023), *Global Historical Climatology Network-Hourly (GHCNh)*, NOAA NCEI, [doi:10.25921/jp3d-3v19](https://doi.org/10.25921/jp3d-3v19). Subset: station USW00014740, first 20 records of 2023; accessed November 4, 2026.
