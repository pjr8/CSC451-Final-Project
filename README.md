# CSC451 Final Project

Predict the **next 24 hourly temperatures** at Hartford Bradley International Airport from the **previous 14 days (336 hours)**. The model produces a full next-day temperature curve so a student can inspect the predicted daytime high, nighttime low, and hourly changes.

This expands the original three-hour, single-value task. It uses longer temporal context and a recurrent network to learn daily cycles and recent weather trends. Longer training alone is not evidence of better forecasts; compare the model with the baselines below.

The [measured full-data epoch](results/sequence/README.md) projects to **12.4 minutes for 30 epochs**, including validation and checkpoint writes, on the RTX 3070 Ti. A T4 measurement is still required; the full 30-epoch run has not yet been performed.

## Model and exact input

| Item | Specification |
|---|---|
| Station | Hartford Bradley, Connecticut; `USW00014740` |
| Input | 336 consecutive hourly temperatures in Celsius, oldest first |
| Input tensor | `float32`, shape `(batch_size, 336, 1)` |
| Target | 24 temperatures: `T(t+1)` through `T(t+24)` |
| Output tensor | Shape `(batch_size, 24)`; converted back to Celsius |
| Network | Three GRU layers with 256 hidden units, then `256 → 32 → 24` |
| Parameters | 997,432, including biases |
| Regularization | Dropout 0.1 in the output head; gradient clipping at norm 1.0 |
| Training | Adam, learning rate 0.001, MSE loss, batch size 32, seed 11 |
| Training cap | 30 epochs; stop after 5 epochs without better validation MAE |

```text
Input  = [T(t−335), T(t−334), ..., T(t−1), T(t)]
Target = [T(t+1), T(t+2), ..., T(t+24)]
```

A GRU is a recurrent neural network that carries a learned state through the input sequence. The final state feeds the output head to predict all 24 future values. GRUs are listed as an option in the assignment but are outside the covered course material, so the team must explain how this model works.

The planned app accepts `timestamp_utc,temperature_c` CSV data with exactly **336 unique, consecutive, hour-aligned UTC rows**, sorted oldest first. It does not require future targets. [Download a complete real input example](results/sequence/example_input.csv). For example, an input ending January 16 at 00:00 UTC predicts the 24 endpoints from January 16 at 01:00 through January 17 at 00:00 UTC.

Fit one mean and standard deviation on the selected training inputs only. Standardize inputs and training targets with those values; convert predictions back with `prediction_c = prediction_scaled × std + mean`. Reuse the saved statistics for validation, testing, and the app. Timestamps, quality codes, station IDs, humidity, pressure, and wind are not network features.

## Run the new pilot

[`sequence_pilot.py`](sequence_pilot.py) defaults to **one epoch on the earliest 10% of eligible training examples**. The complete NOAA data is included in the repository; no data API key or additional download is needed.

For an NVIDIA GPU with a compatible driver:

```bash
python -m pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python sequence_pilot.py --require-gpu
```

To check one full-data epoch, including validation:

```bash
python sequence_pilot.py --require-gpu --fraction 1 --epochs 1 --output-dir outputs/full-epoch
```

To train the full configuration:

```bash
python sequence_pilot.py --require-gpu --fraction 1 --epochs 30 --output-dir outputs/full-run
```

Use `--require-t4` in place of `--require-gpu` when measuring on a Colab T4. It rejects other GPUs and CPU runtimes. `--cpu` is available for a small local check. The 12–15 minute goal is a compute budget; validation stopping can finish a run earlier. No artificial delays are added.

### Google Colab

Choose a T4 runtime and run:

```python
!git clone https://github.com/pjr8/CSC451-Final-Project.git
%cd CSC451-Final-Project
!pip -q install -r requirements.txt
!python sequence_pilot.py --require-t4
```

Save output files before ending the Colab session, or use `--output-dir` with a mounted Google Drive directory.

### Outputs

- `sequence_report.json`: hardware, measured times, sample counts, settings, versions, and epoch history.
- `sequence_checkpoint.pt`: selected model weights, architecture, and scaling statistics.
- `example_input.csv`: the first real 336-hour input window.
- `example_targets.csv`: its 24 observed future values, for inspecting the training labels.

Results default to `outputs/sequence-pilot/`, which is excluded from Git. [Published benchmark results](results/sequence/README.md) distinguish the 10% pilot estimate from a measured full-data epoch. The checkpoint from one epoch is a pilot artifact, not the final trained model.

## Data preparation and leakage prevention

The [three original 2023–2025 station files](data/README.md) contain **40,052 raw observations**, compressed to about 3.1 MB. The script verifies checksums, rejects missing or unacceptable quality-flagged readings, and removes duplicate station/timestamp records.

For each hourly endpoint `h`, retain the last valid report in `(h−1 hour, h]`. Do not fill empty hours. A training example needs **all 336 input hours and all 24 target hours**; discard incomplete windows. No future temperature enters the input or the scaler.

| Split | Dates in UTC | Complete examples |
|---|---|---:|
| Training | January 2, 2023–December 29, 2024 | 15,306 |
| Validation | December 30, 2024–June 29, 2025 | 3,371 |
| Test | June 30–December 28, 2025 | 2,528 |

Build hourly bins and windows separately in each chronological partition. Discard windows crossing a boundary, keeping complete weeks together. The default 10% pilot uses the first **1,531** training examples. It does not use validation or test values for training or scaling. Full-data runs use validation MAE to select a checkpoint and stop training; the test set remains unused until the final evaluation.

## Evaluation plan

Compare the 24-hour forecast with:

1. Persistence: repeat the latest temperature for all 24 future hours.
2. Daily repetition: repeat the previous 24-hour temperature curve.
3. Linear or ridge regression: predict the same 24 targets from the same 336 input temperatures.

Report MAE and RMSE in Celsius, both overall and separately for each forecast hour. Include training/validation curves, 10–20 large-error cases, and discussion of rapid temperature changes. Run seven logged experiments changing one factor at a time, select settings by validation MAE, then train the chosen configuration with seeds 11, 22, and 33. Evaluate the final test set once and later collect at least 30 fresh forecast cases.

## Quota and original prototype

The original GPU quota request was made on **November 2, 2026**. A follow-up request was approved on **November 4**, increasing the global GPU limit to 1. T4 allocation attempts in three US regions failed due to capacity, so the published local measurements use the RTX 3070 Ti. They are not T4 measurements.

The original 24-input, three-hour MLP remains in [`pilot.py`](pilot.py), with its [historical benchmark](results/README.md). Its 0.81-minute estimate applies to that smaller task, not the current 24-hour recurrent forecast.

## What the data means

Each raw NOAA row is a weather observation from one station at one time. For this project, all observations come from Hartford Bradley International Airport. The recurrent model uses **temperature history only**. The other columns identify the station, organize the observations, or help check data quality.

| Raw field | Meaning | How we will use it |
|---|---|---|
| `STATION` | Station identifier; `USW00014740` is Hartford Bradley. | Keep records from the chosen station. Not a numeric model input. |
| `DATE` | Time of the observation, in UTC. | Sort readings, build hourly values, and align inputs with the future target. Not a numeric model input. |
| `temperature` | Air temperature near 2 metres above the ground, in degrees Celsius. `10.0` means **10.0°C**, not 1.0°C. | The measured value used to build the 336 input temperatures and the training target. |
| `temperature_Quality_Code` | NOAA quality-control result. Its meaning depends on the source. | Filter readings before training; not a temperature or a confidence percentage. |
| `temperature_Report_Type` | Weather-report format. `FM15` is a routine METAR, `FM16` is a special SPECI, and `FM12` is a fixed-land SYNOP report. | Preserve provenance; not a model input. |
| `temperature_Source_Code` | Identifies the upstream source of the observation. | Interpret quality codes correctly and preserve provenance; not a model input. |

In these samples, quality code `5` with source `343` means the value passed all quality checks and came from an NCEI source. Code `1` with source `223` means good quality. These definitions are source-specific; a larger code does not mean better quality. See the [NOAA documentation](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf), sections IV, VI, and VII.

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

These are the first 20 observations in the station's 2023 source file. Multiple observations can occur within an hour. They are raw source samples, not 20 complete forecasting examples. The scripts clean the observations and aggregate them into hourly values before building training examples.

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
