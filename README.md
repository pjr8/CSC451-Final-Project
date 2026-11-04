# CSC451 Final Project

Local temperature forecasting with a small PyTorch neural network. The project will predict the temperature three hours ahead from the previous 24 hourly temperatures and compare it with persistence and linear regression.

## Run the early pilot

[`pilot.py`](pilot.py) trains the `24 → 64 → 32 → 1` MLP for **one epoch on the earliest 10% of complete training examples**, rounded up. It uses Adam, learning rate `0.001`, batch size `64`, and seed `11`. The remaining training examples are counted for the runtime estimate, but are not used to fit this pilot. Scaling is fitted only on the selected pilot inputs.

The complete 2023–2025 station data is included as [compressed raw files](data/README.md), about 3.1 MB total. The script reads and verifies these files; no NOAA download or API key is needed.

With the included data and cleaning rules, there are **17,312 training examples**, **4,287 validation examples**, and **4,070 test examples**. The default pilot trains on the first **1,732** training examples. Raw records and forecasting examples are different counts: one example needs 24 hourly inputs plus a later target.

[Measured early-pilot results and saved artifacts](results/README.md): the RTX 3070 Ti run took **0.243 seconds for one epoch**, estimating **0.81 minutes for 20 full-data epochs**. This is an RTX measurement, not a T4 measurement.

### Google Colab T4

1. Open a Colab notebook and choose **Runtime → Change runtime type → T4 GPU**.
2. Run this cell:

```python
!git clone https://github.com/pjr8/CSC451-Final-Project.git
%cd CSC451-Final-Project
!pip -q install -r requirements.txt
```

3. Save the pilot results to Google Drive so they survive a Colab disconnect:

```python
from google.colab import drive
drive.mount('/content/drive')
!python pilot.py --require-t4 --output-dir '/content/drive/MyDrive/CSC451/pilot'
```

`--require-t4` stops with an error if the selected runtime is not a T4. The report records the actual GPU name. GPU work is synchronized before and after timing.

### Local NVIDIA GPU

For a compatible NVIDIA GPU and driver, install the CUDA build and run:

```bash
python -m pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python pilot.py --require-gpu
```

`--require-gpu` prevents an accidental CPU measurement. The report records the detected GPU model.

### Local CPU check

From the repository directory, with Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python pilot.py --cpu
```

A CPU run checks the script, but does not satisfy the assignment's T4 timing requirement.

### Output files

| File | Contents |
|---|---|
| `pilot_report.json` | Actual device, sample counts, one-epoch time, 20-epoch runtime estimate, cleaning counts, partition counts, settings, and package versions. |
| `pilot_checkpoint.pt` | Weights after one pilot epoch plus the input scaling statistics. This is not the final trained model. |
| `example_input.csv` | The first real pilot example's 24 hourly inputs, in the app's CSV format. |
| `example_target.json` | That example's observed temperature three hours later and its target timestamp. |

Default outputs go to `outputs/pilot/`, which is excluded from Git. Preparation is timed separately. The estimate is `epoch seconds × full training examples / pilot examples × 20 / 60`. It estimates training time only; confirm it with a full-data epoch before committing to a complete run. A training-loss number from this pilot is not a validation or test score.

The partitions are training **January 2, 2023–December 29, 2024**, validation **December 30, 2024–June 29, 2025**, and test **June 30–December 28, 2025**. This script counts validation/test examples but does not train on them, calculate their scores, or tune using them. See the [quality and hourly-binning policy](data/README.md).

Run the small data-alignment checks with:

```bash
python -m unittest discover -s tests -v
```

## What the data means

Each raw NOAA row is a weather observation from one station at one time. For this project, all observations come from Hartford Bradley International Airport. The first model will use **temperature history only**. The other columns identify the station, organize the observations, or help check data quality.

| Raw field | Meaning | How we will use it |
|---|---|---|
| `STATION` | Station identifier; `USW00014740` is Hartford Bradley. | Keep records from the chosen station. Not a numeric model input. |
| `DATE` | Time of the observation, in UTC. | Sort readings, build hourly values, and align inputs with the future target. Not a numeric model input. |
| `temperature` | Air temperature near 2 metres above the ground, in degrees Celsius. `10.0` means **10.0°C**, not 1.0°C. | The measured value used to build the 24 input temperatures and the training target. |
| `temperature_Quality_Code` | NOAA quality-control result. Its meaning depends on the source. | Filter readings before training; not a temperature or a confidence percentage. |
| `temperature_Report_Type` | Weather-report format. `FM15` is a routine METAR, `FM16` is a special SPECI, and `FM12` is a fixed-land SYNOP report. | Preserve provenance; not a model input. |
| `temperature_Source_Code` | Identifies the upstream source of the observation. | Interpret quality codes correctly and preserve provenance; not a model input. |

In these samples, quality code `5` with source `343` means the value passed all quality checks and came from an NCEI source. Code `1` with source `223` means good quality. These definitions are source-specific; a larger code does not mean better quality. See the [NOAA documentation](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf), sections IV, VI, and VII.

## From raw observations to hourly temperatures

The training-data plan uses the station's 2023–2025 files. The 20 rows below are source examples, not the complete training dataset.

1. Read the timestamps as UTC and temperatures as numeric Celsius values. Reject missing, invalid, or unacceptable quality-flagged readings using NOAA's source-specific definitions, and remove duplicate station/time records.
2. Group valid readings into hours ending at an exact UTC hour. For an endpoint `h`, choose the **last valid observation in `(h − 1 hour, h]`**. A report exactly at `h` belongs to that endpoint. No later report can fill an earlier hour.
3. Label that hourly value `T(h)`. If the hour has no valid observation, leave it missing; do not interpolate or carry an older value forward.
4. Form examples only when all 24 input hours and the three-hour-ahead target are available. Log how many records and examples each cleaning step removes.

For example, among the displayed samples, the last observation before **01:00 UTC** is the **00:59** reading, so `T(01:00) = 10.0°C`. The last observation before **02:00** is the **01:51** reading, so `T(02:00) = 9.4°C`.

## Exact model inputs and output

Let `t` be the latest completed hourly endpoint. One input is **24 temperatures ordered from oldest to newest**:

```text
Input X(t) = [T(t−23), T(t−22), ..., T(t−1), T(t)]
Target y(t) = T(t+3)
```

| Item | Exact definition |
|---|---|
| Number of input features | 24 numeric Celsius temperatures, one per consecutive hour. |
| Forecast horizon | Three hours after the latest input endpoint. |
| Model | PyTorch MLP: `24 → 64 → 32 → 1`, with ReLU hidden layers and a linear output. |
| Input tensor | `float32`, shape `(batch_size, 24)`; one forecast uses `(1, 24)`. |
| Training target | The observed hourly temperature at `t+3`, shape `(batch_size, 1)`, in Celsius. |
| Prediction | One estimated temperature in Celsius. Negative values are allowed. |

Before training and inference, standardize each input position using its mean and standard deviation from the **training set only**. Save those statistics and reuse them for validation, testing, and the app. Keep the target in Celsius. The app takes Celsius readings and applies the saved input transformation automatically.

The model does not receive humidity, wind, pressure, station IDs, quality codes, or future readings. Timestamps control ordering and alignment, but are not among the 24 features. During training, `T(t+3)` is the known answer used to calculate error. During a real forecast, that future observation is unavailable and is not supplied to the model. Temperatures at `t+1` and `t+2` are not inputs either.

## Exact app input format

The planned Streamlit app accepts a CSV with **exactly 24 rows** and these two columns:

| Column | Required contents |
|---|---|
| `timestamp_utc` | An ISO 8601 UTC timestamp ending in `Z`, on an exact hour. The 24 timestamps must be unique and consecutive, oldest first. |
| `temperature_c` | A finite numeric Celsius temperature for that hourly endpoint. Negative values are valid. |

Each CSV row represents an hourly value already prepared by the rule above, not an individual raw NOAA report. The app validates the rows, extracts the temperatures in order, applies the saved scaling, and predicts the temperature at the last timestamp plus three hours. No target column is required.

This **illustrative CSV** shows the format; its temperatures are invented examples, not additional NOAA records:

```csv
timestamp_utc,temperature_c
2025-01-01T13:00:00Z,2.1
2025-01-01T14:00:00Z,2.4
2025-01-01T15:00:00Z,2.8
2025-01-01T16:00:00Z,3.0
2025-01-01T17:00:00Z,2.7
2025-01-01T18:00:00Z,2.2
2025-01-01T19:00:00Z,1.8
2025-01-01T20:00:00Z,1.2
2025-01-01T21:00:00Z,0.8
2025-01-01T22:00:00Z,0.4
2025-01-01T23:00:00Z,0.1
2025-01-02T00:00:00Z,-0.2
2025-01-02T01:00:00Z,-0.5
2025-01-02T02:00:00Z,-0.8
2025-01-02T03:00:00Z,-1.0
2025-01-02T04:00:00Z,-1.1
2025-01-02T05:00:00Z,-1.0
2025-01-02T06:00:00Z,-0.7
2025-01-02T07:00:00Z,-0.3
2025-01-02T08:00:00Z,0.2
2025-01-02T09:00:00Z,0.9
2025-01-02T10:00:00Z,1.6
2025-01-02T11:00:00Z,2.3
2025-01-02T12:00:00Z,3.0
```

For this input, the latest endpoint is **January 2, 2025 at 12:00 UTC**, and the model predicts **January 2 at 15:00 UTC**. The persistence baseline predicts **3.0°C**, the latest input value. The neural model's prediction can only be obtained after training and running it.

The raw sample file below cannot be uploaded directly as this 24-hour CSV: it has multiple reports per hour and covers less than 24 hours.

## Training and evaluation alignment

Build the hourly series and forecasting examples separately within the chronological train, validation, and test partitions. Keep complete weeks together and drop examples whose inputs or target cross a partition boundary. Fit scaling on training data only, choose settings using validation data, and reserve the test partition for the final evaluation.

Persistence predicts the latest temperature, `T(t)`. Linear regression and the MLP both use the same 24-feature input and the same eligible examples. Compare their predictions with the observed `T(t+3)` using MAE and RMSE in Celsius.

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
