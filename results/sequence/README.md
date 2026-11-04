# Recurrent forecast pilot

Measured November 4, 2026 on an **NVIDIA GeForce RTX 3070 Ti**. These are successful one-epoch runs; a complete 30-epoch training run has **not** been performed.

## Task and configuration

Predict the next **24 hourly temperatures** from the preceding **336 hours**. Three GRU layers with 256 hidden units feed a `256 → 32 → 24` head. The model has **997,432 parameters**. Training uses Adam at 0.001, batch size 32, gradient clipping at 1.0, and dropout 0.1 in the output head. The training cap is 30 epochs with validation stopping after five non-improving epochs.

| Data split | Complete examples |
|---|---:|
| Training | 15,306 |
| Validation | 3,371 |
| Test | 2,528 |
| First 10% training pilot | 1,531 |

## Measured timing

| Measurement | Time |
|---|---:|
| One epoch on the first 10% | 2.786 seconds |
| 30-epoch training estimate from that pilot | 13.93 minutes |
| One measured full-data training epoch | 22.579 seconds |
| Validation after that full-data epoch | 2.220 seconds |
| Checkpoint write after that epoch | 0.008 seconds |
| 30-epoch training-only estimate from the full epoch | 11.29 minutes |
| 30-epoch estimate including validation and checkpoint writes | **12.40 minutes** |
| One-time preparation for the full-data check | 2.355 seconds |

The **12.40-minute** estimate is the measured full training, validation, and checkpoint cycle multiplied by 30. It excludes one-time preparation. Early stopping may finish sooner. The short pilot and full-data extrapolations differ because fixed costs have more influence on a small run. These are hardware-specific estimates, not measured 30-epoch durations or T4 timings.

The one-full-epoch validation MAE was 3.237°C across the 24 forecast hours. This is an initial validation result, not a final test score. No test-set predictions were evaluated.

## Saved evidence

- [First 10% pilot report](pilot_report.json)
- [Full-data single-epoch report](full_epoch_report.json)
- [First 10% pilot checkpoint](pilot_checkpoint.pt)
- [Real 336-hour input example](example_input.csv)
- [Its 24 observed future temperatures](example_targets.csv)

The reports include settings, package versions, cleaning counts, and the script hash. Both runs finished with exit code 0. The checkpoint is from the first 10% pilot only.

## Proposal wording

We will forecast the next 24 hourly temperatures at Hartford Bradley International Airport using the previous 14 days of temperature readings. A three-layer GRU with 256 hidden units and a 24-value output has 997,432 parameters. The dataset contains 15,306 training, 3,371 validation, and 2,528 test examples. On an RTX 3070 Ti, one full training epoch took 22.58 seconds, with 2.22 seconds for validation. Thirty epochs are estimated at 12.4 minutes including validation and checkpoint writes. We will stop earlier if validation MAE does not improve for five epochs.

The original GPU quota request was made on November 2, 2026, and the follow-up quota increase was approved on November 4. T4 capacity was unavailable in the three regions tested. These measurements use the RTX 3070 Ti; the required T4 measurement remains to be completed when capacity is available.
