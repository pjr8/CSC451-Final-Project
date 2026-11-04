# Original three-hour MLP pilot

The current, more complex forecast is documented in [the recurrent model results](sequence/README.md).

Measured November 4, 2026. This is a one-epoch feasibility check, not a final accuracy evaluation.

| Item | Measured value or setting |
|---|---:|
| GPU | NVIDIA GeForce RTX 3070 Ti |
| Full training examples | 17,312 |
| Pilot examples | 1,732 (first 10%, rounded up) |
| Model | 24 → 64 → 32 → 1 MLP; 3,713 parameters |
| Optimizer | Adam, learning rate 0.001 |
| Batch size | 64 |
| Measured epochs | 1 |
| Data/model preparation | 1.873 seconds |
| Measured training epoch | 0.243 seconds |
| Estimated full 20-epoch training | 0.81 minutes |

The runtime estimate is `epoch seconds × (17,312 / 1,732) × 20 / 60`. It excludes preparation and is specific to this GPU and run. A full-data epoch has not yet been timed. CUDA work was synchronized before and after the measured epoch; no training warm-up was performed.

## T4 availability

The original GPU quota request was made on November 2, 2026. A follow-up request increased the Google Cloud global GPU quota from 0 to 1 on November 4, 2026. The regional T4 limit was already 1. Colab Enterprise allocation attempts in `us-central1`, `us-east1`, and `us-west1` then failed because those regions did not have sufficient T4 capacity. No T4 training measurement was obtained. The RTX 3070 Ti result above uses the planned model and settings, but does not establish T4 performance.

## Proposal paragraph

We requested GPU quota on November 2. Quota was approved on November 4, but T4 capacity was unavailable in the three regions tested. We therefore timed one epoch on an NVIDIA RTX 3070 Ti using the planned 24–64–32–1 MLP, Adam with learning rate 0.001, and batch size 64. The first 1,732 of 17,312 training examples took 0.243 seconds. The estimated 20-epoch full-data runtime is 0.81 minutes, excluding preparation. This estimate applies to the RTX 3070 Ti; we will repeat the measurement on a T4 when capacity is available.

## Saved artifacts

- [Full measurement and cleaning report](rtx3070ti/pilot_report.json)
- [Pilot checkpoint](rtx3070ti/pilot_checkpoint.pt), trained for one epoch only
- [Real 24-hour example input](rtx3070ti/example_input.csv)
- [Observed target for that example](rtx3070ti/example_target.json)

The checkpoint is an early pilot model. Its training loss is not a validation or test score. Validation and test examples were counted but were not used to train, scale, or evaluate this pilot.
