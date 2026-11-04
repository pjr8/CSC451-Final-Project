# Early pilot results

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

## Saved artifacts

- [Full measurement and cleaning report](rtx3070ti/pilot_report.json)
- [Pilot checkpoint](rtx3070ti/pilot_checkpoint.pt), trained for one epoch only
- [Real 24-hour example input](rtx3070ti/example_input.csv)
- [Observed target for that example](rtx3070ti/example_target.json)

The checkpoint is an early pilot model. Its training loss is not a validation or test score. Validation and test examples were counted but were not used to train, scale, or evaluate this pilot.
