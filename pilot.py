"""Train one epoch on the earliest 10% of training examples. Run: python pilot.py."""

import argparse
import gzip
import hashlib
import json
import math
import platform
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
STATION = "USW00014740"
INPUT_HOURS, HORIZON, BATCH_SIZE, FULL_EPOCHS, SEED = 24, 3, 64, 20, 11
SPLITS = {
    "train": ("2023-01-02", "2024-12-30"),
    "validation": ("2024-12-30", "2025-06-30"),
    "test": ("2025-06-30", "2025-12-29"),
}
# Only accept documented passing codes from these source families.
GOOD_CODES = {
    **{str(s): {"0", "1", "4", "5", "9"}
       for s in (313, 314, 315, 322, 335, 343, 344, 346)},
    **{str(s): {"1"} for s in (220, 221, 222, 223, 347, 348)},
}


def read_data():
    """Read the full archived files and log every filtering step."""
    manifest = json.loads((ROOT / "data/raw/manifest.json").read_text())
    columns = ["STATION", "DATE", "temperature", "temperature_Quality_Code",
               "temperature_Source_Code"]
    frames = []
    for item in manifest["files"]:
        path = ROOT / "data/raw" / item["file"]
        with gzip.open(path, "rb") as stream:
            raw = stream.read()
        if hashlib.sha256(raw).hexdigest() != item["raw_sha256"]:
            raise ValueError(f"Data checksum mismatch: {path}")
        frames.append(pd.read_csv(BytesIO(raw), sep="|", usecols=columns,
                                  dtype=str, keep_default_na=False))
    frame = pd.concat(frames, ignore_index=True)
    counts = {"raw_records": len(frame)}

    frame["DATE"] = pd.to_datetime(frame["DATE"], utc=True, errors="coerce")
    frame["temperature"] = pd.to_numeric(frame["temperature"], errors="coerce")
    valid = ((frame["STATION"] == STATION) & frame["DATE"].notna()
             & np.isfinite(frame["temperature"]))
    counts["invalid_station_timestamp_or_temperature"] = int((~valid).sum())
    frame = frame.loc[valid].copy()

    good = pd.Series(False, index=frame.index)
    for source, codes in GOOD_CODES.items():
        good |= ((frame["temperature_Source_Code"] == source)
                 & frame["temperature_Quality_Code"].isin(codes))
    counts["rejected_quality_or_unrecognized_source"] = int((~good).sum())
    frame = frame.loc[good].sort_values("DATE", kind="stable")
    counts["duplicate_station_timestamps"] = int(frame.duplicated(["STATION", "DATE"]).sum())
    frame = frame.drop_duplicates(["STATION", "DATE"], keep="last")
    counts["retained_observations"] = len(frame)
    return frame, counts


def make_examples(frame, start, stop):
    """Use only complete hourly bins inside one chronological partition."""
    start, stop = pd.Timestamp(start, tz="UTC"), pd.Timestamp(stop, tz="UTC")
    part = frame.loc[(frame["DATE"] >= start) & (frame["DATE"] < stop)].copy()
    part["hour"] = part["DATE"].dt.ceil("h")
    hourly = part.sort_values("DATE", kind="stable").groupby("hour")["temperature"].last()
    # Exclude bins at the partition boundaries; no input bin can reach outside it.
    hours = pd.date_range(start + pd.Timedelta(hours=1), stop, freq="h", inclusive="left")
    hourly = hourly.reindex(hours)
    inputs = [hourly.shift(lag).rename(f"t_minus_{lag}")
              for lag in range(INPUT_HOURS - 1, -1, -1)]
    candidates = pd.concat(inputs + [hourly.shift(-HORIZON).rename("target")], axis=1)
    complete = candidates.dropna()
    stats = {"hourly_positions": len(hourly), "missing_hours": int(hourly.isna().sum()),
             "candidate_origins": len(candidates), "usable_examples": len(complete),
             "dropped_incomplete_or_boundary_examples": len(candidates) - len(complete)}
    return complete, stats


def make_model():
    return nn.Sequential(nn.Linear(24, 64), nn.ReLU(), nn.Linear(64, 32),
                         nn.ReLU(), nn.Linear(32, 1))


def run(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    hardware = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "CPU"
    is_t4 = device.type == "cuda" and "T4" in hardware.upper().split()
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("A CUDA GPU was requested but is unavailable; CPU timing was not run.")
    if args.require_t4 and not is_t4:
        raise RuntimeError(f"A T4 GPU is required for this measurement; detected {device}: {hardware}")

    preparation_start = time.perf_counter()
    frame, cleaning = read_data()
    training = None
    split_counts = {}
    for name, (start, stop) in SPLITS.items():
        examples, split_counts[name] = make_examples(frame, start, stop)
        if name == "train":
            training = examples
    full_count = len(training)
    if full_count == 0:
        raise ValueError("No complete training examples remain after cleaning.")
    pilot_count = math.ceil(full_count * 0.10)
    pilot = training.iloc[:pilot_count]  # First 10% chronologically, never validation/test.
    x = pilot.iloc[:, :INPUT_HOURS].to_numpy(dtype=np.float32)
    y = pilot[["target"]].to_numpy(dtype=np.float32)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1
    x = (x - mean) / std  # Fit scaling on this pilot's training inputs only.
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    # Include forward/backward passes, optimization, and batch transfer in the timing.
    # CUDA runs asynchronously, so synchronize before and after the measured epoch.
    if device.type == "cuda":
        torch.cuda.synchronize()
    preparation_seconds = time.perf_counter() - preparation_start
    epoch_start = time.perf_counter()
    model.train()
    loss_sum = torch.zeros((), device=device)
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach() * len(batch_x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    epoch_seconds = time.perf_counter() - epoch_start
    estimate_minutes = epoch_seconds * (full_count / pilot_count) * FULL_EPOCHS / 60

    report = {
        "purpose": "One-epoch early pilot; not final model evaluation",
        "device": str(device), "hardware": hardware, "is_t4_measurement": is_t4,
        "seed": SEED, "batch_size": BATCH_SIZE, "learning_rate": 0.001,
        "optimizer": "Adam", "loss": "MSE", "architecture": [24, 64, 32, 1],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "full_training_examples": full_count, "pilot_examples": pilot_count,
        "pilot_fraction": pilot_count / full_count, "pilot_epochs": 1,
        "pilot_first_origin_utc": pilot.index[0].isoformat(),
        "pilot_last_origin_utc": pilot.index[-1].isoformat(),
        "preparation_seconds": preparation_seconds, "epoch_seconds": epoch_seconds,
        "planned_full_epochs": FULL_EPOCHS, "estimated_full_run_minutes": estimate_minutes,
        "estimated_under_30_minutes": estimate_minutes <= 30,
        "training_epoch_mse_celsius_squared": loss_sum.item() / pilot_count,
        "scaler_fitted_on": "pilot training inputs only",
        "cleaning": cleaning, "splits": split_counts,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"model_state_dict": model.cpu().state_dict(), "mean": mean.tolist(),
                "std": std.tolist(), "input_hours": INPUT_HOURS, "horizon_hours": HORIZON,
                "training_epochs": 1, "pilot_only": True}, output / "pilot_checkpoint.pt")
    first_origin = pilot.index[0]
    pd.DataFrame({"timestamp_utc": pd.date_range(end=first_origin, periods=24, freq="h").strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "temperature_c": pilot.iloc[0, :24].to_numpy()}).to_csv(output / "example_input.csv", index=False)
    (output / "example_target.json").write_text(json.dumps({
        "target_timestamp_utc": (first_origin + pd.Timedelta(hours=HORIZON)).isoformat(),
        "observed_target_celsius": float(pilot.iloc[0]["target"]),
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not is_t4:
        print("This is a local/other-device check. Run on a Colab T4 for the required pilot timing.")
    print(f"Saved report, pilot checkpoint, and real example input/target to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Force CPU for a local check.")
    parser.add_argument("--require-gpu", action="store_true", help="Fail unless a CUDA GPU is available.")
    parser.add_argument("--require-t4", action="store_true", help="Fail unless using a T4 GPU.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/pilot"))
    run(parser.parse_args())
