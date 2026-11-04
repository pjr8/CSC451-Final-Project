"""Forecast the next 24 hourly temperatures from the previous 14 days."""
import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pilot import ROOT, SEED, SPLITS, read_data

HISTORY_HOURS = 336
FORECAST_HOURS = 24


def make_sequences(frame, start, stop, history=HISTORY_HOURS, horizon=FORECAST_HOURS):
    """Keep whole input/target windows inside one split; never fill gaps."""
    start, stop = pd.Timestamp(start, tz="UTC"), pd.Timestamp(stop, tz="UTC")
    part = frame.loc[(frame.DATE >= start) & (frame.DATE < stop)].copy()
    part["hour"] = part.DATE.dt.ceil("h")
    hourly = part.sort_values("DATE", kind="stable").groupby("hour").temperature.last()
    hours = pd.date_range(start + pd.Timedelta(hours=1), stop, freq="h", inclusive="left")
    values = hourly.reindex(hours).to_numpy(dtype=np.float32)
    width = history + horizon
    if len(values) < width:
        return np.empty((0, history, 1), np.float32), np.empty((0, horizon), np.float32), hours[:0]
    windows = np.lib.stride_tricks.sliding_window_view(values, width)
    valid = np.isfinite(windows).all(axis=1)
    x = windows[valid, :history, None].copy()
    y = windows[valid, history:].copy()
    origins = hours[history - 1:len(hours) - horizon][valid]
    return x, y, origins


class TemperatureGRU(nn.Module):
    def __init__(self, hidden_size=256, layers=3, horizon=FORECAST_HOURS):
        super().__init__()
        # Keep dropout in the head: cuDNN recurrent dropout can crash at exit on Windows.
        self.gru = nn.GRU(1, hidden_size, num_layers=layers, batch_first=True, dropout=0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(32, horizon))

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum = torch.zeros((), device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.detach() * len(x)
    return loss_sum.item() / len(loader.dataset)


@torch.no_grad()
def validation_mae(model, loader, device, scale):
    model.eval()
    absolute_error = torch.zeros((), device=device)
    for x, y in loader:
        absolute_error += (model(x.to(device)) - y.to(device)).abs().sum()
    return absolute_error.item() * scale / (len(loader.dataset) * FORECAST_HOURS)


def run(args):
    if (not 0 < args.fraction <= 1 or args.epochs < 1 or args.planned_epochs < 1
            or min(args.batch_size, args.hidden_size, args.layers, args.patience) < 1):
        raise ValueError("Use fraction in (0, 1] and positive epoch counts.")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("CUDA GPU required; no CPU timing was performed.")
    hardware = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "CPU"
    is_t4 = device.type == "cuda" and "T4" in hardware.upper().split()
    if args.require_t4 and not is_t4:
        raise RuntimeError(f"T4 required; detected {hardware}.")

    preparation_start = time.perf_counter()
    frame, cleaning = read_data()
    datasets = {name: make_sequences(frame, *dates) for name, dates in SPLITS.items()}
    train_x, train_y, origins = datasets["train"]
    full_count = len(train_x)
    if not full_count:
        raise ValueError("No complete training sequences.")
    count = math.ceil(full_count * args.fraction)
    selected_x, selected_y = train_x[:count], train_y[:count]
    mean, scale = float(selected_x.mean()), float(selected_x.std()) or 1.0
    x = torch.from_numpy((selected_x - mean) / scale)
    y = torch.from_numpy((selected_y - mean) / scale)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    val_loader = None
    if args.fraction == 1:
        val_x, val_y, _ = datasets["validation"]
        if not len(val_x):
            raise ValueError("No complete validation sequences.")
        val_loader = DataLoader(TensorDataset(torch.from_numpy((val_x - mean) / scale),
                                             torch.from_numpy((val_y - mean) / scale)),
                                batch_size=args.batch_size)
    model = TemperatureGRU(args.hidden_size, args.layers).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters >= 1_000_000:
        raise ValueError(f"Model has {parameters:,} parameters; keep it below 1 million.")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if device.type == "cuda":
        torch.cuda.synchronize()
    preparation_seconds = time.perf_counter() - preparation_start
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history, best_mae, best_epoch = [], float("inf"), None
    checkpoint_path = output / "sequence_checkpoint.pt"

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        loss = train_epoch(model, loader, optimizer, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        training_seconds = time.perf_counter() - started
        val_seconds, mae = 0.0, None
        if val_loader is not None:
            started = time.perf_counter()
            mae = validation_mae(model, val_loader, device, scale)
            if device.type == "cuda":
                torch.cuda.synchronize()
            val_seconds = time.perf_counter() - started
        if not math.isfinite(loss) or (mae is not None and not math.isfinite(mae)):
            raise RuntimeError("Non-finite training or validation result.")
        checkpoint_seconds = 0.0
        # Pilot: save the final epoch. Full run: retain the best validation checkpoint.
        if mae is None or mae < best_mae:
            best_mae, best_epoch = mae if mae is not None else best_mae, epoch
            started = time.perf_counter()
            torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "mean": mean, "scale": scale, "history_hours": HISTORY_HOURS,
                        "forecast_hours": FORECAST_HOURS, "hidden_size": args.hidden_size,
                        "layers": args.layers, "epoch": epoch, "training_fraction": args.fraction},
                       checkpoint_path)
            checkpoint_seconds = time.perf_counter() - started
        history.append({"epoch": epoch, "training_seconds": training_seconds,
                        "training_mse_standardized": loss,
                        "validation_mae_celsius": mae, "validation_seconds": val_seconds,
                        "checkpoint_seconds": checkpoint_seconds})
        print(json.dumps(history[-1]), flush=True)
        if val_loader is not None and epoch - best_epoch >= args.patience:
            print("Stopped after validation MAE did not improve.", flush=True)
            break

    mean_epoch_seconds = float(np.mean([r["training_seconds"] for r in history]))
    estimate = mean_epoch_seconds * full_count / count * args.planned_epochs / 60
    report = {
        "purpose": "24-hour temperature-curve forecasting runtime pilot",
        "device": str(device), "hardware": hardware, "is_t4_measurement": is_t4,
        "history_hours": HISTORY_HOURS, "forecast_hours": FORECAST_HOURS,
        "model": "GRU", "hidden_size": args.hidden_size, "layers": args.layers,
        "parameter_count": parameters, "head_dropout": 0.1, "recurrent_dropout": 0.0,
        "batch_size": args.batch_size, "optimizer": "Adam", "learning_rate": 0.001,
        "gradient_clip_norm": 1.0, "seed": SEED,
        "full_training_examples": full_count, "selected_training_examples": count,
        "requested_fraction": args.fraction, "epochs_completed": len(history),
        "planned_full_epochs": args.planned_epochs, "preparation_seconds": preparation_seconds,
        "mean_training_epoch_seconds": mean_epoch_seconds,
        "estimated_full_training_minutes": estimate,
        "validation_seconds_per_epoch": float(np.mean([r["validation_seconds"] for r in history])),
        "estimated_full_run_minutes": (float(np.mean([r["training_seconds"] + r["validation_seconds"]
                                                        + r["checkpoint_seconds"] for r in history]))
                                       * args.planned_epochs / 60 if args.fraction == 1 else None),
        "training_estimate_excludes": "data preparation, validation, and checkpoint writing",
        "full_run_estimate_includes": "training, validation, and checkpoint writing; excludes one-time preparation",
        "split_examples": {name: len(data[0]) for name, data in datasets.items()},
        "first_training_origin_utc": origins[0].isoformat(),
        "last_selected_training_origin_utc": origins[count - 1].isoformat(),
        "scaling": {"mean_celsius": mean, "std_celsius": scale,
                    "fitted_on": "selected training inputs only", "target_standardized": True},
        "best_epoch": best_epoch, "history": history, "cleaning": cleaning,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "numpy": np.__version__, "pandas": pd.__version__},
    }
    (output / "sequence_report.json").write_text(json.dumps(report, indent=2) + "\n")
    first_origin = origins[0]
    pd.DataFrame({"timestamp_utc": pd.date_range(end=first_origin, periods=HISTORY_HOURS, freq="h").strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "temperature_c": selected_x[0, :, 0]}).to_csv(output / "example_input.csv", index=False)
    pd.DataFrame({"timestamp_utc": pd.date_range(start=first_origin + pd.Timedelta(hours=1), periods=FORECAST_HOURS, freq="h").strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "temperature_c": selected_y[0]}).to_csv(output / "example_targets.csv", index=False)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--planned-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-t4", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/sequence-pilot"))
    run(parser.parse_args())
