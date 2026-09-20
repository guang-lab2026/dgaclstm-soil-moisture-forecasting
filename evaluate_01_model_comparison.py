"""Evaluate the four models used in the model-comparison experiment.

The module also contains the shared data loading and metric routines imported by
the other two test entry points. Paths are resolved relative to this file.
"""

from __future__ import annotations

import glob
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from osgeo import gdal
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
HISTORICAL_DATA_ROOT = ROOT / "test_data" / "historical_inputs"
GFS_FORECAST_ROOT = ROOT / "test_data" / "gfs_forecasts"
WEIGHT_ROOT = ROOT / "weights"
INPUT_LENGTH = 3
TARGET_LENGTH = 7
INFERENCE_BATCH_SIZE = 10
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def install_timm_fallback() -> None:
    """Provide initialization-only timm symbols when timm cannot be imported."""
    try:
        from timm.layers import trunc_normal_tf_  # noqa: F401
        from timm.models import named_apply  # noqa: F401
        return
    except Exception:
        for name in tuple(sys.modules):
            if name == "timm" or name.startswith("timm."):
                sys.modules.pop(name, None)

    timm = types.ModuleType("timm")
    timm_models = types.ModuleType("timm.models")
    timm_layers = types.ModuleType("timm.layers")
    timm_models.named_apply = lambda function, module, *args, **kwargs: module
    timm_layers.trunc_normal_tf_ = lambda tensor, *args, **kwargs: tensor
    timm.models = timm_models
    timm.layers = timm_layers
    sys.modules.update({"timm": timm, "timm.models": timm_models, "timm.layers": timm_layers})


install_timm_fallback()
gdal.UseExceptions()

from models.historical.dgaclstm import DGACLSTM  # noqa: E402
from models.historical.convlstm import ConvLSTM  # noqa: E402
from models.historical.predrnn import PredRNN, PredRNNV2  # noqa: E402


class RasterSequenceDataset(Dataset):
    """Load three input days, seven target days, and optional GFS forecasts."""

    def __init__(self, samples: list[list[str]], forecast_root: Path | None = None):
        self.samples = samples
        self.forecast_root = forecast_root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        paths = self.samples[index]
        arrays = []
        for path in paths:
            dataset = gdal.Open(path)
            array = np.stack(
                [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
            )
            arrays.append(np.nan_to_num(array))
            dataset = None

        sequence = torch.from_numpy(np.stack(arrays)).float()
        anchor_name = Path(paths[INPUT_LENGTH - 1]).name.split(".")[1]
        date, region = anchor_name.split("_")
        forecast = torch.zeros(1, 1, 1, 1)
        if self.forecast_root is not None:
            path = self.forecast_root / date[:4] / region / f"gfs_forecast.{date}_{region}.npy"
            forecast = torch.from_numpy(np.load(path)).float()
        return sequence[:INPUT_LENGTH], sequence[INPUT_LENGTH:], forecast, (date, region)


def build_samples(data_root: Path, forecast_root: Path | None = None) -> list[list[str]]:
    """Build sliding windows independently for every year and spatial region."""
    samples: list[list[str]] = []
    skipped = 0
    for year_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for region_dir in sorted(path for path in year_dir.iterdir() if path.is_dir()):
            files = sorted(glob.glob(str(region_dir / "*.tif")))
            for start in range(len(files) - INPUT_LENGTH - TARGET_LENGTH + 1):
                window = files[start : start + INPUT_LENGTH + TARGET_LENGTH]
                if forecast_root is not None:
                    anchor = Path(window[INPUT_LENGTH - 1]).name.split(".")[1]
                    date, region = anchor.split("_")
                    forecast = forecast_root / date[:4] / region / f"gfs_forecast.{date}_{region}.npy"
                    if not forecast.exists():
                        skipped += 1
                        continue
                samples.append(window)
    if skipped:
        print(f"Skipped {skipped} samples with missing GFS forecasts.")
    return samples


def compute_metrics(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return R2, RMSE, PSNR, and MAE for flattened valid pixels."""
    error = target - prediction
    mse = torch.mean(error.square())
    total = torch.sum((target - target.mean()).square())
    r2 = 1.0 - torch.sum(error.square()) / total
    rmse = torch.sqrt(mse)
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
    mae = torch.mean(error.abs())
    return torch.stack((r2, rmse, psnr, mae)).double().cpu()


def create_historical_model(model_class, checkpoint: dict) -> torch.nn.Module:
    """Construct a historical-input model and load its complete state dictionary."""
    args = checkpoint["args"]
    options = {"len": args.input_len + args.target_len, "fixed_number": args.fixed_number}
    model = model_class(
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        args.kernel_size,
        args.num_layers,
        options,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def evaluate(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    aggregation: str,
    use_forecast: bool = False,
    use_amp: bool = True,
) -> dict[str, list[float]]:
    """Evaluate with per-sample or original per-batch metric aggregation."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    totals = torch.zeros(4, TARGET_LENGTH, dtype=torch.float64)
    total_samples = 0
    model = model.to(DEVICE).eval()

    with torch.no_grad():
        for batch_index, (inputs, targets, forecasts, _) in enumerate(loader, start=1):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            forecasts = forecasts.to(DEVICE)
            with torch.amp.autocast(DEVICE.type, enabled=use_amp and DEVICE.type == "cuda"):
                outputs, _ = model(inputs, targets, forecasts) if use_forecast else model(inputs, targets)

            predictions = outputs[-TARGET_LENGTH:, :, 0].permute(1, 0, 2, 3).float()
            truths = targets[:, :, 0]
            batch_count = truths.shape[0]
            if aggregation == "batch":
                for day in range(TARGET_LENGTH):
                    mask = truths[:, day] != -1
                    totals[:, day] += compute_metrics(predictions[:, day][mask], truths[:, day][mask]) * batch_count
            elif aggregation == "sample":
                for sample in range(batch_count):
                    for day in range(TARGET_LENGTH):
                        mask = truths[sample, day] != -1
                        totals[:, day] += compute_metrics(
                            predictions[sample, day][mask], truths[sample, day][mask]
                        )
            else:
                raise ValueError(f"Unsupported aggregation mode: {aggregation}")
            total_samples += batch_count
            if batch_index % 25 == 0:
                print(f"Processed {total_samples}/{len(dataset)} samples.")

    values = totals / total_samples
    names = ("R2", "RMSE", "PSNR", "MAE")
    return {name: values[index].tolist() for index, name in enumerate(names)}


def print_results(name: str, results: dict[str, list[float]]) -> None:
    """Print daily values and their seven-day mean."""
    print(f"\n{name}")
    print("Metric     Day 1     Day 2     Day 3     Day 4     Day 5     Day 6     Day 7      Mean")
    for metric in ("R2", "RMSE", "PSNR", "MAE"):
        values = results[metric]
        body = " ".join(f"{value:9.4f}" for value in values)
        print(f"{metric:<7} {body} {sum(values) / len(values):9.4f}")


def load_checkpoint(path: Path) -> dict:
    """Load a state-dictionary checkpoint on CPU."""
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    """Run the model-comparison experiment on the shared historical test set."""
    samples = build_samples(HISTORICAL_DATA_ROOT)
    dataset = RasterSequenceDataset(samples)
    print(f"Device: {DEVICE}; test samples: {len(dataset)}")
    configurations = (
        ("ConvLSTM", ConvLSTM, "ConvLSTM_best.pth"),
        ("PredRNN", PredRNN, "PredRNN_best.pth"),
        ("PredRNN-V2", PredRNNV2, "PredRNN_V2_best.pth"),
        ("DGACLSTM", DGACLSTM, "DGACLSTM_best.pth"),
    )
    for name, model_class, filename in configurations:
        checkpoint = load_checkpoint(WEIGHT_ROOT / "historical" / filename)
        model = create_historical_model(model_class, checkpoint)
        results = evaluate(model, dataset, INFERENCE_BATCH_SIZE, "sample", use_amp=checkpoint["args"].amp)
        print_results(name, results)
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
