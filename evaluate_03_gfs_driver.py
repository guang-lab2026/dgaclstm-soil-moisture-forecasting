"""Compare historical-only and GFS-driven DGACLSTM forecasts."""

import torch

from evaluate_01_model_comparison import (
    HISTORICAL_DATA_ROOT,
    DEVICE,
    GFS_FORECAST_ROOT,
    INFERENCE_BATCH_SIZE,
    WEIGHT_ROOT,
    RasterSequenceDataset,
    build_samples,
    create_historical_model,
    evaluate,
    load_checkpoint,
    print_results,
)
from models.gfs.dgaclstm import DGACLSTM as GFSModel
from models.historical.dgaclstm import DGACLSTM as HistoricalModel


def create_gfs_model(checkpoint: dict) -> torch.nn.Module:
    """Construct the GFS-driven model and load its complete state dictionary."""
    args = checkpoint["args"]
    options = {"len": args.input_len + args.target_len, "fixed_number": args.fixed_number}
    model = GFSModel(
        args.input_dim,
        args.hidden_dim,
        args.output_dim,
        args.kernel_size,
        args.num_layers,
        args.forecast_chans,
        options,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def main() -> None:
    """Run both rows of the historical-versus-GFS experiment."""
    historical_samples = build_samples(HISTORICAL_DATA_ROOT)
    historical_dataset = RasterSequenceDataset(historical_samples)
    paired_samples = build_samples(HISTORICAL_DATA_ROOT, GFS_FORECAST_ROOT)
    paired_dataset = RasterSequenceDataset(paired_samples, GFS_FORECAST_ROOT)
    print(
        f"Device: {DEVICE}; historical samples: {len(historical_dataset)}; "
        f"GFS-paired samples: {len(paired_dataset)}"
    )

    historical_checkpoint = load_checkpoint(WEIGHT_ROOT / "historical" / "DGACLSTM_best.pth")
    historical_model = create_historical_model(HistoricalModel, historical_checkpoint)
    historical_results = evaluate(
        historical_model,
        historical_dataset,
        INFERENCE_BATCH_SIZE,
        "sample",
        use_amp=historical_checkpoint["args"].amp,
    )
    print_results("Historical-forcing configuration", historical_results)
    del historical_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    gfs_checkpoint = load_checkpoint(WEIGHT_ROOT / "gfs" / "DGACLSTM_best.pth")
    gfs_model = create_gfs_model(gfs_checkpoint)
    gfs_results = evaluate(
        gfs_model,
        paired_dataset,
        batch_size=10,
        aggregation="batch",
        use_forecast=True,
        use_amp=gfs_checkpoint["args"].amp,
    )
    print_results("Forecast-forcing configuration", gfs_results)


if __name__ == "__main__":
    main()
