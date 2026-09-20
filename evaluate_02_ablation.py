"""Evaluate the four variants used in the ablation experiment."""

import torch

from evaluate_01_model_comparison import (
    HISTORICAL_DATA_ROOT,
    PredRNN,
    DEVICE,
    INFERENCE_BATCH_SIZE,
    WEIGHT_ROOT,
    RasterSequenceDataset,
    build_samples,
    create_historical_model,
    evaluate,
    load_checkpoint,
    print_results,
)
from models.historical.dgaclstm import DGACLSTM, EncoderDecoderNoAttention
from models.historical.predrnn_diff_geo import PredRNNDIFFGEO
from models.historical.predrnn_diff import PredRNNDIFF


def main() -> None:
    """Run the ablation experiment on the shared historical test set."""
    samples = build_samples(HISTORICAL_DATA_ROOT)
    dataset = RasterSequenceDataset(samples)
    print(f"Device: {DEVICE}; test samples: {len(dataset)}")
    configurations = (
        ("PredRNN (No DIFF & GEO)", PredRNN, "PredRNN_best.pth"),
        ("PredRNN + DIFF", PredRNNDIFF, "PredRNN_DIFF_best.pth"),
        ("PredRNN + DIFF & GEO", PredRNNDIFFGEO, "PredRNN_DIFF_GEO_best.pth"),
        ("Encoder-Decoder (No Attention)", EncoderDecoderNoAttention, "EncoderDecoder_NoAttention_best.pth"),
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
