# Paper Experiments

This package contains the model code, shared test data, trained weights, and three evaluation entry points used by the manuscript.

- `evaluate_01_model_comparison.py`: ConvLSTM, PredRNN, PredRNN-V2, and DGACLSTM.
- `evaluate_02_ablation.py`: PredRNN baseline, DIFF, DIFF & GEO, encoder-decoder without attention, and DGACLSTM.
- `evaluate_03_gfs_driver.py`: historical-forcing and forecast-forcing configurations.

Historical raster files use `historical.YYYYMMDD_DBxx.tif`. GFS arrays use `gfs_forecast.YYYYMMDD_DBxx.npy`. Ordered variable names are listed in `test_data/historical_inputs/variables.json`.
