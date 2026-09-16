# operational-pressure-forecasting
Estimate the future hourly operational load index for each series_id


This project is about multivariate time series forecasting with deep learning.

## Important Links

- Dataset: https://huggingface.co/datasets/AIML-TUDA/dlam-ts-project-data-2026
- Hugging Face leaderboard Space: https://aiml-tuda-dlam-ts-project-leaderboard-2026.hf.space/

Leaderboard metrics: MAE, MSE, RMSE, MAPE, sMAPE, and WAPE. Lower is better for all metrics.

Target: predict the future hourly operational load index for each `series_id`. Higher values mean more operational pressure in that unit.


## Public Validation Leaderboard

Before building the model, generate simple baseline prediction files from `./baseline/`.

Available baselines:

- `naive_last_value`: repeat the final observed target per series.
- `lag24_repeat`: repeat the last 24 observed values.
- `lag168_repeat`: repeat the last 168 observed values.
- `seasonal_mean`: average historical targets by series, day-of-week, and hour-of-day.

These are deliberately simple. Our project model should improve on the provided seasonal-mean baseline.
```bash
DATA_DIR=/path/to/downloaded/hf/dataset
python student/baseline/run_baselines.py \
  --train "$DATA_DIR/train.csv" \
  --forecast-index "$DATA_DIR/forecast_index_validation.csv" \
  --output-dir /tmp/student_baselines
```


Prediction format:

```csv
series_id,timestamp,prediction
```


## Final Model Submission

Final test data is private. We won't receive test labels or private test inputs. Instead, we have to create a runnable model archive. During private evaluation, the script receives a private input directory containing `test_input.csv`, `forecast_index_test.csv`, and `metadata.json`.


`final_submission.zip` must contain:

```text
predict.py
requirements.txt
checkpoint.pt
src/  # optional
```

During private evaluation, instructors run:

```bash
python predict.py --input_dir /data/input --output_file /output/predictions.csv --checkpoint /submission/checkpoint.pt
```

Script must write the prediction CSV to the requested output path.

When uploading the final archive, use the same model name as the validation row that corresponds to the submitted checkpoint.


## Reproducibility Requirements

- Document training and inference steps in README.
- Fix random seeds where reasonable.
- State all important hyperparameters.