COMP90051 group project — FT-Transformer component.
Requires Python, NumPy, pandas, PyTorch and the group's Phase 1 pipeline outputs
in phase1_output/ (gas_sensor_clean.csv and temporal_outer_splits.json).
From the project root, run:
  python train_ft.py --device cuda
  python robustness_ft.py --device cuda
For a quick check, append --validate-only or --repeats 1.
The fixed masks are in shared/sensor_failure_patterns.json. Group shared
scripts are in shared/. Do not include the dataset or model output directories
in the submitted source-code ZIP.
