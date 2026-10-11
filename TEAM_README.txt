TEAM SHARED EXPERIMENT PROTOCOL
===============================

Send these files together:
1. shared_protocol.py
2. shared_metrics.py
3. shared_robustness.py
4. sensor_failure_patterns.json
   (send the FINAL JSON from Phase 6B separately)

Phase 1 files already shared:
- phase1_output/gas_sensor_clean.csv
- phase1_output/temporal_outer_splits.json

COMMON MODEL INPUT
------------------
Use ONLY x1..x128 as model features.

Do NOT use:
row_id, batch, label_raw, target, concentration

OUTER CV
--------
Use the 10 predefined outer splits in temporal_outer_splits.json.

INNER TEMPORAL CV
-----------------
Inside each outer training set:

Inner 1:
train batches 1-2
validation batches 3-6

Inner 2:
train batches 1-5
validation batch 6

Inner 3:
train batches 1-6
validation batch 7

Use shared_protocol.make_inner_temporal_splits(...).

METRICS
-------
Use shared_metrics.classification_metrics_from_scratch(...).

Main clean metrics:
- Accuracy
- Macro-F1

ROBUSTNESS
----------
Train on CLEAN data only.
Do not retrain under sensor failure.

Failure levels:
k = 0, 1, 2, 4, 8

Use exactly the shared Phase 6B sensor_failure_patterns.json.
Do not generate new random masks.

One sensor = 8 raw features:
sensor 1  = x1-x8
sensor 2  = x9-x16
...
sensor 16 = x121-x128

Failure imputation:
- raw/unstandardised input:
  replace failed features with OUTER-TRAIN feature means.
- input standardised with OUTER-TRAIN mean/std:
  set failed standardised features to 0.

Use shared_robustness.py.

MAIN ERROR BARS
---------------
For each outer repeat and severity:
first average Macro-F1 across failure masks.

Then across the 10 outer-repeat means:
report mean +/- SD.

SD across masks is NOT the main CV error bar.
It only measures sensitivity to WHICH sensors fail.

Random seed:
90051
