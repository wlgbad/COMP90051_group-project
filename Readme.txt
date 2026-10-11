COMP90051 Group Project - gas sensor array robustness (RBF-SVM / medium model part)

Requires Python 3.11+ with numpy, pandas, scikit-learn, joblib, matplotlib.
Place the UCI #270 batch*.dat files in
./gas+sensor+array+drift+dataset+at+different+concentrations/ (data is not included).
Shared team protocol: shared_protocol.py, shared_metrics.py, shared_robustness.py,
sensor_failure_patterns.json (see TEAM_README.txt); feature construction: shared_features.py.
Run from the project root:

  python phase1_data_pipeline.py              # clean table + 10 temporal outer splits
  python -m tests.test_metrics                # check hand-written metrics vs sklearn
  python -m medium.run_tuning                 # nested temporal CV, caches models
  python -m medium.run_failure_sweep          # inference-only sweep, k = 0, 1, 2, 4, 8
  python -m medium.run_dropout                # sensor-dropout trained models
  python -m medium.run_failure_sweep --variant dropout
  python -m medium.run_learning_curve
  python -m medium.run_gamma_ablation
  python -m medium.make_figures               # figures/ and results/medium_summary.csv

Add --smoke to any medium.* step for a 1-repeat, tiny-grid check. --n-jobs sets parallelism.
