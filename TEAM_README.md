# COMP90051 Group Project — Group 89

**COMP90051 Statistical Machine Learning · 2026 Semester 2 · Group Project**

## Team

| Name | Student ID | Model / main role |
|---|---|---|
| Bin Xing | 1634582 | Decision Tree |
| Chenguang Shen | 1709322 | Transformer model |
| Jiaxin He | 1455893 | SVM |

- **Weekly meeting:** every Wednesday, 3 pm (at least 3 meetings are required; minutes go in `meetings/`)
- **Repo:** `wlgbad/COMP90051_group-project`
- **Reply time:** within 24 hours on team chat

## Key dates

| Date | Item |
|---|---|
| Fri 25 Sep 2026, 11:59 pm | Plan / questions + group agreement (ungraded) — done |
| **Fri 16 Oct 2026, 11:59 pm** | **Final submission:** 4-page PDF report + code ZIP (with `Readme.txt`) |

## Research question

_Draft, to be confirmed by the team:_ How does **sensor drift over time** affect simple, medium and complex models? Can drift-aware feature construction keep gas classification reliable on later batches when the models are trained on earlier ones?

> Rule: the question must go beyond "predict the gas type" and beyond "compare 3 models".

**Dataset:** UCI Gas Sensor Array Drift Dataset
URL: <https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset>

- 13,910 samples, 128 features (16 chemical sensors × 8 features each), 6 gas classes
- Recorded over 36 months in **10 batches** (`batch1.dat` … `batch10.dat`, libsvm format)
- Reference: Vergara, A. et al. (2012). Chemical gas sensor drift compensation using classifier ensembles. *Sensors and Actuators B: Chemical*, 166–167, 320–329.

**Do not commit or submit data.** Put the raw `.dat` files under `data/` (git-ignored).

## Project requirements checklist

### Data and features (spec 2.b–2.c)
- [x] Tabular data with ≥ 10,000 instances and ≥ 100 features: 13,910 × 128, which already meets the requirement. No duplicated or random padding.
- [ ] Non-trivial feature construction. One-hot encoding or normalisation alone is **not** enough. Ideas:
  - Drift correction: per-batch standardisation, or aligning later batches to earlier ones (e.g. CORAL-style covariance alignment)
  - Ratios between sensors or between steady-state and transient features
  - Removing the drift direction (PCA/component correction estimated from earlier batches)
- [ ] Subsampling is allowed if nested CV is too slow.

### Three models (spec 2.d)
- [ ] Simple / medium / complex, three different algorithms (not one algorithm with different hyperparameters)
- [ ] At least one not covered in class **and** first published in 2017 or later at NeurIPS, ICML/UAI/AISTATS/JMLR, ICLR or TMLR (not a workshop paper)

| Complexity | Model | Owner | Tuned hyperparameters (≥ 3 values each) |
|---|---|---|---|
| Simple | Decision Tree | Bin | e.g. `max_depth`, `min_samples_leaf` |
| Medium | SVM (RBF kernel) | Jiaxin | e.g. $C$, $\gamma$ |
| Complex | Transformer model (e.g. FT-Transformer, NeurIPS 2021) | Chenguang | e.g. number of layers, embedding dim, dropout |

> The Transformer is the post-2017 paper model.

### Evaluation (spec 2.e–2.g): all written from scratch
- [ ] Outer CV: k-fold with $k \ge 10$ (or ≥ 10 repeated splits, or bootstrap $B \ge 10$). The **same scheme for all 3 models.**
  - The data is temporal, so **train on the past and test on the future.** For example, run ≥ 10 repetitions of "train on batches $1..t$, test on batch $t+1$". Another option: randomly sample the training set from earlier batches and the test set from later batches, 10 times. Avoid plain random k-fold, because it hides drift.
- [ ] Inner (nested) CV for tuning: $k \ge 3$ or ≥ 3 splits. No single validation set.
- [ ] Hyperparameter search written by hand (no `GridSearchCV`, `cross_val_score`, etc.)
- [ ] For each hyperparameter, the **middle** value should be the one chosen most often. If not, shift the grid.
- [ ] Metrics written by hand (no `sklearn.metrics`): ≥ 3 results, with **error bars**
  - Ideas: accuracy and macro-F1 per test batch (accuracy vs batch distance shows drift), per-class recall, confusion matrix, learning curves vs **training-set size** (not loss vs epochs)
- [ ] Third-party libraries are fine for the models themselves and for plotting

## Repository layout

```
.
├── data/               # raw data, git-ignored, NEVER commit
├── src/
│   ├── features.py     # loading batches and drift-aware feature construction
│   ├── cv.py           # outer and nested CV, tuning (from scratch)
│   ├── metrics.py      # metrics (from scratch)
│   └── models/         # one file per model
├── notebooks/          # EDA and experiments
├── results/            # saved CV results (csv/json) and figures
├── meetings/           # meeting minutes
├── report/             # LaTeX / Word source of the report
└── Readme.txt          # short "how to run" note for the submission
```

## Workflow

- One branch per model or feature (e.g. `decision-tree`). Merge into `main` after the team agrees.
- Review your own code before you push. Never push data or large result files.
- Fix shared things (random seeds, fold indices, the feature table) **once**, so all three models use identical folds.
- Decisions are made together at meetings and written in `meetings/`.
- If you are going to miss a deadline or a meeting, tell the team **early**.

## Report (4 pages total, including references)

Single column, A4, 10pt font, 1.5–2 cm margins, no cover page. Sections:

1. Student IDs and names
2. Introduction: research question, dataset and URL
3. Literature review: dataset reference and ≥ 2 papers (one on the post-2017 model)
4. Methods: features, preprocessing, 3 models, CV, tuning
5. Results and discussion: tables and figures with error bars; *why* each method works or fails; alternatives we considered
6. Conclusion
7. References (APA)

## Submission (16 Oct)

- [ ] `report.pdf` (≤ 4 pages)
- [ ] `code.zip`: code or notebooks, `Readme.txt` with a few lines on how to run, any automation scripts
- [ ] Optional: GitHub/Slack logs and meeting minutes
- [ ] **No data in the ZIP**

## Marking (30 marks)

- Completeness and correctness: 12
- Clarity and structure: 6
- Critical analysis: 12 (motivation, pros and cons, *why* it works, alternatives)

## Group agreement (summary)

Weekly Wednesday meetings, tell the team ahead if you can't attend, reply within 24 hours, finish tasks on time and to a good standard. Problems are first discussed as a team. If they persist, they are recorded in the end-of-project peer and self-assessment.
