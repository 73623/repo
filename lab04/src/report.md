Lab 04 Report: Adult Income Classification

Objective

This lab builds and evaluates a machine-learning pipeline to predict whether an
adult's income is greater than $50,000 using the UCI Adult dataset (OpenML
version 2). The work also checks data quality, tests derived features, and
saves artifacts needed to reproduce or inspect the processing.

Data and method

The dataset contains 48,842 rows and 15 columns. A fixed random seed of 42 is
used for reproducibility. The code checks for values marked with "?" and
validates ranges for age, hours worked per week, capital gain, and capital
loss. These missing-value markers are converted to missing values before
processing.

The target is converted to a binary label: income above $50,000 is 1 and
income at or below $50,000 is 0. The data is split into training and test
sets using a stratified 70/30 split. Exact duplicate training rows are
removed, while the test set is left unchanged. Numeric features are clipped
to training-set 1st and 99th percentiles, imputed with the median, and
standardized. Categorical features are imputed with the most frequent value
and one-hot encoded.

A logistic regression classifier is evaluated in two versions. The baseline
uses the original features. The second version adds capital net
(capital gain minus capital loss) and hours per age
(hours per week divided by age minus 16). Both versions are evaluated on the
same test split using accuracy and F1 score.

 Results

| Model | Accuracy | F1 score |
|---|---:|---:|
| Baseline, original features | 0.8540 | 0.6601 |
| With derived features | 0.8578 | 0.6696 |

The derived-feature model improved accuracy by approximately 0.38 percentage
points and F1 score by approximately 0.95 percentage points on this test
split. This is a modest improvement; the results do not establish that the
model will perform similarly on other data.

 Data-quality and reproducibility work

The script checks for duplicate feature rows with conflicting income labels
and exports 50 randomly selected source rows for manual label review. It also
saves the processed training data and computes its SHA-256 hash. The
configuration records the dataset name, input dimensions, random seed, hash,
and evaluation metrics.

 Output files

- `results/processed_data.csv`: processed, derived-feature training data and
  target.
- `results/preprocessor.joblib`: fitted baseline preprocessing transformer.
- `results/label_quality_sample50.csv`: 50 rows selected for manual review.
- `results/config.yaml`: run metadata, hash, and model metrics.

 Conclusion and limitations

On this run, adding the two engineered features slightly improved both
reported metrics. The evaluation uses one stratified holdout split, so it
does not quantify variation across multiple splits or establish
generalization to future data. The Adult dataset includes demographic
attributes; these results alone are not evidence that the model is fair or
appropriate for real-world decisions.