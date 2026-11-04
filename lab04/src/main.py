import os
import sys
import random
import hashlib
import joblib
import yaml
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------
# 0. Seeds and Environment Setup (Global Rules)
# ---------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
try:
    import torch
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
except ImportError:
    torch = None

print("=== Environment & Library Versions ===")
print(f"Python: {sys.version.split()[0]}")
print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"joblib: {joblib.__version__}")
print(f"yaml: {yaml.__version__}")
if torch:
    print(f"torch: {torch.__version__}")
print("======================================\n")

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------
# 1 & 2. Ingestion & Problem Definition
# ---------------------------------------------------------
print("[Step 1 & 2] Loading raw Adult dataset via OpenML...")
raw_data = fetch_openml(name='adult', version=2, as_frame=True)
df_raw = raw_data.frame.rename(columns={'class': 'income'})

print(f"Raw dataset loaded. Row count: {df_raw.shape[0]}, Column count: {df_raw.shape[1]}")
assert df_raw.shape[0] == 48842, f"Expected 48842 rows, got {df_raw.shape[0]}"
assert df_raw.shape[1] == 15, f"Expected 15 columns, got {df_raw.shape[1]}"

# ---------------------------------------------------------
# 3. Validation Step
# ---------------------------------------------------------
print("\n[Step 3] Running schema, range, and missing-value checks...")
violations = []

# Check missing values represented as '?'
missing_q_counts = (df_raw == '?').sum()
for col, cnt in missing_q_counts.items():
    if cnt > 0:
        violations.append(f"Column '{col}' has {cnt} values coded as '?' (missing).")

# Range checks
range_checks = {
    'age': (17, 90),
    'hours-per-week': (1, 99),
    'capital-gain': (0, 99999),
    'capital-loss': (0, 4356)
}

for col, (min_val, max_val) in range_checks.items():
    if col in df_raw.columns:
        out_of_bounds = df_raw[(df_raw[col] < min_val) | (df_raw[col] > max_val)]
        if not out_of_bounds.empty:
            violations.append(f"Column '{col}' has {len(out_of_bounds)} values out of range [{min_val}, {max_val}].")

print("- Validation Report ---")
if violations:
    for v in violations:
        print(f" [WARNING] {v}")
else:
    print(" No validation violations found.")

# ---------------------------------------------------------
# 4. Cleaning & Stratified Train/Test Split
# ---------------------------------------------------------
print("\n[Step 4] Splitting 70/30 stratified before cleaning...")
df_clean = df_raw.copy()
df_clean.replace('?', np.nan, inplace=True)

X = df_clean.drop(columns=['income'])
y = df_clean['income'].astype(str).str.strip().str.replace('.', '', regex=False)
y_binary = (y == '>50K').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, stratify=y_binary, random_state=RANDOM_SEED
)

# Drop exact duplicates from train only (keep test untouched)
train_combined = pd.concat([X_train, y_train], axis=1)
train_dedup = train_combined.drop_duplicates()
X_train = train_dedup.drop(columns=['income'])
y_train = train_dedup['income']

print(f"Training rows after deduplication: {X_train.shape[0]}")
print(f"Test rows (untouched): {X_test.shape[0]}")

num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

# Outliers: clip numeric features to 1st/99th percentile from train
for col in num_cols:
    q01 = X_train[col].quantile(0.01)
    q99 = X_train[col].quantile(0.99)
    X_train[col] = X_train[col].clip(q01, q99)
    X_test[col] = X_test[col].clip(q01, q99)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

preprocessor.fit(X_train)
joblib.dump(preprocessor, "results/preprocessor.joblib")
print("Fitted preprocessor saved to results/preprocessor.joblib")

# ---------------------------------------------------------
# 5. Derived Features Engineering & Evaluation
# ---------------------------------------------------------
print("\n[Step 5] Engineering derived features and evaluating baseline...")

def add_derived_features(df):
    df_feat = df.copy()
    df_feat['capital_net'] = df_feat['capital-gain'] - df_feat['capital-loss']
    df_feat['hours_per_age'] = df_feat['hours-per-week'] / (df_feat['age'] - 16)
    return df_feat

X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

clf_base = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
clf_base.fit(X_train_proc, y_train)
y_pred_base = clf_base.predict(X_test_proc)

acc_base = accuracy_score(y_test, y_pred_base)
f1_base = f1_score(y_test, y_pred_base)

# With derived features
X_train_ext = add_derived_features(X_train)
X_test_ext = add_derived_features(X_test)
num_cols_ext = num_cols + ['capital_net', 'hours_per_age']

preprocessor_ext = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols_ext),
    ('cat', categorical_transformer, cat_cols)
])

X_train_ext_proc = preprocessor_ext.fit_transform(X_train_ext)
X_test_ext_proc = preprocessor_ext.transform(X_test_ext)

clf_ext = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
clf_ext.fit(X_train_ext_proc, y_train)
y_pred_ext = clf_ext.predict(X_test_ext_proc)

acc_ext = accuracy_score(y_test, y_pred_ext)
f1_ext = f1_score(y_test, y_pred_ext)

print("=== Logistic Regression Performance ===")
print(f"Without derived features: Accuracy = {acc_base:.4f}, F1 = {f1_base:.4f}")
print(f"With derived features: Accuracy = {acc_ext:.4f}, F1 = {f1_ext:.4f}")

# ---------------------------------------------------------
# 6. Label-Quality Check
# ---------------------------------------------------------
print("\n[Step 6] Running label-quality and duplicate-conflict audit...")
features_only = X.copy()
combined_for_audit = pd.concat([features_only, y_binary], axis=1)

duplicated_features = combined_for_audit[combined_for_audit.duplicated(subset=X.columns, keep=False)]
conflicts = duplicated_features.groupby(list(X.columns))['income'].nunique()
conflict_count = (conflicts > 1).sum()

print(f"Duplicate feature rows with conflicting income labels: {conflict_count}")

sample_50 = df_raw.sample(n=50, random_state=RANDOM_SEED)
sample_50.to_csv("results/label_quality_sample50.csv", index=False)
print("Saved 50 random rows for manual audit to results/label_quality_sample50.csv")

# ---------------------------------------------------------
# 7. Dataset Hash & Config YAML
# ---------------------------------------------------------
print("\n[Step 7] Computing SHA256 hash and saving configuration YAML...")

processed_df = pd.DataFrame(X_train_ext_proc)
processed_df['target'] = y_train.values
processed_path = "results/processed_data.csv"
processed_df.to_csv(processed_path, index=False)

sha256_hash = hashlib.sha256()
with open(processed_path, "rb") as f:
    for byte_block in iter(lambda: f.read(4096), b""):
        sha256_hash.update(byte_block)
hash_val = sha256_hash.hexdigest()

config_data = {
    'dataset_name': 'UCI Adult (version 2)',
    'raw_rows': int(df_raw.shape[0]),
    'raw_cols': int(df_raw.shape[1]),
    'random_seed': RANDOM_SEED,
    'processed_sha256': hash_val,
    'metrics': {
        'baseline_accuracy': float(acc_base),
        'baseline_f1': float(f1_base),
        'derived_accuracy': float(acc_ext),
        'derived_f1': float(f1_ext)
    }
}

with open("results/config.yaml", "w") as f:
    yaml.dump(config_data, f, default_flow_style=False)

print(f"Processed dataset SHA256: {hash_val}")
print("Pipeline execution complete. All artifacts stored in results/.")