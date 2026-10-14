import of os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Set random seed as required by lab rules
np.random.seed(42)

# 1. Load Breast Cancer dataset
X, y = load_breast_cancer(return_X_y=True)

# 2. Split 70/30 stratified with random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 3. Train Logistic Regression baseline
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_acc = lr_model.score(X_test, y_test)

# 4. Train Random Forest baseline
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = rf_model.score(X_test, y_test)

# 5. Save test accuracies to results/baseline_accuracy.csv rounded to 4 decimal
os.makedirs("lab01/results", exist_ok=True)
results_df = pd.DataFrame({
    "Model": ["LogisticRegression", "RandomForestClassifier"],
    "Test_Accuracy": [round(lr_acc, 4), round(rf_acc, 4)]
})
results_df.to_csv("lab01/results/baseline_accuracy.csv", index=False)
print("Baseline accuracies saved successfully!")