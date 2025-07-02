import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

# Load processed features (feature-engineered data)
processed_df = pd.read_csv('data/processed/processed_data.csv')

# Load RFM DataFrame with is_high_risk label
rfm = pd.read_csv('data/processed/rfm_with_risk.csv')

# Merge is_high_risk into processed features
merged_df = processed_df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

# Save merged data for modeling
merged_df.to_csv('data/processed/model_ready_data.csv', index=False)

# Prepare features and target
y = merged_df['is_high_risk']
X = merged_df.drop(['CustomerId', 'is_high_risk'], axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models and parameter grids
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42)
}
param_grids = {
    "LogisticRegression": {"C": [0.1, 1, 10]},
    "RandomForest": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
}

best_estimators = {}

mlflow.set_experiment("credit_scoring")

for name, model in models.items():
    print(f"\nTraining {name}...")
    grid = GridSearchCV(model, param_grids[name], cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_estimators[name] = grid.best_estimator_
    print(f"{name} best params: {grid.best_params_}")

    # Evaluate
    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1] if hasattr(grid, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC: {roc_auc:.4f}")

    # MLflow logging
    input_example = X_test.iloc[:5]
    with mlflow.start_run(run_name=name):
        mlflow.sklearn.log_model(
            grid.best_estimator_,
            name,
            input_example=input_example
        )
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        if roc_auc is not None:
            mlflow.log_metric("roc_auc", roc_auc)

print("\nAll models trained and logged to MLflow.")