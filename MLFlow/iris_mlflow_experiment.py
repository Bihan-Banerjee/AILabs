import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("Step 1: Loading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print("\nStep 2: Setting up MLflow experiment...")
mlflow.set_experiment("Iris_Classification_Models")
print("\nStep 3: Defining hyperparameters...")

lr_params = [
    {"C": 0.1, "max_iter": 100, "solver": "lbfgs"},
    {"C": 1.0, "max_iter": 200, "solver": "liblinear"},
    {"C": 10.0, "max_iter": 500, "solver": "newton-cg"}
]

svm_params = [
    {"C": 0.1, "kernel": "linear", "gamma": "scale"},
    {"C": 1.0, "kernel": "rbf", "gamma": "auto"},
    {"C": 10.0, "kernel": "poly", "degree": 3, "gamma": "scale"}
]

rf_params = [
    {"n_estimators": 50, "max_depth": 3, "random_state": 42},
    {"n_estimators": 100, "max_depth": 5, "random_state": 42},
    {"n_estimators": 200, "max_depth": None, "min_samples_split": 5, "random_state": 42}
]

def evaluate_and_log_model(model, model_name, params, X_train, X_test, y_train, y_test):
    """
    Train model, evaluate it, and log everything to MLflow
    """
    with mlflow.start_run(run_name=model_name):
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=f"iris_{model_name.lower().replace(' ', '_')}"
        )
        mlflow.set_tags({
            "model_type": model_name.split('_')[0],
            "dataset": "iris",
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        })
        
        return accuracy, model
print("\nStep 5: Training and logging all models...")

model_results = []

print("\n=== LOGISTIC REGRESSION MODELS ===")
for i, params in enumerate(lr_params, 1):
    model = LogisticRegression(**params)
    model_name = f"LogisticRegression_{i}"
    accuracy, trained_model = evaluate_and_log_model(
        model, model_name, params, X_train, X_test, y_train, y_test
    )
    model_results.append((model_name, accuracy, trained_model))

print("\n=== SVM MODELS ===")
for i, params in enumerate(svm_params, 1):
    model = SVC(**params)
    model_name = f"SVM_{i}"
    accuracy, trained_model = evaluate_and_log_model(
        model, model_name, params, X_train, X_test, y_train, y_test
    )
    model_results.append((model_name, accuracy, trained_model))

print("\n=== RANDOM FOREST MODELS ===")
for i, params in enumerate(rf_params, 1):
    model = RandomForestClassifier(**params)
    model_name = f"RandomForest_{i}"
    accuracy, trained_model = evaluate_and_log_model(
        model, model_name, params, X_train, X_test, y_train, y_test
    )
    model_results.append((model_name, accuracy, trained_model))

print("\n" + "="*60)
print("SUMMARY OF ALL MODELS")
print("="*60)
print(f"{'Model Name':<20} {'Accuracy':<10}")
print("-"*30)
for name, accuracy, _ in model_results:
    print(f"{name:<20} {accuracy:.4f}")

best_model = max(model_results, key=lambda x: x[1])
print(f"\nBest performing model: {best_model[0]} with accuracy: {best_model[1]:.4f}")
print("\n" + "="*60)
print("DEMONSTRATING MODEL LOADING FROM MLFLOW")
print("="*60)

experiment = mlflow.get_experiment_by_name("Iris_Classification_Models")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
latest_run_id = runs.iloc[0]['run_id']

print(f"Loading model from run: {latest_run_id}")

loaded_model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/model")

test_prediction = loaded_model.predict(X_test[:5])
print(f"Predictions on first 5 test samples: {test_prediction}")
print(f"Actual labels: {y_test[:5]}")

print("\n" + "="*60)
print("EXPERIMENT COMPLETED SUCCESSFULLY!")
print("="*60)