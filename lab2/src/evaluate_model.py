import os
import json
import joblib
from datetime import datetime
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

def load_wine_data():
    """Load the Wine dataset"""
    wine = load_wine()
    return wine.data, wine.target, wine.target_names

def evaluate_model():
    """Evaluate the trained SVM model"""
    # Load the latest model and scaler
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(models_dir, "latest_model.joblib")
    scaler_path = os.path.join(models_dir, "latest_scaler.joblib")
    
    if not os.path.exists(model_path):
        print("Error: No trained model found. Please run train_model.py first.")
        return
    
    print("Loading the trained SVM model...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print("Loading Wine test data...")
    X, y, target_names = load_wine_data()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the test data
    X_test_scaled = scaler.transform(X_test)
    
    print("Making predictions...")
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Support Vector Machine (SVM)",
        "dataset": "Wine",
        "f1_score_weighted": round(f1_score(y_test, y_pred, average='weighted'), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_weighted": round(precision_score(y_test, y_pred, average='weighted'), 4),
        "recall_weighted": round(recall_score(y_test, y_pred, average='weighted'), 4)
    }
    
    print("\n========== Model Evaluation Results ==========")
    print(f"Model:      {metrics['model']}")
    print(f"Dataset:    {metrics['dataset']}")
    print(f"Accuracy:   {metrics['accuracy']}")
    print(f"F1 Score:   {metrics['f1_score_weighted']}")
    print(f"Precision:  {metrics['precision_weighted']}")
    print(f"Recall:     {metrics['recall_weighted']}")
    print("===============================================\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Save metrics to file
    metrics_dir = os.path.join(os.path.dirname(__file__), '..', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_path = os.path.join(metrics_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to: {metrics_path}")
    
    return metrics

if __name__ == "__main__":
    evaluate_model()