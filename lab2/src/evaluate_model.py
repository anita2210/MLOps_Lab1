import os
import json
import joblib
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def generate_synthetic_data():
    """Generate synthetic classification data"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    return X, y

def evaluate_model():
    """Evaluate the trained model"""
    # Load the latest model
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(models_dir, "latest_model.joblib")
    
    if not os.path.exists(model_path):
        print("Error: No trained model found. Please run train_model.py first.")
        return
    
    print("Loading the trained model...")
    model = joblib.load(model_path)
    
    print("Generating test data...")
    X, y = generate_synthetic_data()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4)
    }
    
    print("\n===== Model Evaluation Results =====")
    print(f"F1 Score:  {metrics['f1_score']}")
    print(f"Accuracy:  {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall:    {metrics['recall']}")
    print("====================================\n")
    
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