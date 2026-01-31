import os
import joblib
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

def train_model():
    """Train a Random Forest model on synthetic data"""
    print("Generating synthetic data...")
    X, y = generate_synthetic_data()
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_{timestamp}.joblib"
    model_path = os.path.join(models_dir, model_filename)
    
    # Also save as latest model
    latest_model_path = os.path.join(models_dir, "latest_model.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(model, latest_model_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Latest model saved to: {latest_model_path}")
    
    return model

if __name__ == "__main__":
    train_model()