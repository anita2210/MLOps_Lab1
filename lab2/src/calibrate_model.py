import os
import joblib
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

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

def calibrate_model():
    """Calibrate the trained model using Platt scaling"""
    # Load the latest model
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(models_dir, "latest_model.joblib")
    
    if not os.path.exists(model_path):
        print("Error: No trained model found. Please run train_model.py first.")
        return
    
    print("Loading the trained model...")
    model = joblib.load(model_path)
    
    print("Generating calibration data...")
    X, y = generate_synthetic_data()
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.3, random_state=123)
    
    print("Calibrating model using Platt scaling (sigmoid)...")
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    # Save the calibrated model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    calibrated_filename = f"calibrated_model_{timestamp}.joblib"
    calibrated_path = os.path.join(models_dir, calibrated_filename)
    
    # Also save as latest calibrated model
    latest_calibrated_path = os.path.join(models_dir, "latest_calibrated_model.joblib")
    
    joblib.dump(calibrated_model, calibrated_path)
    joblib.dump(calibrated_model, latest_calibrated_path)
    
    print(f"Calibrated model saved to: {calibrated_path}")
    print(f"Latest calibrated model saved to: {latest_calibrated_path}")
    
    return calibrated_model

if __name__ == "__main__":
    calibrate_model()