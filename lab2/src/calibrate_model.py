import os
import joblib
from datetime import datetime
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

def load_wine_data():
    """Load the Wine dataset"""
    wine = load_wine()
    return wine.data, wine.target

def calibrate_model():
    """Calibrate the trained SVM model using Platt scaling"""
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
    
    print("Loading Wine calibration data...")
    X, y = load_wine_data()
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.3, random_state=123)
    
    # Scale the data
    X_train_scaled = scaler.transform(X_train)
    
    print("Calibrating SVM model using Platt scaling (sigmoid)...")
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    calibrated_model.fit(X_train_scaled, y_train)
    
    # Save the calibrated model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    calibrated_filename = f"calibrated_svm_wine_{timestamp}.joblib"
    calibrated_path = os.path.join(models_dir, calibrated_filename)
    
    # Also save as latest calibrated model
    latest_calibrated_path = os.path.join(models_dir, "latest_calibrated_model.joblib")
    
    joblib.dump(calibrated_model, calibrated_path)
    joblib.dump(calibrated_model, latest_calibrated_path)
    
    print(f"\nCalibrated model saved to: {calibrated_path}")
    print(f"Latest calibrated model saved to: {latest_calibrated_path}")
    
    return calibrated_model

if __name__ == "__main__":
    calibrate_model()