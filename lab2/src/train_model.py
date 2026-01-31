import os
import joblib
from datetime import datetime
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def load_wine_data():
    """Load the Wine dataset"""
    wine = load_wine()
    X = wine.data
    y = wine.target
    print(f"Dataset: Wine Classification")
    print(f"Features: {wine.feature_names}")
    print(f"Classes: {wine.target_names}")
    print(f"Total samples: {len(X)}")
    return X, y

def train_model():
    """Train an SVM model on Wine dataset"""
    print("Loading Wine dataset...")
    X, y = load_wine_data()
    
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("Training Support Vector Machine (SVM) Classifier...")
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the model and scaler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"svm_wine_model_{timestamp}.joblib"
    scaler_filename = f"scaler_{timestamp}.joblib"
    model_path = os.path.join(models_dir, model_filename)
    scaler_path = os.path.join(models_dir, scaler_filename)
    
    # Also save as latest model
    latest_model_path = os.path.join(models_dir, "latest_model.joblib")
    latest_scaler_path = os.path.join(models_dir, "latest_scaler.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(model, latest_model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(scaler, latest_scaler_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Latest model saved to: {latest_model_path}")
    
    return model, scaler

if __name__ == "__main__":
    train_model()