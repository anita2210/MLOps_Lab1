import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class TestWineSVMModel(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        wine = load_wine()
        self.X = wine.data
        self.y = wine.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def test_data_loading(self):
        """Test if Wine dataset loads correctly"""
        self.assertEqual(len(self.X), 178)
        self.assertEqual(self.X.shape[1], 13)
    
    def test_model_training(self):
        """Test if SVM model trains without errors"""
        model = SVC(kernel='rbf', C=1.0, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        self.assertTrue(hasattr(model, 'support_vectors_'))
    
    def test_model_prediction(self):
        """Test if model makes predictions"""
        model = SVC(kernel='rbf', C=1.0, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        predictions = model.predict(self.X_test_scaled)
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_model_accuracy(self):
        """Test if model accuracy is above 90%"""
        model = SVC(kernel='rbf', C=1.0, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        accuracy = model.score(self.X_test_scaled, self.y_test)
        self.assertGreater(accuracy, 0.90)

if __name__ == '__main__':
    unittest.main()