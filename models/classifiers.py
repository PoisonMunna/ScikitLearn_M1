# models/classifiers.py
"""
Estimators vs Transformers vs Predictors
Learning about estimators, predictors, and their methods
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

class IrisClassifier:
    """
    Wrapper class for different classifiers.
    Demonstrates the Estimator/Predictor concepts.
    """
    
    def __init__(self, model_type='knn', **kwargs):
        """
        Initialize the appropriate classifier.
        
        Estimators are objects that learn from data.
        """
        self.model_type = model_type
        self.model = self._create_model(model_type, **kwargs)
        self.is_fitted = False
        
    def _create_model(self, model_type, **kwargs):
        """Factory method to create different models"""
        models = {
            'knn': KNeighborsClassifier,
            'svm': SVC,
            'decision_tree': DecisionTreeClassifier,
            'random_forest': RandomForestClassifier
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Set default parameters
        default_params = {
            'knn': {'n_neighbors': 5},
            'svm': {'kernel': 'rbf', 'random_state': 42},
            'decision_tree': {'random_state': 42},
            'random_forest': {'n_estimators': 100, 'random_state': 42}
        }
        
        params = default_params.get(model_type, {})
        params.update(kwargs)
        
        return models[model_type](**params)
    
    def fit(self, X, y):
        """
        Train the model.
        fit() learns patterns from training data.
        """
        print(f"\n🚀 Training {self.model_type.upper()} classifier...")
        self.model.fit(X, y)
        self.is_fitted = True
        print("✅ Training complete!")
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        predict() uses learned patterns to make predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction!")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities (if available).
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        score() evaluates model performance.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring!")
        return self.model.score(X, y)
    
    def evaluate(self, X_test, y_test):
        """
        Comprehensive model evaluation.
        """
        y_pred = self.predict(X_test)
        
        print(f"\n📊 {self.model_type.upper()} Performance Report:")
        print("-" * 50)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return y_pred
    
    def get_params(self, deep=True):
        """Get model parameters"""
        return self.model.get_params(deep=deep)
    
    def set_params(self, **params):
        """Set model parameters"""
        self.model.set_params(**params)
        return self

# Compare multiple models
class ModelComparator:
    """
    Compare different classifiers' performance.
    """
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = {}
        
    def compare_models(self, model_types=None):
        """
        Train and compare multiple models.
        """
        if model_types is None:
            model_types = ['knn', 'svm', 'decision_tree', 'random_forest']
        
        print("="*60)
        print("🔍 MODEL COMPARISON")
        print("="*60)
        
        for model_type in model_types:
            # Create and train classifier
            classifier = IrisClassifier(model_type=model_type)
            classifier.fit(self.X_train, self.y_train)
            
            # Evaluate
            train_score = classifier.score(self.X_train, self.y_train)
            test_score = classifier.score(self.X_test, self.y_test)
            
            self.results[model_type] = {
                'train_score': train_score,
                'test_score': test_score,
                'model': classifier
            }
            
            print(f"\n{model_type.upper()}:")
            print(f"  Train Accuracy: {train_score:.4f}")
            print(f"  Test Accuracy:  {test_score:.4f}")
            print(f"  Overfitting Gap: {train_score - test_score:.4f}")
        
        return self.results
    
    def get_best_model(self):
        """Return the best performing model"""
        best_model = max(self.results, key=lambda x: self.results[x]['test_score'])
        return self.results[best_model]['model']