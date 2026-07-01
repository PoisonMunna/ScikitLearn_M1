# preprocessing/data_transformer.py
"""
Estimators vs Transformers vs Predictors
Learn to create custom transformers
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature scaling.
    Demonstrates the Transformer concept in Scikit-learn.
    """
    
    def __init__(self, method='standard'):
        """
        Parameters:
        -----------
        method : str, default='standard'
            'standard' for StandardScaler, 'minmax' for MinMaxScaler
        """
        self.method = method
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None
        
    def fit(self, X, y=None):
        """
        Fit the transformer on training data.
        Learn parameters (mean, std, min, max) from X.
        
        Understanding fit(), transform(), score()
        """
        if self.method == 'standard':
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
        elif self.method == 'minmax':
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
        return self
    
    def transform(self, X):
        """
        Transform new data using learned parameters.
        This is where the actual transformation happens.
        """
        if self.method == 'standard':
            return (X - self.mean_) / (self.std_ + 1e-8)
        elif self.method == 'minmax':
            return (X - self.min_) / (self.max_ - self.min_ + 1e-8)
        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to select specific features.
    Demonstrates composition of transformers.
    """
    
    def __init__(self, feature_indices=None):
        self.feature_indices = feature_indices
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.feature_indices is not None:
            return X[:, self.feature_indices]
        return X

class DataPreprocessor:
    """
    Complete preprocessing pipeline combining multiple transformers.
    """
    
    def __init__(self, scale_method='standard', selected_features=None):
        self.scaler = FeatureScaler(method=scale_method)
        self.selector = FeatureSelector(feature_indices=selected_features)
        
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        X_selected = self.selector.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_selected)
        return X_scaled
    
    def transform(self, X):
        """Transform new data"""
        X_selected = self.selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)
        return X_scaled

# Test the transformers
if __name__ == "__main__":
    # Test data
    X_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Test FeatureScaler
    scaler = FeatureScaler(method='standard')
    scaler.fit(X_test)
    X_scaled = scaler.transform(X_test)
    print("Original:\n", X_test)
    print("\nScaled (mean=0, std=1):\n", X_scaled)
    
    # Test FeatureSelector
    selector = FeatureSelector(feature_indices=[0, 2])
    X_selected = selector.transform(X_test)
    print("\nSelected features (0 and 2):\n", X_selected)