# data/iris_dataset.py
"""
Understanding Datasets
Learn about: Samples, Features, Target, X, y
"""

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

class IrisDataset:
    def __init__(self):
        """Load and prepare Iris dataset"""
        self.data = load_iris()
        self.X = self.data.data          # Features (samples × features)
        self.y = self.data.target        # Target labels
        self.feature_names = self.data.feature_names
        self.target_names = self.data.target_names
        
    def explore_dataset(self):
        """Print dataset information"""
        print("="*50)
        print("📊 IRIS DATASET EXPLORATION")
        print("="*50)
        
        # Chapter 3: Understanding Dataset Structure
        print(f"\n📌 Dataset Shape:")
        print(f"   - Total Samples: {self.X.shape[0]}")
        print(f"   - Features per Sample: {self.X.shape[1]}")
        
        print(f"\n📌 Features (X):")
        for i, name in enumerate(self.feature_names):
            print(f"   - {name}: {self.X[:, i].min():.2f} to {self.X[:, i].max():.2f}")
        
        print(f"\n📌 Target Classes (y):")
        for i, name in enumerate(self.target_names):
            count = np.sum(self.y == i)
            print(f"   - {name} (class {i}): {count} samples")
        
        # Create DataFrame for better visualization
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df['species'] = self.y
        df['species_name'] = df['species'].map(
            {i: name for i, name in enumerate(self.target_names)}
        )
        
        print(f"\n📌 First 5 Samples:")
        print(df.head())
        
        return df
    
    def get_data(self):
        """Return X and y arrays"""
        return self.X, self.y

# Example usage
if __name__ == "__main__":
    dataset = IrisDataset()
    df = dataset.explore_dataset()