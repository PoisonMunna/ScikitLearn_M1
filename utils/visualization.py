# utils/visualization.py
"""
Visualization utilities for data exploration and model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

class IrisVisualizer:
    """Handles all visualization for the project"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_feature_distributions(self, X, feature_names, target_names, y=None):
        """
        Plot feature distributions and relationships.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Distribution plots
        for i, ax in enumerate(axes[0]):
            ax.hist(X[:, i], bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax.set_title(f'Distribution: {feature_names[i]}')
            ax.set_xlabel(feature_names[i])
            ax.set_ylabel('Frequency')
        
        # Scatter plots (first 2 features)
        ax = axes[1, 0]
        if y is not None:
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='black')
            ax.legend(*scatter.legend_elements(), title='Species')
        else:
            ax.scatter(X[:, 0], X[:, 1], alpha=0.6)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title('Feature 0 vs Feature 1')
        
        # Scatter plots (features 2 and 3)
        ax = axes[1, 1]
        if y is not None:
            scatter = ax.scatter(X[:, 2], X[:, 3], c=y, cmap='viridis', edgecolor='black')
        else:
            ax.scatter(X[:, 2], X[:, 3], alpha=0.6)
        ax.set_xlabel(feature_names[2])
        ax.set_ylabel(feature_names[3])
        ax.set_title('Feature 2 vs Feature 3')
        
        # Correlation heatmap
        ax = axes[1, 2]
        corr_matrix = np.corrcoef(X.T)
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(feature_names)))
        ax.set_yticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_yticklabels(feature_names)
        ax.set_title('Feature Correlation Matrix')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()
    
    def plot_decision_boundary(self, model, X, y, feature_names, target_names):
        """
        Visualize decision boundaries using PCA.
        """
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create mesh grid
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        
        # Predict on mesh grid
        # Need to inverse transform mesh points
        mesh_points = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
        Z = model.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='black')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Decision Boundaries (PCA Projection)')
        plt.legend(*scatter.legend_elements(), title='Species')
        plt.colorbar(scatter)
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, target_names):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    
    def plot_model_comparison(self, results):
        """Compare model accuracies"""
        model_names = list(results.keys())
        train_scores = [results[m]['train_score'] for m in model_names]
        test_scores = [results[m]['test_score'] for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, train_scores, width, label='Train', color='blue', alpha=0.7)
        rects2 = ax.bar(x + width/2, test_scores, width, label='Test', color='green', alpha=0.7)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        
        # Add value labels on bars
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()