# 📘 Module 1: Scikit-learn Fundamentals — Learning Notes

> **Goal**: Understand the basics of machine learning and the Scikit-learn ecosystem.

---

## 📌 Table of Contents

1. [Introduction to Machine Learning & Scikit-learn](#1-introduction-to-machine-learning--scikit-learn)
2. [Scikit-learn Architecture & API](#2-scikit-learn-architecture--api)
3. [Understanding Datasets](#3-understanding-datasets)
4. [Scikit-learn Workflow](#4-scikit-learn-workflow)
5. [Types of Machine Learning](#5-types-of-machine-learning)
6. [Estimators vs Transformers vs Predictors](#6-estimators-vs-transformers-vs-predictors)
7. [Interview Questions & Cheat Sheet](#7-interview-questions--cheat-sheet)

---

## 1. Introduction to Machine Learning & Scikit-learn

### What is Machine Learning?
Machine Learning is the science of getting computers to learn and act like humans do — by feeding them data and letting them learn from patterns instead of explicitly programming rules.

### What is Scikit-learn?
- **Imported as**: `sklearn`
- **Purpose**: The industry-standard Python library for classical/traditional machine learning.
- **Built on**: NumPy, SciPy, Matplotlib.
- **Why use it?**: It has a simple, consistent API, great documentation, and covers classification, regression, clustering, and preprocessing.

---

## 2. Scikit-learn Architecture & API

### The Golden Design Principles
1. **Consistency**: All objects share a common interface (`fit`, `predict`, `transform`).
2. **Inspection**: All model parameters are exposed as public attributes.
3. **Composition**: Easy to chain steps together using `Pipeline`.
4. **Sensible Defaults**: Models work reasonably well right out of the box.

### Key Modules
```python
from sklearn.datasets import load_iris              # Toy datasets
from sklearn.model_selection import train_test_split # Splitting data
from sklearn.preprocessing import StandardScaler    # Preprocessing
from sklearn.linear_model import LogisticRegression # Models
from sklearn.metrics import accuracy_score          # Evaluation
from sklearn.pipeline import Pipeline               # Composition
```

---

## 3. Understanding Datasets

Every machine learning problem in Scikit-learn revolves around two fundamental objects:

| Variable | Name | Shape | Description |
|----------|------|-------|-------------|
| **X** | Features | `(n_samples, n_features)` | The input data (2D array/matrix). What the model learns from. |
| **y** | Target | `(n_samples,)` | The output data (1D array/vector). What the model predicts. |

### Terminology
- **Sample**: One row (one observation/data point).
- **Feature**: One column (one attribute/measurement).

### Code Example
```python
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

print(f"X shape (Features): {X.shape}")  # Output: (150, 4) -> 150 samples, 4 features
print(f"y shape (Target): {y.shape}")    # Output: (150,) -> 150 target labels
```

---

## 4. Scikit-learn Workflow

### The 4 Core Methods
1. `fit(X, y)`: The model learns the patterns from the training data.
2. `predict(X)`: The model makes predictions on new, unseen data.
3. `transform(X)`: Changes/preprocesses the data (used by scalers/encoders).
4. `score(X, y)`: Evaluates how well the model performed.

### End-to-End Workflow Code
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Load Data
X, y = load_iris(return_X_y=True)

# 2. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Preprocess Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # FIT and TRANSFORM training data
X_test_scaled = scaler.transform(X_test)        # ONLY TRANSFORM test data (Prevents Leakage)

# 4. Train Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)              # Model learns

# 5. Make Predictions
y_pred = model.predict(X_test_scaled)           # Model predicts

# 6. Evaluate
accuracy = model.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy * 100}%")
```

---

## 5. Types of Machine Learning

### 1. Supervised Learning (Data has labels)
- **Classification**: Predicting discrete categories (e.g., Spam vs. Not Spam, Cat vs. Dog).
  - *Algorithms*: Logistic Regression, Support Vector Machines (SVM), Random Forest Classifier.
- **Regression**: Predicting continuous numbers (e.g., House Prices, Temperature).
  - *Algorithms*: Linear Regression, Ridge, Random Forest Regressor.

### 2. Unsupervised Learning (Data has NO labels)
- **Clustering**: Grouping similar data points together.
  - *Algorithms*: K-Means, DBSCAN.
- **Dimensionality Reduction**: Reducing the number of features while keeping essential info.
  - *Algorithms*: PCA (Principal Component Analysis).

---

## 6. Estimators vs Transformers vs Predictors

In Scikit-learn, every object plays a specific role:

| Object Type | Has `.fit()`? | Has `.transform()`? | Has `.predict()`? | Example |
|-------------|---------------|---------------------|-------------------|---------|
| **Transformer** | ✅ Yes | ✅ Yes | ❌ No | `StandardScaler`, `PCA` |
| **Predictor** | ✅ Yes | ❌ No | ✅ Yes | `LogisticRegression`, `RandomForest` |
| **Pipeline** | ✅ Yes | ✅ Yes | ✅ Yes | `Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])` |

---

## 7. Interview Questions & Cheat Sheet

### Top 5 Q&A

**Q1: What are X and y in Scikit-learn?**
> **A:** `X` is the 2D feature matrix (inputs), and `y` is the 1D target vector (outputs/labels).

**Q2: What is the difference between `fit()`, `transform()`, and `fit_transform()`?**
> **A:** `fit()` calculates the parameters (like mean/std). `transform()` applies those parameters to change the data. `fit_transform()` does both in a single, optimized step. 

**Q3: Why must we NEVER call `fit()` on test data?**
> **A:** It causes **data leakage**. The model/scaler would "peek" at the test data, resulting in artificially high accuracy during testing, but the model will fail in real-world production. You must only `fit()` on training data, and `transform()` on test data.

**Q4: What is the difference between Classification and Regression?**
> **A:** Classification predicts categories (discrete), while Regression predicts quantities/numbers (continuous).

**Q5: What is a Scikit-learn Pipeline?**
> **A:** A tool that chains multiple transformers and a final predictor into a single object. It ensures training and testing data undergo the exact same steps and prevents data leakage.

---

### ✅ Module 1 Checklist
- [x] Understand Scikit-learn basics and API rules.
- [x] Differentiate between `X` (Features) and `y` (Target).
- [x] Write a complete `fit`, `predict`, `score` workflow.
- [x] Prevent Data Leakage during train/test split.
- [x] Differentiate Supervised (Classification/Regression) vs. Unsupervised Learning.
- [x] Understand Transformers vs. Predictors.