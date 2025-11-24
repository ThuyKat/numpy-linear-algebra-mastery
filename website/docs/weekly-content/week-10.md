---
title: Week 10 - Principal Component Analysis (PCA)
sidebar_label: Week 10 - PCA
---

# Week 10: Principal Component Analysis (PCA)

## Time Allocation
- **Total Hours**: 14 hours
- **Theory**: 5 hours
- **NumPy Coding**: 5 hours
- **Project**: 4 hours

## Learning Objectives
- Understand PCA and its mathematical foundation
- Implement PCA from scratch using SVD and eigendecomposition
- Apply PCA for dimensionality reduction
- Visualize high-dimensional data

## Theory (5 hours)

### Resources
1. **StatQuest**: [PCA Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ)
2. **3Blue1Brown**: Change of basis
3. **MIT OCW**: Lecture on PCA and covariance matrices

### Core Concepts
- **Goal**: Find directions of maximum variance
- **Steps**: Center data → Compute covariance → Find eigenvectors
- **Principal Components**: Eigenvectors of covariance matrix
- **Variance Explained**: Eigenvalues show importance
- Connection to SVD: V from SVD = PCs

## NumPy Coding (5 hours)

```python
import numpy as np
import matplotlib.pyplot as plt

class PCA:
    """Principal Component Analysis from scratch"""

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None

    def fit(self, X):
        """Fit PCA on data X"""
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        cov = np.cov(X_centered.T)

        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store principal components
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]

        return self

    def transform(self, X):
        """Transform data to PC space"""
        X_centered = X - self.mean
        return X_centered @ self.components

    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """Transform back to original space"""
        return (X_transformed @ self.components.T) + self.mean

# Generate sample data
np.random.seed(42)
mean = [0, 0]
cov = [[3, 1.5], [1.5, 1]]
X = np.random.multivariate_normal(mean, cov, 300)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Original data with PCs
ax1.scatter(X[:, 0], X[:, 1], alpha=0.5)
for i in range(2):
    pc = pca.components[:, i]
    ax1.arrow(0, 0, pc[0]*3, pc[1]*3, head_width=0.2,
             head_length=0.2, fc=f'C{i+1}', ec=f'C{i+1}',
             label=f'PC{i+1}', linewidth=2)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_title('Original Data with Principal Components')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Transformed data
ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_title('Data in PC Space')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()

# Variance explained
print(f"Variance explained by PC1: {pca.explained_variance[0]:.4f}")
print(f"Variance explained by PC2: {pca.explained_variance[1]:.4f}")
total_var = sum(pca.explained_variance)
print(f"Total variance explained: {total_var:.4f}")
```

## Project: Dimensionality Reduction on Real Data (4 hours)

```python
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load digits dataset (64 dimensions)
digits = load_digits()
X = digits.data
y = digits.target

print(f"Original shape: {X.shape}")

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Visualize
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                     c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='Digit')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Digits Dataset: 64D → 2D using PCA')
plt.grid(True, alpha=0.3)
plt.show()

# Scree plot (variance explained)
pca_full = PCA(n_components=20)
pca_full.fit(X)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), pca_full.explained_variance, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Scree Plot')
plt.grid(True, alpha=0.3)
plt.show()
```

## Weekly Checkpoint
- ✅ Implement PCA from scratch
- ✅ Understand variance explained
- ✅ Apply PCA to real datasets
- ✅ Visualize high-dimensional data

## Resources
- [StatQuest - PCA](https://www.youtube.com/watch?v=FgakZw6K1QQ)
- [sklearn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

## Next Week Preview
Week 11 begins advanced topics with QR decomposition and numerical methods.
