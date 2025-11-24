---
title: Week 4 - Norms, Distance & Orthogonality
sidebar_label: Week 4 - Norms, Distance & Orthogonality
---

# Week 4: Norms, Distance & Orthogonality

## Time Allocation
- **Total Hours**: 12 hours
- **Theory**: 4 hours
- **NumPy Coding**: 4 hours
- **Project**: 4 hours

## Learning Objectives
- Understand vector norms (L1, L2, L∞)
- Learn distance metrics and their applications
- Master orthogonality and orthonormal vectors
- Apply these concepts to machine learning problems

## Theory (4 hours)

### Resources
1. **3Blue1Brown - Essence of Linear Algebra**
   - Review Chapter 9: Dot products and duality
   - Focus on geometric interpretations

2. **Khan Academy**
   - Vector norms
   - Distance formulas
   - Orthogonal complements

### Core Concepts

#### Vector Norms
- **L1 Norm (Manhattan)**: ||x||₁ = Σ|xᵢ|
- **L2 Norm (Euclidean)**: ||x||₂ = √(Σxᵢ²)
- **L∞ Norm (Max)**: ||x||∞ = max|xᵢ|
- **P-Norm**: ||x||ₚ = (Σ|xᵢ|ᵖ)^(1/p)

#### Distance Metrics
- Euclidean distance
- Manhattan distance
- Cosine distance
- Applications in ML: KNN, clustering

#### Orthogonality
- Orthogonal vectors: a · b = 0
- Orthonormal vectors: orthogonal + unit length
- Orthogonal matrices: QᵀQ = I
- Gram-Schmidt process

## NumPy Coding Exercises (4 hours)

### Vector Norms

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a vector
v = np.array([3, 4])

# Calculate different norms
l1_norm = np.linalg.norm(v, ord=1)
l2_norm = np.linalg.norm(v, ord=2)
linf_norm = np.linalg.norm(v, ord=np.inf)

print(f"Vector: {v}")
print(f"L1 norm (Manhattan): {l1_norm}")
print(f"L2 norm (Euclidean): {l2_norm}")
print(f"L∞ norm (Max): {linf_norm}")

# Manual calculations to verify
l1_manual = np.sum(np.abs(v))
l2_manual = np.sqrt(np.sum(v**2))
linf_manual = np.max(np.abs(v))

print(f"\nVerification:")
print(f"L1 match: {np.isclose(l1_norm, l1_manual)}")
print(f"L2 match: {np.isclose(l2_norm, l2_manual)}")
print(f"L∞ match: {np.isclose(linf_norm, linf_manual)}")
```

### Visualizing Different Norms

```python
# Visualize unit circles for different norms
theta = np.linspace(0, 2*np.pi, 1000)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# L1 norm (diamond)
l1_x = np.sign(np.cos(theta)) * np.abs(np.cos(theta))
l1_y = np.sign(np.sin(theta)) * (1 - np.abs(l1_x))
axes[0].plot(l1_x, l1_y, 'b-', linewidth=2)
axes[0].set_title('L1 Norm (Manhattan)\n||x||₁ = 1')
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')
axes[0].set_xlim(-1.5, 1.5)
axes[0].set_ylim(-1.5, 1.5)

# L2 norm (circle)
l2_x = np.cos(theta)
l2_y = np.sin(theta)
axes[1].plot(l2_x, l2_y, 'r-', linewidth=2)
axes[1].set_title('L2 Norm (Euclidean)\n||x||₂ = 1')
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')
axes[1].set_xlim(-1.5, 1.5)
axes[1].set_ylim(-1.5, 1.5)

# L∞ norm (square)
linf_x = np.concatenate([np.linspace(-1, 1, 100), np.ones(100),
                         np.linspace(1, -1, 100), -np.ones(100)])
linf_y = np.concatenate([-np.ones(100), np.linspace(-1, 1, 100),
                         np.ones(100), np.linspace(1, -1, 100)])
axes[2].plot(linf_x, linf_y, 'g-', linewidth=2)
axes[2].set_title('L∞ Norm (Max)\n||x||∞ = 1')
axes[2].grid(True, alpha=0.3)
axes[2].set_aspect('equal')
axes[2].set_xlim(-1.5, 1.5)
axes[2].set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()
```

### Distance Metrics

```python
def calculate_distances(point1, point2):
    """Calculate various distance metrics between two points"""
    # Euclidean distance (L2)
    euclidean = np.linalg.norm(point1 - point2)

    # Manhattan distance (L1)
    manhattan = np.linalg.norm(point1 - point2, ord=1)

    # Chebyshev distance (L∞)
    chebyshev = np.linalg.norm(point1 - point2, ord=np.inf)

    # Cosine distance
    cosine_sim = np.dot(point1, point2) / (
        np.linalg.norm(point1) * np.linalg.norm(point2)
    )
    cosine_dist = 1 - cosine_sim

    return {
        'euclidean': euclidean,
        'manhattan': manhattan,
        'chebyshev': chebyshev,
        'cosine': cosine_dist
    }

# Test with sample points
p1 = np.array([1, 2, 3])
p2 = np.array([4, 5, 6])

distances = calculate_distances(p1, p2)
print("Distances between [1,2,3] and [4,5,6]:")
for metric, value in distances.items():
    print(f"  {metric.capitalize()}: {value:.4f}")

# Visualize different distance metrics in 2D
fig, ax = plt.subplots(figsize=(10, 10))

p1_2d = np.array([1, 1])
p2_2d = np.array([5, 4])

# Plot points
ax.plot(*p1_2d, 'ro', markersize=10, label='Point 1')
ax.plot(*p2_2d, 'bo', markersize=10, label='Point 2')

# Euclidean distance (straight line)
ax.plot([p1_2d[0], p2_2d[0]], [p1_2d[1], p2_2d[1]],
        'r-', linewidth=2, label='Euclidean')

# Manhattan distance (grid path)
ax.plot([p1_2d[0], p2_2d[0]], [p1_2d[1], p1_2d[1]],
        'g-', linewidth=2)
ax.plot([p2_2d[0], p2_2d[0]], [p1_2d[1], p2_2d[1]],
        'g-', linewidth=2, label='Manhattan')

ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.legend()
ax.set_title('Distance Metrics Visualization')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```

### Orthogonality

```python
# Check if vectors are orthogonal
def are_orthogonal(v1, v2, tolerance=1e-10):
    """Check if two vectors are orthogonal"""
    return np.abs(np.dot(v1, v2)) < tolerance

# Orthogonal vectors
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])

print("Standard basis vectors:")
print(f"v1 ⊥ v2? {are_orthogonal(v1, v2)}")
print(f"v1 ⊥ v3? {are_orthogonal(v1, v3)}")
print(f"v2 ⊥ v3? {are_orthogonal(v2, v3)}")

# Create orthonormal basis
def normalize(v):
    """Normalize a vector to unit length"""
    return v / np.linalg.norm(v)

v1_norm = normalize(v1)
v2_norm = normalize(v2)

print(f"\nv1 normalized: {v1_norm}, norm: {np.linalg.norm(v1_norm)}")
print(f"v2 normalized: {v2_norm}, norm: {np.linalg.norm(v2_norm)}")
```

### Gram-Schmidt Process

```python
def gram_schmidt(vectors):
    """
    Gram-Schmidt process to create orthonormal basis
    Input: list of linearly independent vectors
    Output: list of orthonormal vectors
    """
    orthonormal = []

    for v in vectors:
        # Start with the current vector
        u = v.copy().astype(float)

        # Subtract projections onto all previous orthonormal vectors
        for basis_vec in orthonormal:
            projection = np.dot(u, basis_vec) * basis_vec
            u = u - projection

        # Normalize
        u = u / np.linalg.norm(u)
        orthonormal.append(u)

    return orthonormal

# Test with non-orthogonal vectors
v1 = np.array([1, 1, 0])
v2 = np.array([1, 0, 1])
v3 = np.array([0, 1, 1])

original_vectors = [v1, v2, v3]
orthonormal_vectors = gram_schmidt(original_vectors)

print("Original vectors:")
for i, v in enumerate(original_vectors):
    print(f"  v{i+1}: {v}")

print("\nOrthonormal vectors:")
for i, v in enumerate(orthonormal_vectors):
    print(f"  u{i+1}: {v}, norm: {np.linalg.norm(v):.6f}")

# Verify orthogonality
print("\nOrthogonality check:")
for i in range(len(orthonormal_vectors)):
    for j in range(i+1, len(orthonormal_vectors)):
        dot = np.dot(orthonormal_vectors[i], orthonormal_vectors[j])
        print(f"  u{i+1} · u{j+1} = {dot:.10f}")
```

### Orthogonal Matrices

```python
# Create an orthogonal matrix (rotation)
theta = np.pi / 4
Q = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

print("Orthogonal matrix Q (rotation):")
print(Q)

# Verify Q^T Q = I
QTQ = Q.T @ Q
print(f"\nQ^T Q:")
print(QTQ)
print(f"Is identity? {np.allclose(QTQ, np.eye(2))}")

# Orthogonal matrices preserve length
v = np.array([3, 4])
Qv = Q @ v

print(f"\nOriginal vector: {v}, norm: {np.linalg.norm(v)}")
print(f"After Q transform: {Qv}, norm: {np.linalg.norm(Qv)}")
print(f"Length preserved? {np.isclose(np.linalg.norm(v), np.linalg.norm(Qv))}")
```

## Project: K-Nearest Neighbors from Scratch (4 hours)

Implement KNN algorithm using distance metrics.

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class KNN:
    """K-Nearest Neighbors classifier using various distance metrics"""

    def __init__(self, k=3, metric='euclidean'):
        """
        k: number of neighbors
        metric: 'euclidean', 'manhattan', or 'cosine'
        """
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store training data"""
        self.X_train = X
        self.y_train = y

    def _calculate_distance(self, x1, x2):
        """Calculate distance based on chosen metric"""
        if self.metric == 'euclidean':
            return np.linalg.norm(x1 - x2)
        elif self.metric == 'manhattan':
            return np.linalg.norm(x1 - x2, ord=1)
        elif self.metric == 'cosine':
            return 1 - np.dot(x1, x2) / (
                np.linalg.norm(x1) * np.linalg.norm(x2)
            )
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def predict_single(self, x):
        """Predict class for a single sample"""
        # Calculate distances to all training points
        distances = [
            self._calculate_distance(x, x_train)
            for x_train in self.X_train
        ]

        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]

        # Return most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        """Predict classes for multiple samples"""
        return np.array([self.predict_single(x) for x in X])

    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Generate sample data (2 classes in 2D)
np.random.seed(42)

# Class 0: centered at (2, 2)
class0 = np.random.randn(50, 2) + np.array([2, 2])

# Class 1: centered at (6, 6)
class1 = np.random.randn(50, 2) + np.array([6, 6])

# Combine data
X = np.vstack([class0, class1])
y = np.hstack([np.zeros(50), np.ones(50)])

# Shuffle
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# Split train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train and evaluate with different metrics
metrics = ['euclidean', 'manhattan', 'cosine']
results = {}

for metric in metrics:
    knn = KNN(k=5, metric=metric)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    results[metric] = accuracy
    print(f"{metric.capitalize()} distance - Accuracy: {accuracy:.3f}")

# Visualize decision boundaries
def plot_decision_boundary(knn, X, y, title):
    """Plot decision boundary for KNN classifier"""
    # Create mesh
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Predict for each point in mesh
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1],
                c='red', label='Class 0', edgecolors='k')
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1],
                c='blue', label='Class 1', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Plot for each metric
for metric in metrics:
    knn = KNN(k=5, metric=metric)
    knn.fit(X_train, y_train)
    plot_decision_boundary(
        knn, X_train, y_train,
        f'KNN Decision Boundary ({metric.capitalize()} distance)'
    )
```

## Weekly Checkpoint

By the end of Week 4, you should be able to:
- ✅ Calculate and interpret different vector norms
- ✅ Use various distance metrics appropriately
- ✅ Understand and verify orthogonality
- ✅ Apply Gram-Schmidt process
- ✅ Implement distance-based ML algorithms (KNN)

## Resources

### Video Lectures
- [Vector Norms - Khan Academy](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces)
- [Orthogonality - MIT OCW](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)

### Reading
- [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [Distance Metrics in ML](https://machinelearningmastery.com/distance-measures-for-machine-learning/)

## Next Week Preview
Week 5 will focus on linear independence, basis vectors, and span - key concepts for understanding vector spaces and dimensionality.
