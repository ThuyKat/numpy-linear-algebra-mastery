---
title: Week 5 - Linear Independence & Span
sidebar_label: Week 5 - Linear Independence & Span
---

# Week 5: Linear Independence & Span

## Time Allocation
- **Total Hours**: 12 hours
- **Theory**: 5 hours
- **NumPy Coding**: 4 hours
- **Project**: 3 hours

## Learning Objectives
- Understand linear independence and dependence
- Master the concepts of span and basis
- Learn about column space and row space
- Understand rank and its significance

## Theory (5 hours)

### Resources
1. **3Blue1Brown - Essence of Linear Algebra**
   - Chapter 2: Span and linear combinations
   - Chapter 7: Inverse matrices, column space, rank, null space
   - [Watch here](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

2. **MIT OCW 18.06**
   - Lecture 6: Column space and nullspace
   - Lecture 9: Independence, basis, and dimension

### Core Concepts

#### Linear Combinations
- Definition: c₁v₁ + c₂v₂ + ... + cₙvₙ
- Span: all possible linear combinations
- Geometric interpretation

#### Linear Independence
- Vectors are linearly independent if:
  - c₁v₁ + c₂v₂ + ... + cₙvₙ = 0 only when all cᵢ = 0
- Otherwise, they are linearly dependent
- Geometric meaning: vectors don't lie in same subspace

#### Basis and Dimension
- Basis: linearly independent set that spans the space
- Dimension: number of vectors in a basis
- Standard basis for ℝⁿ

#### Column Space and Rank
- Column space: span of matrix columns
- Rank: dimension of column space
- Full rank vs rank deficient

## NumPy Coding Exercises (4 hours)

### Linear Combinations

```python
import numpy as np
import matplotlib.pyplot as plt

# Define vectors
v1 = np.array([1, 2])
v2 = np.array([2, 1])

# Create linear combinations
coefficients = [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (-1, 1)]

plt.figure(figsize=(12, 8))
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

# Plot original vectors
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
           color='red', width=0.01, label='v1')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
           color='blue', width=0.01, label='v2')

# Plot linear combinations
for c1, c2 in coefficients:
    result = c1 * v1 + c2 * v2
    plt.quiver(0, 0, result[0], result[1], angles='xy', scale_units='xy',
               scale=1, color='green', width=0.005, alpha=0.6)
    plt.text(result[0], result[1], f'({c1},{c2})',
             fontsize=8, ha='center')

plt.xlim(-3, 5)
plt.ylim(-2, 6)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Linear Combinations: c₁v₁ + c₂v₂')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()
```

### Visualizing Span

```python
def visualize_span_2d(v1, v2):
    """Visualize the span of two vectors in 2D"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Generate many linear combinations
    t = np.linspace(-3, 3, 100)
    s = np.linspace(-3, 3, 100)

    for ax, title, single_vec in zip(axes,
                                      ['Span of v1 (line)', 'Span of v1 and v2'],
                                      [True, False]):
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

        if single_vec:
            # Span of single vector (line)
            points = np.outer(t, v1)
            ax.plot(points[:, 0], points[:, 1], 'g-', alpha=0.5, linewidth=2)
            ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy',
                     scale=1, color='red', width=0.01)
        else:
            # Span of two vectors (plane - fills entire 2D space if independent)
            for ti in t[::5]:
                for si in s[::5]:
                    point = ti * v1 + si * v2
                    ax.plot(point[0], point[1], 'g.', alpha=0.3)

            ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy',
                     scale=1, color='red', width=0.01)
            ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy',
                     scale=1, color='blue', width=0.01)

        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

v1 = np.array([1, 2])
v2 = np.array([2, 1])
visualize_span_2d(v1, v2)
```

### Testing Linear Independence

```python
def are_linearly_independent(vectors):
    """
    Check if vectors are linearly independent
    Returns: (bool, explanation)
    """
    # Stack vectors as columns
    matrix = np.column_stack(vectors)

    # Calculate rank
    rank = np.linalg.matrix_rank(matrix)
    n_vectors = len(vectors)

    is_independent = (rank == n_vectors)

    explanation = f"Matrix has rank {rank}, with {n_vectors} vectors. "
    if is_independent:
        explanation += "Vectors are linearly independent."
    else:
        explanation += "Vectors are linearly dependent."

    return is_independent, explanation

# Test cases
test_cases = [
    ("Standard basis", [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1])
    ]),
    ("Independent vectors", [
        np.array([1, 2, 3]),
        np.array([4, 5, 6]),
        np.array([7, 8, 10])
    ]),
    ("Dependent vectors (third is sum)", [
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, 1])
    ]),
    ("Dependent vectors (parallel)", [
        np.array([1, 2]),
        np.array([2, 4])
    ])
]

for name, vectors in test_cases:
    is_indep, explanation = are_linearly_independent(vectors)
    print(f"\n{name}:")
    for i, v in enumerate(vectors):
        print(f"  v{i+1} = {v}")
    print(f"  {explanation}")
```

### Finding Basis

```python
def find_basis(vectors):
    """
    Find a basis from a set of vectors
    Returns linearly independent vectors that span the same space
    """
    # Stack vectors as columns
    matrix = np.column_stack(vectors)

    # Use QR decomposition to find independent columns
    Q, R = np.linalg.qr(matrix)

    # Find independent columns (where diagonal of R is non-zero)
    tolerance = 1e-10
    independent_indices = np.where(np.abs(np.diag(R)) > tolerance)[0]

    basis = [vectors[i] for i in independent_indices]

    return basis, independent_indices

# Test with redundant vectors
vectors = [
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([1, 1, 0]),  # Linear combination of first two
    np.array([0, 0, 1]),
    np.array([2, 0, 0])   # Multiple of first
]

basis, indices = find_basis(vectors)

print("Original vectors:")
for i, v in enumerate(vectors):
    print(f"  v{i} = {v}")

print(f"\nBasis (using vectors at indices {list(indices)}):")
for i, v in enumerate(basis):
    print(f"  basis{i} = {v}")

print(f"\nDimension of span: {len(basis)}")
```

### Matrix Rank

```python
def analyze_matrix_rank(matrix, name="Matrix"):
    """Analyze rank and related properties of a matrix"""
    m, n = matrix.shape
    rank = np.linalg.matrix_rank(matrix)

    print(f"\n{name} ({m}×{n}):")
    print(matrix)
    print(f"\nRank: {rank}")
    print(f"Full rank? {rank == min(m, n)}")
    print(f"Column rank = Row rank = {rank}")

    # Determine type
    if rank == min(m, n):
        if m == n:
            print("Type: Full rank square matrix (invertible)")
        elif rank == n:
            print("Type: Full column rank (columns are independent)")
        else:
            print("Type: Full row rank (rows are independent)")
    else:
        print("Type: Rank deficient")

    return rank

# Test matrices
test_matrices = {
    "Identity": np.eye(3),
    "Full rank": np.array([[1, 2], [3, 4], [5, 6]]),
    "Rank deficient (duplicate column)": np.array([[1, 2, 1], [3, 4, 3]]),
    "Rank 1 (outer product)": np.outer([1, 2, 3], [4, 5, 6])
}

for name, matrix in test_matrices.items():
    analyze_matrix_rank(matrix, name)
```

### Column Space Visualization

```python
def visualize_column_space():
    """Visualize column space of a matrix"""
    # Matrix with 2 independent columns in 3D space
    A = np.array([
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    # Generate points in column space
    t = np.linspace(-2, 2, 20)
    s = np.linspace(-2, 2, 20)
    T, S = np.meshgrid(t, s)

    # Each point is a linear combination of columns
    points = []
    for ti, si in zip(T.flatten(), S.flatten()):
        point = ti * A[:, 0] + si * A[:, 1]
        points.append(point)

    points = np.array(points)

    # Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot column space (as points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
              c='green', alpha=0.1, s=1)

    # Plot column vectors
    origin = np.zeros(3)
    ax.quiver(*origin, *A[:, 0], color='red', arrow_length_ratio=0.1,
             linewidth=3, label='Column 1')
    ax.quiver(*origin, *A[:, 1], color='blue', arrow_length_ratio=0.1,
             linewidth=3, label='Column 2')

    # Plot some example linear combinations
    examples = [(1, 0), (0, 1), (1, 1), (2, 1)]
    for c1, c2 in examples:
        result = c1 * A[:, 0] + c2 * A[:, 1]
        ax.quiver(*origin, *result, color='purple', arrow_length_ratio=0.1,
                 alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Column Space of Matrix A (plane in 3D)')
    ax.legend()
    plt.show()

visualize_column_space()
```

## Project: Dimensionality Analysis Tool (3 hours)

Create a tool to analyze the dimensional properties of datasets.

```python
import numpy as np
import matplotlib.pyplot as plt

class DimensionalityAnalyzer:
    """Analyze linear independence and effective dimensionality of data"""

    def __init__(self, data):
        """
        data: array of shape (n_samples, n_features)
        """
        self.data = data
        self.n_samples, self.n_features = data.shape
        self.centered_data = data - np.mean(data, axis=0)

    def check_column_independence(self):
        """Check if feature columns are linearly independent"""
        rank = np.linalg.matrix_rank(self.data)

        print(f"Dataset shape: {self.data.shape}")
        print(f"Number of features: {self.n_features}")
        print(f"Matrix rank: {rank}")

        if rank == self.n_features:
            print("✓ All features are linearly independent")
        else:
            print(f"✗ Features are dependent (rank deficient by {self.n_features - rank})")

        return rank

    def find_dependent_features(self, tolerance=1e-10):
        """Identify which features might be redundant"""
        # Use SVD to find dependent features
        U, s, Vt = np.linalg.svd(self.centered_data, full_matrices=False)

        # Features with small singular values are potentially dependent
        dependent_threshold = tolerance * s[0]
        dependent_indices = np.where(s < dependent_threshold)[0]

        if len(dependent_indices) > 0:
            print(f"\nPotentially dependent features: {dependent_indices}")
            print(f"Singular values: {s}")
        else:
            print("\nNo obviously dependent features found")

        return s

    def compute_effective_rank(self, tolerance=0.01):
        """
        Compute effective rank (number of singular values above threshold)
        """
        _, s, _ = np.linalg.svd(self.centered_data, full_matrices=False)

        # Normalize singular values
        s_normalized = s / s[0]

        # Count values above threshold
        effective_rank = np.sum(s_normalized > tolerance)

        print(f"\nEffective rank (threshold={tolerance}): {effective_rank}")
        print(f"Normalized singular values: {s_normalized}")

        return effective_rank, s

    def visualize_singular_values(self):
        """Plot singular value spectrum"""
        _, s, _ = np.linalg.svd(self.centered_data, full_matrices=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Singular values
        ax1.plot(range(1, len(s) + 1), s, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Singular Value')
        ax1.set_title('Singular Value Spectrum')
        ax1.grid(True, alpha=0.3)

        # Cumulative explained variance
        variance_explained = (s ** 2) / np.sum(s ** 2)
        cumulative_variance = np.cumsum(variance_explained)

        ax2.plot(range(1, len(s) + 1), cumulative_variance, 'ro-',
                linewidth=2, markersize=8)
        ax2.axhline(y=0.95, color='g', linestyle='--',
                   label='95% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Variance Explained')
        ax2.set_title('Cumulative Variance Explained')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

# Generate test data with redundant features
np.random.seed(42)

# Create base features
n_samples = 100
feature1 = np.random.randn(n_samples)
feature2 = np.random.randn(n_samples)
feature3 = np.random.randn(n_samples)

# Create dataset with redundancy
data = np.column_stack([
    feature1,                    # Independent feature
    feature2,                    # Independent feature
    feature3,                    # Independent feature
    2 * feature1 + feature2,     # Dependent feature (linear combination)
    feature1 + 0.1 * np.random.randn(n_samples)  # Nearly dependent
])

print("Created dataset with 5 features (2 are dependent/near-dependent)\n")
print("=" * 60)

# Analyze
analyzer = DimensionalityAnalyzer(data)
rank = analyzer.check_column_independence()
singular_values = analyzer.find_dependent_features()
effective_rank, s = analyzer.compute_effective_rank(tolerance=0.01)
analyzer.visualize_singular_values()

# Test with independent data
print("\n" + "=" * 60)
print("Testing with fully independent data:")
print("=" * 60)

independent_data = np.random.randn(100, 5)
analyzer2 = DimensionalityAnalyzer(independent_data)
analyzer2.check_column_independence()
analyzer2.compute_effective_rank(tolerance=0.01)
```

## Weekly Checkpoint

By the end of Week 5, you should be able to:
- ✅ Understand and identify linear independence
- ✅ Calculate and interpret matrix rank
- ✅ Work with basis vectors and span
- ✅ Analyze dimensionality of datasets
- ✅ Identify redundant features in data

## Resources

### Video Lectures
- [3Blue1Brown - Linear combinations, span, and basis vectors](https://www.youtube.com/watch?v=k7RM-ot2NWY)
- [3Blue1Brown - Column space and null space](https://www.youtube.com/watch?v=uQhTuRlWMxw)
- [MIT 18.06 - Lecture 6](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-6-column-space-and-nullspace/)

### Reading
- [Linear Independence - Khan Academy](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/linear-independence/v/linear-algebra-introduction-to-linear-independence)
- [Matrix Rank - Wikipedia](https://en.wikipedia.org/wiki/Rank_(linear_algebra))

## Next Week Preview
Week 6 will introduce matrix decompositions, starting with LU decomposition for solving linear systems efficiently.
