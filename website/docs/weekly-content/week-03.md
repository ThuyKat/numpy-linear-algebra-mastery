---
title: Week 3 - Transpose & Dot Product
sidebar_label: Week 3 - Transpose & Dot Product
---

# Week 3: Transpose & Dot Product

## Time Allocation
- **Total Hours**: 13 hours
- **Theory**: 4 hours
- **NumPy Coding**: 5 hours
- **Project**: 4 hours

## Learning Objectives
- Master the transpose operation and its properties
- Understand the dot product (inner product) deeply
- Learn outer product and its applications
- Apply these concepts to real data analysis

## Theory (4 hours)

### Resources
1. **3Blue1Brown - Essence of Linear Algebra**
   - Chapter 9: Dot products and duality
   - [Watch here](https://www.youtube.com/watch?v=LyGKycYT2v0)

2. **Khan Academy**
   - Matrix transpose properties
   - Inner and outer products

### Core Concepts

#### Matrix Transpose
- Definition: Flipping rows and columns (Aᵀ)ᵢⱼ = Aⱼᵢ
- Properties:
  - (Aᵀ)ᵀ = A
  - (A + B)ᵀ = Aᵀ + Bᵀ
  - (AB)ᵀ = BᵀAᵀ
  - (cA)ᵀ = cAᵀ
- Symmetric matrices: A = Aᵀ

#### Dot Product (Inner Product)
- Geometric interpretation: projection and magnitude
- a · b = |a| |b| cos(θ)
- Algebraic computation: a · b = Σ aᵢbᵢ
- Properties: commutative, distributive, bilinear

#### Outer Product
- Creates a matrix from two vectors
- a ⊗ b = abᵀ
- Rank-1 matrices

## NumPy Coding Exercises (5 hours)

### Transpose Operations

```python
import numpy as np

# Basic transpose
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"Original A ({A.shape}):")
print(A)

A_T = A.T
print(f"\nTranspose A.T ({A_T.shape}):")
print(A_T)

# Verify property: (A^T)^T = A
print(f"\n(A^T)^T equals A? {np.allclose(A_T.T, A)}")

# Transpose with multiple methods
B = np.array([[1, 2], [3, 4]])
print(f"\nB.T:\n{B.T}")
print(f"\nnp.transpose(B):\n{np.transpose(B)}")
print(f"\nB.swapaxes(0, 1):\n{B.swapaxes(0, 1)}")
```

### Transpose Properties

```python
# Property: (A + B)^T = A^T + B^T
A = np.random.rand(3, 4)
B = np.random.rand(3, 4)

left = (A + B).T
right = A.T + B.T
print(f"(A + B)^T = A^T + B^T? {np.allclose(left, right)}")

# Property: (AB)^T = B^T A^T
C = np.random.rand(3, 4)
D = np.random.rand(4, 2)

left = (C @ D).T
right = D.T @ C.T
print(f"(AB)^T = B^T A^T? {np.allclose(left, right)}")

# Symmetric matrix
S = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 5, 6]])
print(f"\nIs S symmetric? {np.allclose(S, S.T)}")

# Create symmetric matrix from any matrix
M = np.random.rand(3, 3)
M_sym = (M + M.T) / 2
print(f"Is M_sym symmetric? {np.allclose(M_sym, M_sym.T)}")
```

### Dot Product (Inner Product)

```python
# Vector dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Multiple ways to compute dot product
dot1 = np.dot(a, b)
dot2 = a @ b
dot3 = np.sum(a * b)

print(f"np.dot(a, b) = {dot1}")
print(f"a @ b = {dot2}")
print(f"np.sum(a * b) = {dot3}")
print(f"All equal? {dot1 == dot2 == dot3}")

# Geometric interpretation
def angle_between(v1, v2):
    """Calculate angle between two vectors"""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

v1 = np.array([1, 0])
v2 = np.array([1, 1])

print(f"\nAngle between {v1} and {v2}: {angle_between(v1, v2):.2f}°")

# Orthogonal vectors (dot product = 0)
ortho1 = np.array([1, 0, 0])
ortho2 = np.array([0, 1, 0])
print(f"Dot product of orthogonal vectors: {np.dot(ortho1, ortho2)}")
```

### Dot Product Visualization

```python
import matplotlib.pyplot as plt

def visualize_dot_product(a, b):
    """Visualize dot product as projection"""
    # Calculate projection of b onto a
    proj_length = np.dot(a, b) / np.linalg.norm(a)
    proj = proj_length * a / np.linalg.norm(a)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Origin
    origin = np.array([0, 0])

    # Draw vectors
    ax.quiver(*origin, *a, angles='xy', scale_units='xy', scale=1,
              color='blue', width=0.01, label='a')
    ax.quiver(*origin, *b, angles='xy', scale_units='xy', scale=1,
              color='red', width=0.01, label='b')
    ax.quiver(*origin, *proj, angles='xy', scale_units='xy', scale=1,
              color='green', width=0.01, label='projection of b onto a')

    # Draw projection line
    ax.plot([b[0], proj[0]], [b[1], proj[1]],
            'k--', linewidth=1, alpha=0.5)

    # Calculate and display info
    dot_product = np.dot(a, b)
    angle = angle_between(a, b)

    ax.set_xlim(-1, max(a[0], b[0]) + 1)
    ax.set_ylim(-1, max(a[1], b[1]) + 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Dot Product = {dot_product:.2f}, Angle = {angle:.2f}°')
    plt.show()

# Test with different vectors
a = np.array([3, 1])
b = np.array([1, 2])
visualize_dot_product(a, b)
```

### Outer Product

```python
# Outer product creates a matrix
a = np.array([1, 2, 3])
b = np.array([4, 5, 6, 7])

outer = np.outer(a, b)
print(f"Outer product shape: {outer.shape}")
print(f"Outer product:\n{outer}")

# Alternative computation
outer_alt = a.reshape(-1, 1) @ b.reshape(1, -1)
print(f"\nAlternative computation:\n{outer_alt}")
print(f"Are they equal? {np.allclose(outer, outer_alt)}")

# Rank of outer product matrix
rank = np.linalg.matrix_rank(outer)
print(f"\nRank of outer product: {rank}")  # Always 1 (rank-1 matrix)
```

### Matrix-Vector Multiplication as Dot Products

```python
# Matrix-vector multiplication
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
x = np.array([1, 2, 3])

result = A @ x
print(f"A @ x = {result}")

# Understanding it as dot products of rows
row_dots = np.array([
    np.dot(A[0], x),
    np.dot(A[1], x),
    np.dot(A[2], x)
])
print(f"Row-wise dot products: {row_dots}")
print(f"Are they equal? {np.allclose(result, row_dots)}")

# Understanding it as linear combination of columns
col_combination = x[0] * A[:, 0] + x[1] * A[:, 1] + x[2] * A[:, 2]
print(f"Column combination: {col_combination}")
print(f"Are they equal? {np.allclose(result, col_combination)}")
```

## Project: Data Analysis with Dot Products (4 hours)

Apply dot products and transposes to analyze real-world data.

```python
import numpy as np
import matplotlib.pyplot as plt

class DocumentSimilarity:
    """
    Calculate document similarity using dot products
    (Simplified TF-IDF style analysis)
    """

    def __init__(self):
        self.vocabulary = {}
        self.vocab_size = 0

    def build_vocabulary(self, documents):
        """Build vocabulary from documents"""
        words = set()
        for doc in documents:
            words.update(doc.lower().split())

        self.vocabulary = {word: idx for idx, word in enumerate(sorted(words))}
        self.vocab_size = len(self.vocabulary)

    def vectorize(self, document):
        """Convert document to vector (word counts)"""
        vector = np.zeros(self.vocab_size)
        for word in document.lower().split():
            if word in self.vocabulary:
                vector[self.vocabulary[word]] += 1
        return vector

    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity (normalized dot product)"""
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)

        if magnitude1 == 0 or magnitude2 == 0:
            return 0

        return dot_product / (magnitude1 * magnitude2)

    def analyze_documents(self, documents):
        """Analyze similarity between all documents"""
        self.build_vocabulary(documents)

        # Create document-term matrix
        doc_matrix = np.array([self.vectorize(doc) for doc in documents])

        print(f"Document-Term Matrix shape: {doc_matrix.shape}")
        print(f"(documents × vocabulary size)")

        # Calculate pairwise similarities
        n_docs = len(documents)
        similarity_matrix = np.zeros((n_docs, n_docs))

        for i in range(n_docs):
            for j in range(n_docs):
                similarity_matrix[i, j] = self.cosine_similarity(
                    doc_matrix[i], doc_matrix[j]
                )

        return doc_matrix, similarity_matrix

# Test with sample documents
documents = [
    "machine learning is fascinating",
    "deep learning uses neural networks",
    "machine learning and deep learning are related",
    "python is great for data science",
    "data science uses machine learning"
]

analyzer = DocumentSimilarity()
doc_matrix, sim_matrix = analyzer.analyze_documents(documents)

print("\nSimilarity Matrix:")
print(sim_matrix)

# Visualize similarity matrix
plt.figure(figsize=(10, 8))
plt.imshow(sim_matrix, cmap='YlOrRd', interpolation='nearest')
plt.colorbar(label='Cosine Similarity')
plt.title('Document Similarity Matrix')
plt.xlabel('Document Index')
plt.ylabel('Document Index')

# Add values to cells
for i in range(len(documents)):
    for j in range(len(documents)):
        plt.text(j, i, f'{sim_matrix[i, j]:.2f}',
                ha='center', va='center',
                color='white' if sim_matrix[i, j] > 0.5 else 'black')

plt.tight_layout()
plt.show()

# Find most similar document pairs
print("\nMost similar documents:")
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        print(f"Doc {i} & Doc {j}: {sim_matrix[i, j]:.3f}")
        if sim_matrix[i, j] > 0.5 and i != j:
            print(f"  '{documents[i]}'")
            print(f"  '{documents[j]}'")
```

### Extension: Correlation Analysis

```python
# Generate sample data: student scores
np.random.seed(42)
n_students = 100

# Create correlated scores
math_scores = np.random.normal(75, 15, n_students)
physics_scores = 0.8 * math_scores + np.random.normal(0, 10, n_students)
english_scores = np.random.normal(70, 12, n_students)

# Create data matrix (students × subjects)
scores = np.column_stack([math_scores, physics_scores, english_scores])

# Calculate correlation using dot products
def correlation_matrix(data):
    """Calculate correlation matrix using transposes and dot products"""
    # Center the data (subtract mean)
    centered = data - np.mean(data, axis=0)

    # Calculate covariance matrix using transpose and dot product
    cov_matrix = (centered.T @ centered) / (len(data) - 1)

    # Convert to correlation matrix
    std_devs = np.sqrt(np.diag(cov_matrix))
    correlation = cov_matrix / np.outer(std_devs, std_devs)

    return correlation

corr = correlation_matrix(scores)

print("\nCorrelation Matrix:")
print("           Math  Physics  English")
subjects = ['Math    ', 'Physics ', 'English ']
for i, subject in enumerate(subjects):
    print(f"{subject}", end=" ")
    for j in range(3):
        print(f"{corr[i, j]:7.3f}", end=" ")
    print()

# Visualize correlations
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(math_scores, physics_scores, alpha=0.5)
axes[0].set_xlabel('Math Scores')
axes[0].set_ylabel('Physics Scores')
axes[0].set_title(f'Correlation: {corr[0, 1]:.3f}')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(math_scores, english_scores, alpha=0.5)
axes[1].set_xlabel('Math Scores')
axes[1].set_ylabel('English Scores')
axes[1].set_title(f'Correlation: {corr[0, 2]:.3f}')
axes[1].grid(True, alpha=0.3)

axes[2].scatter(physics_scores, english_scores, alpha=0.5)
axes[2].set_xlabel('Physics Scores')
axes[2].set_ylabel('English Scores')
axes[2].set_title(f'Correlation: {corr[1, 2]:.3f}')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Weekly Checkpoint

By the end of Week 3, you should be able to:
- ✅ Perform transpose operations and understand their properties
- ✅ Calculate and interpret dot products (inner products)
- ✅ Use outer products to create matrices
- ✅ Apply these operations to data analysis problems
- ✅ Understand correlation and similarity using dot products

## Resources

### Video Lectures
- [3Blue1Brown - Dot Products and Duality](https://www.youtube.com/watch?v=LyGKycYT2v0)
- [Khan Academy - Matrix Transpose](https://www.khanacademy.org/math/linear-algebra/matrix-transformations/matrix-transpose/v/linear-algebra-transpose-of-a-matrix)

### Reading
- [NumPy Dot Product](https://numpy.org/doc/stable/reference/generated/numpy.dot.html)
- [Cosine Similarity Explained](https://en.wikipedia.org/wiki/Cosine_similarity)

## Next Week Preview
Week 4 will explore norms, distance metrics, and orthogonality - fundamental concepts for understanding vector spaces and machine learning algorithms.
