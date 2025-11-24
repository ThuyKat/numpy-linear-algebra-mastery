---
title: Week 8 - Eigenvalues & Eigenvectors
sidebar_label: Week 8 - Eigenvalues & Eigenvectors
---

# Week 8: Eigenvalues & Eigenvectors

## Time Allocation
- **Total Hours**: 13 hours
- **Theory**: 5 hours
- **NumPy Coding**: 4 hours
- **Project**: 4 hours

## Learning Objectives
- Understand eigenvalues and eigenvectors conceptually
- Learn to compute eigenvalues and eigenvectors
- Understand diagonalization
- Apply to real-world problems (PageRank, dynamics)

## Theory (5 hours)

### Resources
1. **3Blue1Brown - Essence of Linear Algebra**
   - Chapter 14: Eigenvectors and eigenvalues
   - [Watch here](https://www.youtube.com/watch?v=PFDu9oVAE-g)

2. **MIT OCW 18.06**
   - Lecture 21: Eigenvalues and eigenvectors
   - Lecture 22: Diagonalization and powers of A

### Core Concepts

#### Eigenvalues and Eigenvectors
- Definition: Av = λv (v ≠ 0)
- Geometric interpretation: vectors that don't change direction
- Characteristic equation: det(A - λI) = 0
- Eigenspace: all eigenvectors for a given eigenvalue

#### Computing Eigenvalues
- Solve characteristic polynomial
- Properties: sum of eigenvalues = trace(A)
- Product of eigenvalues = det(A)

#### Diagonalization
- A = PDP⁻¹ where D is diagonal
- Conditions for diagonalizability
- Applications: computing A^n efficiently

## NumPy Coding Exercises (4 hours)

### Computing Eigenvalues and Eigenvectors

```python
import numpy as np

# Define matrix
A = np.array([[4, -2], [1, 1]])
print(f"Matrix A:\n{A}")

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"\nEigenvalues: {eigenvalues}")
print(f"\nEigenvectors:\n{eigenvectors}")

# Verify: Av = λv for each eigenpair
for i in range(len(eigenvalues)):
    λ = eigenvalues[i]
    v = eigenvectors[:, i]

    Av = A @ v
    λv = λ * v

    print(f"\nEigenpair {i+1}:")
    print(f"  λ = {λ:.4f}")
    print(f"  v = {v}")
    print(f"  Av = {Av}")
    print(f"  λv = {λv}")
    print(f"  Equal? {np.allclose(Av, λv)}")
```

### Visualizing Eigenvectors

```python
import matplotlib.pyplot as plt

def visualize_eigenvectors(A):
    """Visualize how matrix transforms eigenvectors vs regular vectors"""
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Create some test vectors
    angles = np.linspace(0, 2*np.pi, 16)
    unit_vectors = np.array([[np.cos(θ), np.sin(θ)] for θ in angles]).T

    # Transform all vectors
    transformed = A @ unit_vectors

    # Transform eigenvectors
    eigen_transformed = A @ eigenvectors

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Before transformation
    ax1.quiver(np.zeros(len(angles)), np.zeros(len(angles)),
              unit_vectors[0], unit_vectors[1],
              angles='xy', scale_units='xy', scale=1, alpha=0.3)
    ax1.quiver([0, 0], [0, 0],
              eigenvectors[0], eigenvectors[1],
              angles='xy', scale_units='xy', scale=1,
              color=['red', 'blue'], width=0.01, label='Eigenvectors')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_title('Before Transformation')
    ax1.legend()

    # After transformation
    ax2.quiver(np.zeros(len(angles)), np.zeros(len(angles)),
              transformed[0], transformed[1],
              angles='xy', scale_units='xy', scale=1, alpha=0.3)
    ax2.quiver([0, 0], [0, 0],
              eigen_transformed[0], eigen_transformed[1],
              angles='xy', scale_units='xy', scale=1,
              color=['red', 'blue'], width=0.01, label='Transformed Eigenvectors')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_title('After Transformation')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Test
A = np.array([[2, 1], [1, 2]])
visualize_eigenvectors(A)
```

### Diagonalization

```python
# Diagonalize matrix A = PDP^(-1)
A = np.array([[4, -2], [1, 1]])

eigenvalues, P = np.linalg.eig(A)
D = np.diag(eigenvalues)

print("Diagonalization: A = PDP^(-1)")
print(f"\nP (eigenvectors):\n{P}")
print(f"\nD (diagonal of eigenvalues):\n{D}")

# Reconstruct A
P_inv = np.linalg.inv(P)
A_reconstructed = P @ D @ P_inv

print(f"\nReconstructed A:\n{A_reconstructed}")
print(f"Matches original? {np.allclose(A, A_reconstructed)}")

# Application: Compute A^10 efficiently
A_10_direct = np.linalg.matrix_power(A, 10)
D_10 = np.diag(eigenvalues ** 10)
A_10_diagonal = P @ D_10 @ P_inv

print(f"\nA^10 matches? {np.allclose(A_10_direct, A_10_diagonal)}")
```

## Project: PageRank Algorithm (4 hours)

Implement Google's PageRank using eigenvalues.

```python
import numpy as np

class PageRank:
    """Simplified PageRank algorithm using power iteration"""

    def __init__(self, damping=0.85):
        self.damping = damping
        self.ranks = None

    def build_transition_matrix(self, links):
        """
        Build transition matrix from link structure
        links: dict where links[i] = list of pages that page i links to
        """
        n_pages = max(max(links.keys()), max([p for pages in links.values() for p in pages])) + 1
        M = np.zeros((n_pages, n_pages))

        for page, outgoing in links.items():
            if len(outgoing) > 0:
                for target in outgoing:
                    M[target, page] = 1.0 / len(outgoing)

        return M

    def calculate_pagerank(self, M, max_iter=100, tol=1e-6):
        """Calculate PageRank using power iteration"""
        n = M.shape[0]

        # Add damping
        M_damped = self.damping * M + (1 - self.damping) / n * np.ones((n, n))

        # Initialize ranks equally
        ranks = np.ones(n) / n

        # Power iteration
        for i in range(max_iter):
            new_ranks = M_damped @ ranks

            if np.linalg.norm(new_ranks - ranks) < tol:
                print(f"Converged in {i+1} iterations")
                break

            ranks = new_ranks

        self.ranks = ranks
        return ranks

# Example: Simple web graph
links = {
    0: [1, 2],      # Page 0 links to pages 1 and 2
    1: [2],         # Page 1 links to page 2
    2: [0],         # Page 2 links to page 0
    3: [0, 1, 2]    # Page 3 links to pages 0, 1, 2
}

pr = PageRank(damping=0.85)
M = pr.build_transition_matrix(links)

print("Transition Matrix:")
print(M)

ranks = pr.calculate_pagerank(M)

print("\nPageRank scores:")
for page, rank in enumerate(ranks):
    print(f"  Page {page}: {rank:.4f}")

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(range(len(ranks)), ranks, color='skyblue')
plt.xlabel('Page')
plt.ylabel('PageRank Score')
plt.title('PageRank Scores')
plt.grid(True, alpha=0.3)
plt.show()
```

## Weekly Checkpoint

By the end of Week 8, you should be able to:
- ✅ Compute eigenvalues and eigenvectors
- ✅ Understand geometric meaning of eigenvectors
- ✅ Diagonalize matrices
- ✅ Apply eigendecomposition to real problems

## Resources

- [3Blue1Brown - Eigenvectors and eigenvalues](https://www.youtube.com/watch?v=PFDu9oVAE-g)
- [MIT 18.06 - Lecture 21](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- [PageRank Explained](https://en.wikipedia.org/wiki/PageRank)

## Next Week Preview
Week 9 will introduce Singular Value Decomposition (SVD), one of the most powerful matrix factorizations in linear algebra.
