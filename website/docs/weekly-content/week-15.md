---
title: Week 15 - Advanced Decompositions
sidebar_label: Week 15 - Advanced Decompositions
---

# Week 15: Advanced Matrix Decompositions

## Time Allocation
**Total**: 12 hours | Theory: 4h | Coding: 4h | Project: 4h

## Learning Objectives
- Master Cholesky decomposition
- Learn Schur decomposition
- Understand Jordan normal form
- Apply to specialized problems

## Core Concepts
- **Cholesky**: A = LLᵀ (for positive definite matrices)
- **Schur**: A = QTQᵀ (Q orthogonal, T upper triangular)
- **Applications**: Solving systems, simulations, eigenvalue algorithms

## NumPy Coding

```python
import numpy as np

# Cholesky decomposition
A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
L = np.linalg.cholesky(A)

print(f"Cholesky factor L:\n{L}")
print(f"L @ Lᵀ = A? {np.allclose(L @ L.T, A)}")

# Solve using Cholesky (more efficient than LU for SPD matrices)
b = np.array([1, 2, 3])
y = np.linalg.solve(L, b)  # Forward substitution
x = np.linalg.solve(L.T, y)  # Backward substitution
print(f"\nSolution: {x}")
print(f"Verification: {np.allclose(A @ x, b)}")

# Schur decomposition
A = np.random.randn(5, 5)
T, Z = np.linalg.schur(A)

print(f"\nSchur form T:\n{T}")
print(f"Orthogonal Z @ Zᵀ = I? {np.allclose(Z @ Z.T, np.eye(5))}")
print(f"Z @ T @ Zᵀ = A? {np.allclose(Z @ T @ Z.T, A)}")
```

## Project: Multivariate Normal Sampling

```python
class MultivariateNormalSampler:
    """Sample from multivariate normal using Cholesky"""

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.L = np.linalg.cholesky(cov)

    def sample(self, n_samples=1):
        """Generate samples"""
        d = len(self.mean)
        z = np.random.randn(n_samples, d)
        return z @ self.L.T + self.mean

# Test
mean = np.array([0, 0])
cov = np.array([[1, 0.5], [0.5, 1]])

sampler = MultivariateNormalSampler(mean, cov)
samples = sampler.sample(1000)

import matplotlib.pyplot as plt
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3)
plt.title('Multivariate Normal Samples')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
```

## Resources
- [Cholesky Decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition)
- [Schur Decomposition](https://nhigham.com/2021/01/26/what-is-the-schur-decomposition/)
