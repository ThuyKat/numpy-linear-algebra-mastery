---
title: Week 9 - Singular Value Decomposition (SVD)
sidebar_label: Week 9 - SVD
---

# Week 9: Singular Value Decomposition (SVD)

## Time Allocation
- **Total Hours**: 14 hours
- **Theory**: 5 hours
- **NumPy Coding**: 5 hours
- **Project**: 4 hours

## Learning Objectives
- Understand SVD and its significance
- Learn to compute and interpret SVD components
- Apply SVD to dimensionality reduction
- Use SVD for image compression

## Theory (5 hours)

### Resources
1. **3Blue1Brown**: [Abstract vector spaces](https://www.youtube.com/watch?v=TgKwz5Ikpc8)
2. **MIT OCW 18.06**: Lecture 29 - Singular value decomposition
3. **Steve Brunton**: [SVD playlist](https://www.youtube.com/playlist?list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv)

### Core Concepts
- **SVD Decomposition**: A = UΣVᵀ
- **U**: Left singular vectors (eigenvectors of AAᵀ)
- **Σ**: Singular values (square root of eigenvalues)
- **V**: Right singular vectors (eigenvectors of AᵀA)
- Applications: dimensionality reduction, compression, denoising

## NumPy Coding Exercises (5 hours)

```python
import numpy as np

# Compute SVD
A = np.array([[1, 2], [3, 4], [5, 6]])
U, s, Vt = np.linalg.svd(A, full_matrices=False)

print(f"A ({A.shape}):\n{A}")
print(f"\nU ({U.shape}):\n{U}")
print(f"\nSingular values: {s}")
print(f"\nVᵀ ({Vt.shape}):\n{Vt}")

# Reconstruct A
Sigma = np.diag(s)
A_reconstructed = U @ Sigma @ Vt
print(f"\nReconstructed A:\n{A_reconstructed}")
print(f"Match? {np.allclose(A, A_reconstructed)}")

# Low-rank approximation
def low_rank_approx(A, k):
    """Approximate matrix using top k singular values"""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Sigma_k = np.diag(s[:k])
    return U[:, :k] @ Sigma_k @ Vt[:k, :]

# Test with image data
from PIL import Image
import matplotlib.pyplot as plt

# Load image (convert to grayscale)
img = np.random.rand(100, 100)  # Or load real image

# Compress with different ranks
ranks = [1, 5, 10, 20, 50]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')

for idx, k in enumerate(ranks):
    ax = axes[(idx+1)//3, (idx+1)%3]
    compressed = low_rank_approx(img, k)
    ax.imshow(compressed, cmap='gray')
    ax.set_title(f'Rank {k}')

plt.tight_layout()
plt.show()
```

## Project: Image Compression with SVD (4 hours)

```python
class ImageCompressor:
    """Compress images using SVD"""

    def compress(self, image, k):
        """Compress image using top k singular values"""
        if len(image.shape) == 3:  # RGB
            compressed = np.zeros_like(image)
            for channel in range(3):
                U, s, Vt = np.linalg.svd(image[:, :, channel], full_matrices=False)
                compressed[:, :, channel] = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
            return compressed
        else:  # Grayscale
            U, s, Vt = np.linalg.svd(image, full_matrices=False)
            return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

    def compression_ratio(self, original_shape, k):
        """Calculate compression ratio"""
        m, n = original_shape[:2]
        original_size = m * n
        compressed_size = k * (m + n + 1)
        return original_size / compressed_size

compressor = ImageCompressor()
# Use with your images
```

## Weekly Checkpoint
- ✅ Understand SVD components
- ✅ Implement SVD-based compression
- ✅ Analyze singular value spectrum
- ✅ Apply low-rank approximations

## Resources
- [SVD Tutorial](https://www.youtube.com/watch?v=nbBvuuNVfco)
- [NumPy SVD docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
