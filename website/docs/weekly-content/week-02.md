---
title: Week 2 - Matrices & Transformations
sidebar_label: Week 2 - Matrices & Transformations
---

# Week 2: Matrices & Transformations

## Time Allocation
- **Total Hours**: 13 hours
- **Theory**: 5 hours
- **NumPy Coding**: 4 hours
- **Project**: 4 hours

## Learning Objectives
- Understand matrix basics and notation
- Learn matrix addition, multiplication, and scalar operations
- Explore linear transformations and their geometric interpretations
- Implement matrix operations in NumPy

## Theory (5 hours)

### Resources
1. **3Blue1Brown - Essence of Linear Algebra**
   - Chapter 3: Linear transformations and matrices
   - Chapter 4: Matrix multiplication as composition
   - [Watch here](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

2. **MIT OCW 18.06**
   - Lecture 2: Elimination with matrices
   - Focus on matrix operations and transformations

### Core Concepts

#### Matrix Basics
- Matrix notation and dimensions (m × n)
- Special matrices: identity, zero, diagonal
- Matrix equality and comparison

#### Matrix Operations
- **Addition and Subtraction**: Element-wise operations
- **Scalar Multiplication**: Multiplying matrix by a scalar
- **Matrix Multiplication**: Row-column dot product
- **Hadamard Product**: Element-wise multiplication

#### Linear Transformations
- Transformation as a function from ℝⁿ → ℝᵐ
- Geometric interpretation (rotation, scaling, shearing, reflection)
- Representing transformations as matrices

## NumPy Coding Exercises (4 hours)

### Matrix Creation and Basic Operations

```python
import numpy as np

# Creating matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix A:")
print(A)
print(f"Shape: {A.shape}")
print(f"Size: {A.size}")

# Matrix addition
C = A + B
print(f"\nA + B:\n{C}")

# Scalar multiplication
D = 3 * A
print(f"\n3 * A:\n{D}")

# Element-wise multiplication (Hadamard product)
E = A * B
print(f"\nA * B (element-wise):\n{E}")

# Matrix multiplication
F = np.dot(A, B)
# or
F = A @ B
print(f"\nA @ B (matrix multiplication):\n{F}")
```

### Special Matrices

```python
# Identity matrix
I = np.eye(3)
print(f"Identity matrix (3x3):\n{I}")

# Zero matrix
Z = np.zeros((3, 3))
print(f"\nZero matrix:\n{Z}")

# Diagonal matrix
D = np.diag([1, 2, 3])
print(f"\nDiagonal matrix:\n{D}")

# Random matrix
R = np.random.rand(3, 3)
print(f"\nRandom matrix:\n{R}")
```

### Linear Transformations

```python
import matplotlib.pyplot as plt

# Define a transformation matrix (rotation by 45 degrees)
theta = np.pi / 4  # 45 degrees in radians
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Original vectors
vectors = np.array([[1, 0], [0, 1], [1, 1]]).T

# Apply transformation
transformed = rotation_matrix @ vectors

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original vectors
ax1.quiver([0, 0, 0], [0, 0, 0],
           vectors[0], vectors[1],
           angles='xy', scale_units='xy', scale=1,
           color=['r', 'g', 'b'])
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.grid(True)
ax1.set_aspect('equal')
ax1.set_title('Original Vectors')

# Transformed vectors
ax2.quiver([0, 0, 0], [0, 0, 0],
           transformed[0], transformed[1],
           angles='xy', scale_units='xy', scale=1,
           color=['r', 'g', 'b'])
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
ax2.grid(True)
ax2.set_aspect('equal')
ax2.set_title('Rotated by 45°')

plt.tight_layout()
plt.show()
```

### Common Transformations

```python
def create_transformation_matrix(transform_type, param=1):
    """
    Create common transformation matrices

    transform_type: 'rotation', 'scaling', 'shearing', 'reflection'
    param: angle for rotation (radians), scale factor, or shear factor
    """
    if transform_type == 'rotation':
        theta = param
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
    elif transform_type == 'scaling':
        return np.array([
            [param, 0],
            [0, param]
        ])
    elif transform_type == 'shearing':
        return np.array([
            [1, param],
            [0, 1]
        ])
    elif transform_type == 'reflection':
        # Reflection over x-axis
        return np.array([
            [1, 0],
            [0, -1]
        ])

# Test transformations
original = np.array([[1, 0], [0, 1]]).T

transforms = {
    'Rotation 90°': create_transformation_matrix('rotation', np.pi/2),
    'Scaling 2x': create_transformation_matrix('scaling', 2),
    'Shearing': create_transformation_matrix('shearing', 0.5),
    'Reflection': create_transformation_matrix('reflection')
}

for name, matrix in transforms.items():
    result = matrix @ original
    print(f"\n{name}:")
    print(f"Matrix:\n{matrix}")
    print(f"Result:\n{result}")
```

### Matrix Properties

```python
# Commutativity (generally NOT commutative)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

AB = A @ B
BA = B @ A

print("A @ B:")
print(AB)
print("\nB @ A:")
print(BA)
print(f"\nAre they equal? {np.allclose(AB, BA)}")

# Associativity (IS associative)
C = np.array([[2, 1], [1, 2]])

ABC1 = (A @ B) @ C
ABC2 = A @ (B @ C)

print(f"\nAssociativity check: {np.allclose(ABC1, ABC2)}")

# Identity property
I = np.eye(2)
print(f"\nA @ I equals A? {np.allclose(A @ I, A)}")
```

## Project: Interactive Transformation Visualizer (4 hours)

Create an interactive tool to visualize how different matrices transform 2D shapes.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def plot_transformation(original_shape, transform_matrix, title=""):
    """
    Visualize how a transformation matrix affects a shape
    """
    # Apply transformation
    transformed_shape = (transform_matrix @ original_shape.T).T

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Original shape
    poly1 = Polygon(original_shape, fill=False, edgecolor='blue', linewidth=2)
    ax1.add_patch(poly1)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.set_title('Original Shape')

    # Transformed shape
    poly2 = Polygon(transformed_shape, fill=False, edgecolor='red', linewidth=2)
    ax2.add_patch(poly2)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.set_title(f'After Transformation: {title}')

    plt.tight_layout()
    plt.show()

    print(f"Transformation Matrix:\n{transform_matrix}")
    print(f"\nDeterminant: {np.linalg.det(transform_matrix):.2f}")

# Define a square
square = np.array([
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1]
])

# Test different transformations
transformations = {
    'Rotation 30°': np.array([
        [np.cos(np.pi/6), -np.sin(np.pi/6)],
        [np.sin(np.pi/6), np.cos(np.pi/6)]
    ]),
    'Scaling (2x, 0.5y)': np.array([
        [2, 0],
        [0, 0.5]
    ]),
    'Shearing': np.array([
        [1, 0.5],
        [0, 1]
    ]),
    'Reflection (y-axis)': np.array([
        [-1, 0],
        [0, 1]
    ])
}

for name, matrix in transformations.items():
    plot_transformation(square, matrix, name)
```

### Extension: Composite Transformations

```python
# Combine multiple transformations
def composite_transform():
    """Demonstrate composition of transformations"""
    # Original shape
    triangle = np.array([
        [0, 0],
        [1, 0],
        [0.5, 1]
    ])

    # Define transformations
    rotate = np.array([
        [np.cos(np.pi/4), -np.sin(np.pi/4)],
        [np.sin(np.pi/4), np.cos(np.pi/4)]
    ])

    scale = np.array([
        [1.5, 0],
        [0, 1.5]
    ])

    # Apply transformations in sequence
    step1 = (rotate @ triangle.T).T
    step2 = (scale @ step1.T).T

    # Or combine matrices first
    combined = scale @ rotate
    direct = (combined @ triangle.T).T

    print(f"Sequential equals combined? {np.allclose(step2, direct)}")

    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    shapes = [triangle, step1, step2, direct]
    titles = ['Original', 'After Rotation', 'After Scaling', 'Direct (Combined)']

    for ax, shape, title in zip(axes, shapes, titles):
        poly = Polygon(shape, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(poly)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

composite_transform()
```

## Weekly Checkpoint

By the end of Week 2, you should be able to:
- ✅ Create and manipulate matrices in NumPy
- ✅ Perform matrix operations (addition, multiplication)
- ✅ Understand and visualize linear transformations
- ✅ Apply transformation matrices to 2D shapes
- ✅ Recognize properties of matrix operations

## Resources

### Video Lectures
- [3Blue1Brown - Linear Transformations](https://www.youtube.com/watch?v=kYB8IZa5AuE)
- [3Blue1Brown - Matrix Multiplication](https://www.youtube.com/watch?v=XkY2DOUCWMU)
- [MIT 18.06 - Lecture 2](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-2-elimination-with-matrices/)

### Reading
- [NumPy Matrix Operations](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [Linear Transformations - Khan Academy](https://www.khanacademy.org/math/linear-algebra/matrix-transformations)

## Next Week Preview
Week 3 will dive into the transpose operation and the dot product, exploring their mathematical properties and applications in machine learning.
