---
title: Week 0 - Setup & Prerequisites
sidebar_label: Week 0 - Setup & Prerequisites
---

# Week 0: Setup & Prerequisites

## Time Allocation
- **Total Hours**: 4 hours

## Learning Objectives
- Set up Python development environment
- Install essential libraries (NumPy, Matplotlib, Jupyter)
- Understand basic Python programming concepts
- Familiarize yourself with Jupyter notebooks

## Theory (2 hours)

### Python Basics Review
If you're new to Python, review these fundamentals:
- Variables and data types
- Control structures (if/else, loops)
- Functions
- Lists and dictionaries

### Development Environment
- Install Python 3.8+ (recommended: Python 3.11)
- Set up a virtual environment
- Install Jupyter Notebook or JupyterLab
- Install essential libraries

## NumPy Coding Exercises (2 hours)

### Environment Setup

```python
# Create a virtual environment
# In your terminal:
# python -m venv numpy-env
# source numpy-env/bin/activate  # On Windows: numpy-env\Scripts\activate

# Install required packages
# pip install numpy matplotlib jupyter scipy pandas
```

### Verify Installation

```python
import numpy as np
import matplotlib.pyplot as plt
import sys

print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")

# Test basic NumPy functionality
test_array = np.array([1, 2, 3, 4, 5])
print(f"Test array: {test_array}")
print(f"Array type: {type(test_array)}")
```

### First NumPy Operations

```python
# Create arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Basic operations
print(f"Addition: {a + b}")
print(f"Multiplication: {a * b}")
print(f"Sum: {np.sum(a)}")
print(f"Mean: {np.mean(a)}")
```

### First Visualization

```python
# Create a simple plot
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.show()
```

## Weekly Checkpoint

By the end of Week 0, you should be able to:
- ✅ Successfully run Python and import NumPy
- ✅ Create and manipulate basic NumPy arrays
- ✅ Run Jupyter notebooks
- ✅ Create simple visualizations with Matplotlib

## Resources

### Installation Guides
- [Python Installation](https://www.python.org/downloads/)
- [NumPy Installation](https://numpy.org/install/)
- [Jupyter Installation](https://jupyter.org/install)

### Python Basics
- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python - Python Basics](https://realpython.com/learning-paths/python-basics/)

## Next Week Preview
Week 1 will introduce you to vectors, vector operations, and the fundamentals of NumPy arrays. Make sure your environment is fully set up before moving forward!
