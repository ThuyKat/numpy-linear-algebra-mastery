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
- **Create a virtual environment**: This is an isolated workspace for this project. Each workspace/virtual environment has its own Python version and packages. Without this common environment, incompatible packages can be used together and break the code
```bash
conda create -n linear-algebra python=3.11
# Creates a separate "workspace" called "linear-algebra"

conda activate linear-algebra  
# Switches into that workspace

conda install numpy=1.24 pandas matplotlib
# Install packages which only go to linear-algebra environment

python my_code.py  # Uses this environment's Python & packages
# Work in this environment

conda deactivate
#  DEACTIVATE when done

conda env remove -n myenv
# Delete environment (if you mess up), other Python projects stay safe

# Environment 2: Different project
conda create -n web-project python=3.9
conda activate web-project
conda install numpy=1.20 flask
# NO CONFLICTS! Each is isolated
```
## VS Code Integration

VS Code needs to know which environment to use:
```bash
1. Create environment → conda create -n ml-mastery
2. In VS Code → Ctrl+Shift+P → "Python: Select Interpreter"
3. Choose → "ml-mastery" environment
4. Now VS Code uses that environment's Python!
```
### Verify Installation
- Restart terminal, or run:
```bash
source ~/.bashrc  # or ~/.zshrc
```
- Open terminal/CLI and run: 
```bash
conda --version
```

**If installed**, you'll see:
```bash
conda 23.7.4
# (or some version number)
```

**If NOT installed**, you'll see:
```bash
'conda' is not recognized as an internal or external command
# (Windows)

conda: command not found
# (Mac/Linux)
```
### More detail check
```bash
conda info

# Should show:
#   - active environment
#   - conda version
#   - python version
#   - base environment location
# Check if you can see conda environments
conda env list

# Should show at least "base" environment
```
### Convert python script to jupyter notebook
- Jupytext is particularly nice because it:

    Maintains sync between .py and .ipynb files if you want
    Recognizes cell markers like # %% in your Python file
    Is well-maintained and widely used in the Jupyter ecosystem
-  Install jupytex
```bash
    conda install -c conda-forge jupytext
```
- Convert the file
```bash
    jupytext --to notebook your_script.py
```

### Code for testing if package runs
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
