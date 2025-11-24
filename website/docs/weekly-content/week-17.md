---
title: Week 17 - Convolutional Neural Networks
sidebar_label: Week 17 - CNNs
---

# Week 17: Convolutional Neural Networks (CNNs)

## Time Allocation
**Total**: 13 hours | Theory: 5h | Coding: 4h | Project: 4h

## Learning Objectives
- Understand convolution as matrix multiplication
- Learn pooling and stride operations
- Implement CNN layers from scratch
- Apply to image classification

## Core Concepts
- **Convolution**: (I * K)[i,j] = ΣΣ I[i+m, j+n] K[m,n]
- **Im2col**: Convert convolution to matrix multiplication
- **Pooling**: Max/average pooling for downsampling
- **Translation Invariance**: Shared weights across spatial dimensions

## NumPy Coding

```python
import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    """2D convolution using im2col"""
    # Add padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant')

    h, w = image.shape
    kh, kw = kernel.shape

    # Output dimensions
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1

    # Im2col: Convert to matrix multiplication
    cols = np.zeros((kh * kw, oh * ow))
    idx = 0
    for i in range(0, h - kh + 1, stride):
        for j in range(0, w - kw + 1, stride):
            patch = image[i:i+kh, j:j+kw].flatten()
            cols[:, idx] = patch
            idx += 1

    # Convolution as matrix multiply
    kernel_flat = kernel.flatten()
    output = kernel_flat @ cols

    return output.reshape(oh, ow)

# Test
image = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

# Edge detection kernel
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

result = conv2d(image, kernel)
print(f"Convolution result:\n{result}")

# Max pooling
def max_pool2d(image, pool_size=2, stride=2):
    """Max pooling"""
    h, w = image.shape
    oh = (h - pool_size) // stride + 1
    ow = (w - pool_size) // stride + 1

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            h_start = i * stride
            w_start = j * stride
            patch = image[h_start:h_start+pool_size, w_start:w_start+pool_size]
            output[i, j] = np.max(patch)

    return output

pooled = max_pool2d(image, pool_size=2)
print(f"\nMax pooling:\n{pooled}")
```

## CNN Layer Implementation

```python
class Conv2D:
    """Convolutional layer"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Initialize kernels
        k = kernel_size
        self.kernels = np.random.randn(out_channels, in_channels, k, k) * 0.1

    def forward(self, X):
        """Forward pass"""
        batch_size, in_channels, h, w = X.shape
        k = self.kernel_size

        oh = h - k + 1
        ow = w - k + 1

        output = np.zeros((batch_size, self.out_channels, oh, ow))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(in_channels):
                    output[b, oc] += conv2d(X[b, ic], self.kernels[oc, ic])

        self.X = X
        return output

class MaxPool2D:
    """Max pooling layer"""

    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, X):
        batch_size, channels, h, w = X.shape
        ps = self.pool_size

        oh = h // ps
        ow = w // ps

        output = np.zeros((batch_size, channels, oh, ow))

        for b in range(batch_size):
            for c in range(channels):
                output[b, c] = max_pool2d(X[b, c], ps, ps)

        return output

# Simple CNN
class SimpleCNN:
    """Minimal CNN for demonstration"""

    def __init__(self):
        self.conv1 = Conv2D(1, 8, kernel_size=3)
        self.pool = MaxPool2D(pool_size=2)

    def forward(self, X):
        # X shape: (batch, 1, 28, 28)
        x = self.conv1.forward(X)  # -> (batch, 8, 26, 26)
        x = self.pool.forward(x)    # -> (batch, 8, 13, 13)
        return x

# Test
X = np.random.randn(2, 1, 28, 28)  # 2 images, 1 channel, 28x28
cnn = SimpleCNN()
output = cnn.forward(X)
print(f"\nCNN output shape: {output.shape}")
```

## Project: Build CNN Classifier

```python
# Full CNN with training
# (Simplified - use frameworks for real applications)

class CNNClassifier:
    """Simple CNN for image classification"""

    def __init__(self, num_classes=10):
        self.conv1 = Conv2D(1, 16, 3)
        self.conv2 = Conv2D(16, 32, 3)
        self.pool = MaxPool2D(2)
        # Add Dense layers for classification
        # (Implementation similar to Week 16)

# See project code for full implementation
```

## Resources
- [CS231n - CNNs](http://cs231n.github.io/convolutional-networks/)
- [Understanding Convolutions](https://colah.github.io/posts/2014-07-Understanding-Convolutions/)
