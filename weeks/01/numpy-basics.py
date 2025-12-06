import numpy as np

# Array creation

a = np.array([1, 2, 3])
print("Array a:", a)

print("type of a", type(a))

a0 = np.zeros(2)
print("Array of zeros a0:", a0)

a1 = np.ones((2, 3))
print("Array of ones a1:\n", a1)

a2 = np.full((2, 2), 7)
print("Array full of sevens a2:\n", a2)

a3 = np.eye(3)
print("Identity matrix a3:\n", a3)

a4 = np.arange(5)
print("Array with arange a4:", a4)

a5 = np.linspace(0, 1, 5)
print("Array with linspace a5:", a5)

a6 = np.random.rand(2, 2)
print("Random array a6:\n", a6)

a7 = np.random.randint(0, 10, (2, 2))
print("Random integer array a7:\n", a7) 

a8 = np.arange(1,10,2)
print("Array contains a range of evenly spaced intervals a8:", a8)

a9 = np.arange(10).reshape(2,5)
print("Reshaped array a9:\n", a9)

# Array properties

print("Shape of a9:", a9.shape)
print("Data type of a9:", a9.dtype)
print("Number of dimensions of a9:", a9.ndim)
print("Size of each element in bytes of a9:", a9.itemsize)
print("Total number of elements in a9:", a9.size)
print("Total size of a9 in bytes:", a9.nbytes)
print("Type of a9:", type(a9))

