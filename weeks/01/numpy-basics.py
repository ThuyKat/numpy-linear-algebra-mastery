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

# SORTING ARRAYS

## Sort array in ascending order
unsorted_array = np.array([3, 1, 4, 1, 5, 9, 2, 6])
sorted_array = np.sort(unsorted_array)
print("Sorted array:", sorted_array)

## Sort array in descending order
descending_sorted_array = np.sort(unsorted_array)[::-1]
print("Descending sorted array:", descending_sorted_array)

## Sort along a specific axis
array_2d = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
sorted_2d_axis0 = np.sort(array_2d, axis=0) #sorting vertically - each column gets sorted, [3, 1, 2] → [1, 2, 3], Second column [1, 5, 6] → [1, 5, 6], Third column [4, 9, 5] → [4, 5, 9]
sorted_2d_axis1 = np.sort(array_2d, axis=1) #sorting horizontally - each row gets sorted, First row [3, 1, 4] → [1, 3, 4], Second row [1, 5, 9] → [1, 5, 9], Third row [2, 6, 5] → [2, 5, 6]
print("2D array sorted along axis 0:\n", sorted_2d_axis0)
print("2D array sorted along axis 1:\n", sorted_2d_axis1)   

indices = np.argsort(array_2d) #[[3, 1, 4], [1, 5, 9], [2, 6, 5]]
print("Indices that would sort the array:", indices)

## Sort using multiple keys - without lexsort
structured_array = np.array([(1, 'b'), (3, 'a'), (2, 'c')],
                            dtype=[('num', 'i4'), ('char', 'U1')])
sorted_structured_array = np.sort(structured_array, order=['num', 'char'])
print("Structured array sorted by multiple keys:\n", sorted_structured_array)   

## Sort using multiple keys - with lexsort
keys = (structured_array['char'], structured_array['num'])
lexsorted_indices = np.lexsort(keys) # Sort by 'num' then by 'char'
lexsorted_array = structured_array[lexsorted_indices]
print("Structured array lexsorted by multiple keys:\n", lexsorted_array)

##  finds the index where each value should be inserted into a sorted 1D array to keep it sorted (binary search). 
array_1d = np.array([1, 3, 5, 7, 9])
values_to_insert = np.array([0, 2, 4, 6, 8, 10])
insertion_indices = np.searchsorted(array_1d, values_to_insert)
insertion_indices_left = np.searchsorted(array_1d, values_to_insert, side='left')
insertion_indices_right = np.searchsorted(array_1d, values_to_insert, side='right')
print("Insertion indices (left) for values:", insertion_indices_left)
print("Insertion indices (right) for values:", insertion_indices_right)
print("Insertion indices for values:", insertion_indices) # same result as left or right in this case. side='left' and side='right' only matters when the value already exists in the array.
