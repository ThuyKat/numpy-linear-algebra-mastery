import numpy as np
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

## Find the k-th smallest element in an array
array_kth = np.array([7, 2, 1, 6, 8, 5, 4, 3])
k = 3
kth_smallest = np.partition(array_kth, k)[k]
print(f"The {k}-th smallest element in the array is:", kth_smallest)    

## Get the kth percentile of the data
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
k_percentile = 90
percentile_value = np.percentile(data, k_percentile)        
print(f"The {k_percentile}th percentile of the data is:", percentile_value) 
