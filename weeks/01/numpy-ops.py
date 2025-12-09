import numpy as np

# ORGANISING ARRAYS
## Concatenate arrays
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
concatenated_array = np.concatenate((array1, array2))
print("Concatenated array:", concatenated_array)

## Stack arrays vertically
array3 = np.array([[1, 2], [3, 4]])
array4 = np.array([[5, 6], [7, 8]])
vstacked_array = np.vstack((array3, array4))
print("Vertically stacked array:\n", vstacked_array)

## Stack arrays vertially with axis parameter
concatenated_array_axis1 = np.concatenate((array3, array4), axis=0)
print("Concatenated array along axis 0:\n", concatenated_array_axis1)

## Stack arrays horizontally
hstacked_array = np.hstack((array3, array4))
print("Horizontally stacked array:\n", hstacked_array)

## Split array into multiple sub-arrays
array5 = np.array([1, 2, 3, 4, 5, 6])
split_arrays = np.array_split(array5, 3)
print("Split arrays:", split_arrays)

## Split array into multiple sub-arrays unequally
split_arrays = np.array_split(array5, 4)
print("Unequally split arrays:", split_arrays)

## Split array into multiple sub-arrays unequally
unequal_split_arrays = np.array_split(array5, [2, 5])
print("Unequally split arrays:", unequal_split_arrays)

# ARRAY TRANSFORMATION
## Tile array
array6 = np.array([1, 2, 3])
tiled_array = np.tile(array6, 3)
print("Tiled array:", tiled_array)

## Repeat elements of an array
repeated_array = np.repeat(array6, 3)
print("Repeated array:", repeated_array)

# nD to 1D ARRAY CONVERSION
## Flatten a multi-dimensional array. flatten() always returns a copy (safe, independent)
array7 = np.array([[1, 2, 3], [4, 5, 6]])
flattened_array = array7.flatten()
print("Flattened array:", flattened_array)  

## Flattern with order parameter
flattened_array_F = array7.flatten(order='F')  # Column-major order
print("Flattened array (Fortran order):", flattened_array_F) # results in [1,4,2,5,3,6]
flattened_array_C = array7.flatten(order='C')  # Row-major order
print("Flattened array (C order):", flattened_array_C) # results in [1,2,3,4,5,6]

## Ravel a multi-dimensional array. ravel() returns a view (memory-efficient) when possible, falling back to copy if needed. flatten() is array method (ndarray only); ravel() is function (works on lists too).
## Assigning raveled = arr.ravel() creates a variable holding a view (not a copy)—both point to the original data buffer. 
raveled_array = array7.ravel()
print("Raveled array:", raveled_array)

### Only for non-contiguous arrays (e.g., transposed or fancy-indexed), ravel() falls back to copy.
### Non-contiguous arrays are like a linked list (elements scattered around memory, connected by "pointers" or strides

### Transpose the array to make it non-contiguous
array8 = np.array([[1, 2, 3], [4, 5, 6]])  # Contiguous: memory = [1,2,3,4,5,6]
print(array8.flags['C_CONTIGUOUS'])  # True

transposed = array8.T  # Now [[1,4], [2,5], [3,6]]
print(transposed.flags['C_CONTIGUOUS'])  # False - non-contiguous!

### Why? Original memory is row-by-row [1,2,3,4,5,6], but transpose reads column-by-column, jumping around: 1→4→2→5→3→6

### Sliced Array (Skips Rows)
sliced = array8[::2,:]  # Take every 2nd row: [[1,2,3]] starting from 0, takes all columns
print(sliced.flags['C_CONTIGUOUS'])  # False - non-contiguous!

### Fancy indexing (specific rows)
fancy_indexed = array8[[1,0],:]  # Swap rows: [[4,5,6], [1,2,3]]
print(fancy_indexed.flags['C_CONTIGUOUS'])  # False - non-contiguous!

### check strides
print("Original strides:", array8.strides)  # (24, 8) for 64-bit floats - 3 columns * 8 bytes each = 24 bytes to jump to next row, 8 bytes to jump to next column
print("Transposed strides:", transposed.strides)  # (8, 24) - now 8 bytes to jump to next row (down column), 24 bytes to jump to next column (across row)
print("Sliced strides:", sliced.strides)  # (48, 8) - 48 bytes to jump to next row (skipping one row), 8 bytes to jump to next column
print("Fancy indexed strides:", fancy_indexed.strides)  # (24, 8) - same as original, but data is accessed in a non-contiguous manner

### Now ravel() will fall back to copy for these non-contiguous arrays
raveled_transposed = transposed.ravel()
print("Raveled transposed array:", raveled_transposed)
raveled_sliced = sliced.ravel()
print("Raveled sliced array:", raveled_sliced)
raveled_fancy_indexed = fancy_indexed.ravel()
print("Raveled fancy indexed array:", raveled_fancy_indexed) 
### In all these cases, ravel() creates a new contiguous array in memory, since the original data layout is non-contiguous.
print("Is raveled transposed a copy?", np.shares_memory(transposed, raveled_transposed))  # False
print("Is raveled sliced a copy?", np.shares_memory(sliced, raveled_sliced))  # False
print("Is raveled fancy indexed a copy?", np.shares_memory(fancy_indexed, raveled_fancy_indexed))  # False
print("Is raveled flattened a copy?", np.shares_memory(array7, flattened_array))  # False, flatten() always returns a copy
print("Is raveled a view?", np.shares_memory(array7, raveled_array))  # True, ravel() returns a view when possible

### Note: To check if two arrays share the same memory, we can use np.shares_memory()

# 1D to 2D : RESHAPING ARRAYS
## Reshape array
## NOTE: The total number of elements must remain the same when reshaping.
array9 = np.array([1, 2, 3, 4, 5, 6])
reshaped_array = array9.reshape((2, 3)) # 2 rows, 3 columns
print("Reshaped array:\n", reshaped_array)

## reshape() returns a view when possible (shares memory), but makes a copy if the new shape can't be achieved with existing strides (e.g., non-contiguous arrays).
array10 = np.array([1,2,3,4,5,6,7,8])
reshaped = array10.reshape(2,4)      # VIEW ✓
reshaped[0,0] = 99
print(array10)  # [99 2 3 4 5 6 7 8] changed [web:139]
