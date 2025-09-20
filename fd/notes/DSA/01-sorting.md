# Sorting Algorithms

## Quicksort

Quicksort is a divide-and-conquer algorithm that works by selecting a 'pivot' element from the array and partitioning the other elements into two sub-arrays according to whether they are less than or greater than the pivot.

### Time Complexity
- Best case: O(n log n)
- Average case: O(n log n) 
- Worst case: O(n²)

### Space Complexity
- O(log n) due to recursion stack

### Algorithm Steps
1. Choose a pivot element from the array
2. Partition the array so elements smaller than pivot go to left, larger go to right
3. Recursively apply quicksort to the sub-arrays

## Merge Sort

Merge sort is a stable, divide-and-conquer algorithm that divides the array into halves, sorts them separately, and then merges them back together.

### Time Complexity
- All cases: O(n log n)

### Space Complexity
- O(n) for the temporary arrays used in merging

### Advantages
- Stable sorting algorithm
- Guaranteed O(n log n) performance
- Works well for large datasets
