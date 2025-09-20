"""Setup example data and demonstrate the Notes Q&A system."""

import os
from pathlib import Path

from config import NOTES_DIR, AUX_DIR


def create_example_structure():
    """Create example directory structure and sample files."""
    
    # Create course directories
    dsa_dir = os.path.join(NOTES_DIR, "DSA")
    os.makedirs(dsa_dir, exist_ok=True)
    
    # Create sample markdown files
    create_sample_dsa_notes(dsa_dir)
    create_sample_aux_files()
    
    print("Example structure created!")
    print(f"Notes directory: {NOTES_DIR}")
    print(f"DSA course: {dsa_dir}")
    print(f"Auxiliary files: {AUX_DIR}")
    print("\nNext steps:")
    print("1. Add your PDF/markdown files to the course directories")
    print("2. Run: python build_index.py --course DSA")
    print("3. Run: python query.py --course DSA \"What is quicksort?\"")


def create_sample_dsa_notes(dsa_dir: str):
    """Create sample DSA notes."""
    
    # Sorting algorithms note
    sorting_content = """# Sorting Algorithms

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
"""

    with open(os.path.join(dsa_dir, "01-sorting.md"), 'w', encoding='utf-8') as f:
        f.write(sorting_content)
    
    # Data structures note
    ds_content = """# Data Structures

## Arrays

An array is a collection of elements stored at contiguous memory locations. Arrays allow random access to elements using indices.

### Time Complexity
- Access: O(1)
- Search: O(n)
- Insertion: O(n)
- Deletion: O(n)

### Space Complexity
- O(n) where n is the number of elements

## Linked Lists

A linked list is a linear data structure where elements are stored in nodes, and each node contains data and a reference to the next node.

### Types
1. Singly Linked List
2. Doubly Linked List
3. Circular Linked List

### Time Complexity
- Access: O(n)
- Search: O(n)
- Insertion: O(1) at head, O(n) at arbitrary position
- Deletion: O(1) if node is given, O(n) to find and delete

### Space Complexity
- O(n) where n is the number of nodes

## Binary Trees

A binary tree is a hierarchical data structure where each node has at most two children, referred to as left and right child.

### Properties
- Maximum nodes at level i: 2^i
- Maximum nodes in tree of height h: 2^(h+1) - 1
- Minimum height for n nodes: log₂(n+1) - 1
"""

    with open(os.path.join(dsa_dir, "02-data-structures.md"), 'w', encoding='utf-8') as f:
        f.write(ds_content)


def create_sample_aux_files():
    """Create sample auxiliary files."""
    
    # Glossary
    glossary_content = """# Glossary

## A
- **Algorithm**: A step-by-step procedure for solving a problem
- **Array**: A data structure consisting of elements stored at contiguous memory locations

## B
- **Big O**: Notation used to describe the upper bound of algorithm complexity
- **Binary Tree**: A tree data structure where each node has at most two children

## D
- **Divide and Conquer**: An algorithm design paradigm that breaks problems into smaller subproblems

## Q
- **Quicksort**: A fast sorting algorithm using divide-and-conquer approach

## S
- **Stable Sort**: A sorting algorithm that maintains relative order of equal elements
- **Space Complexity**: The amount of memory space an algorithm uses
"""

    with open(os.path.join(AUX_DIR, "glossary.md"), 'w', encoding='utf-8') as f:
        f.write(glossary_content)
    
    # Index exclusions
    exclusions_content = """# Files and patterns to exclude from indexing

# Cover pages
*cover*
*title*

# Table of contents
*toc*
*contents*

# References and bibliography
*references*
*bibliography*
*citations*

# Appendices (optional)
*appendix*
"""

    with open(os.path.join(AUX_DIR, "index_exclusions.txt"), 'w', encoding='utf-8') as f:
        f.write(exclusions_content)


if __name__ == '__main__':
    create_example_structure()
