# Geometric Partition Entropy Specification

## Overview
This crate implements geometric partition entropy estimation algorithms for analyzing mutual information in continuous state spaces, with particular focus on handling informative outliers and repeated values.

## Core Concepts

### Geometric Partition Entropy
Geometric Partition Entropy (GPE) is an entropy estimation approach that:
- Works on continuous state spaces
- Handles repeated values within a specified tolerance
- Incorporates the impact of outliers
- Uses data-driven partitioning rather than uniform bins

### Fundamental Requirements

1. All input data must exist within a bounded (finite measure) region D ⊂ ℝᵈ
2. A tolerance parameter ε must be specified for handling repeated values
3. A partition parameter k must be specified for each dimension
4. The implementation must support different measure functions g(L,F)

## Data Structures

### GeometricPartitionEntropy
```rust
pub struct GeometricPartitionEntropy {
    epsilon: f64,  // Tolerance for repeated values
    k: usize,     // Number of partitions per dimension
}
```

#### Parameters:
- `epsilon`: Defines the tolerance for considering two points as repeated values
- `k`: Number of partitions per dimension for the π estimator

## Core Algorithms

### 1. Pre-processing (Repeated Values Handling)

#### Purpose:
Process input data to properly handle repeated values within the specified tolerance.

#### Algorithm:
1. For each pair of points in the dataset:
   - Calculate Chebyshev distance between points
   - If distance ≤ epsilon, merge points by averaging their values
2. Output processed dataset with effectively unique points

#### Requirements:
- Must use Chebyshev distance (L∞ norm) for point comparisons
- Must preserve original data structure/dimensions
- Must handle edge cases (empty data, single points)

### 2. Quantile-based Partitioning

#### Purpose:
Create partitions in each dimension based on quantiles of the data.

#### Algorithm:
1. For each dimension:
   - Sort dimension values
   - Compute k+1 partition boundaries:
     * First boundary = minimum value
     * Last boundary = maximum value
     * Internal boundaries = (i/k)-th quantiles for i=1..k-1
2. Return vector of partition boundaries for each dimension

#### Requirements:
- Must handle non-uniform distributions effectively
- Must ensure no empty partitions
- Must include full data range (min to max)

### 3. Partition Intersection Grid

#### Purpose:
Create a grid based on the intersection of quantile partitions across all dimensions.

#### Algorithm:
1. Create grid array of size k^d where d is number of dimensions
2. For each data point:
   - Compute its grid cell index based on partition boundaries
   - Increment count for that cell
3. Return grid with point counts per cell

#### Requirements:
- Must handle arbitrary dimensions
- Must provide O(n) performance where possible
- Must correctly map points to grid cells

### 4. Measure Computation

#### Purpose:
Compute measures for each grid cell using specified g(L,F) function.

#### Supported Measure Functions:
1. g(L,F) = L/F (default)
   - L = geometric measure (cell volume)
   - F = frequency (point count)
2. g(L,F) = L·F
3. g(L,F) = F·e^(-L)
4. g(L,F) = L·e^(-F)

#### Requirements:
- Must normalize measures to sum to 1
- Must handle empty cells appropriately
- Must support switching between measure functions

### 5. Entropy Computation

#### Purpose:
Calculate final entropy value using partition entropy formula.

#### Formula:
H = -∑ᵢ μ(Aᵢ)log₂(μ(Aᵢ))

where:
- μ(Aᵢ) is the measure of partition element i
- Sum only includes elements with positive measure

#### Requirements:
- Must filter out zero measures before computing
- Must use log base 2 for computation
- Must handle numerical stability issues

### 6. Mutual Information Computation

#### Purpose:
Calculate mutual information between two datasets.

#### Formula:
I(X;Y) = H(X) + H(Y) - H(X,Y)

where:
- H(X), H(Y) are individual entropies
- H(X,Y) is joint entropy

#### Requirements:
- Must handle datasets of different dimensions
- Must ensure datasets have same number of samples
- Must handle negative MI estimates appropriately:
  * If using g(L,F) = L/F yields negative MI, automatically switch to g(L,F) = F·e^(-L)

## Error Handling

### Required Error Cases:
1. Empty input data
2. Mismatched dimensions in MI computation
3. Invalid parameters (negative epsilon, zero k)
4. Numerical overflow/underflow conditions
5. NaN or infinite values in input data

### Error Types:
Implementation should define appropriate error types for each case.

## Performance Requirements

### Time Complexity Targets:
- Pre-processing: O(n²) where n is number of points
- Partition computation: O(n log n) per dimension
- Grid creation: O(n)
- Overall entropy computation: O(n² + nd log n) where d is number of dimensions

### Memory Complexity Targets:
- Should not exceed O(n + k^d) where:
  * n is number of points
  * k is partitions per dimension
  * d is number of dimensions

## Testing Requirements

### Unit Tests Must Cover:
1. Basic entropy computation on uniform distributions
2. Repeated value handling
3. Outlier impact verification
4. Different measure functions
5. Edge cases (single point, all same points)
6. Higher dimensional cases
7. Mutual information computation
8. Error handling cases

### Properties to Test:
1. Non-negativity of entropy
2. Scaling with dimension
3. Consistency with known distributions
4. Impact of different epsilon values
5. Impact of different k values

## Documentation Requirements

### Must Include:
1. Theoretical background
2. Usage examples
3. Parameter selection guidelines
4. Performance characteristics
5. Limitations and constraints
6. Error handling guidance
7. Implementation notes

## Version Requirements

### Minimum Rust Version:
Implementation should target stable Rust and specify MSRV.

### Dependencies:
- ndarray: For efficient array operations
- Other dependencies should be minimized

## Future Extensions

### Planned Features:
1. Alternative partition schemes (box estimator)
2. Parallel processing support
3. Additional measure functions
4. Adaptive parameter selection
5. GPU acceleration support
6. Streaming data support