# geometric-entropy

A Rust implementation of Geometric Partition Entropy (GPE) estimation algorithms from the paper [Generalizing Geometric Partition Entropy for the Estimation of Mutual Information in the Presence of Informative Outliers](https://arxiv.org/abs/2410.17367) by C. Tyler Diggans and Abd AlRahman R. AlMomani.

## Overview

This crate provides tools for estimating entropy and mutual information in continuous state spaces while properly handling informative outliers. Unlike traditional entropy estimators that often minimize or ignore the impact of outliers, this implementation preserves and weights outlier contributions, making it particularly useful for analyzing:

- Synchronization dynamics
- Chaotic systems
- Time series with meaningful transient behaviors
- Data with sparse or uneven sampling

## Key Features

- Partition entropy estimation using the π (partition intersection) estimator
- Proper handling of repeated values within specified tolerance
- Support for multiple measure functions to control outlier impact
- Mutual information estimation with adaptive measure selection
- Efficient implementation for high-dimensional data

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
geometric-entropy = "0.1.0"
ndarray = "0.15"
```

## Quick Start

```rust
use geometric_entropy::GeometricPartitionEntropy;
use ndarray::Array2;

fn main() {
    // Create sample 2D data
    let data = Array2::from_shape_vec((5, 2), vec![
        1.0, 2.0,
        2.0, 3.0,
        3.0, 4.0,
        4.0, 5.0,
        5.0, 6.0,
    ]).unwrap();

    // Initialize estimator
    // epsilon: tolerance for repeated values
    // k: number of partitions per dimension
    let estimator = GeometricPartitionEntropy::new(1e-5, 4).unwrap();

    // Compute entropy
    let entropy = estimator.compute_entropy(&data);
    println!("Geometric Partition Entropy: {:?}", entropy);

    // For mutual information between two datasets
    let data_x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let data_y = Array2::from_shape_vec((5, 1), vec![2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let data_z = Array2::from_shape_vec((5, 1), vec![0.1, 100.1, 45.0, 10.0, 1.0]).unwrap();

    let mi = estimator.compute_mutual_information(&data_x, &data_y).unwrap();
    println!("Mutual Information between X and Y: {:?}", mi);

    let mi = estimator.compute_mutual_information(&data_x, &data_z).unwrap();
    println!("Mutual Information between X and Z: {:?}", mi);
}
```

## Theory

The Geometric Partition Entropy estimator works by:

1. Pre-processing data to handle repeated values within a tolerance ε
2. Creating quantile-based partitions in each dimension
3. Forming partition intersections to create a grid
4. Computing measures using geometric and frequency information
5. Calculating entropy using the partition entropy formula

For mutual information estimation, the method computes:
```
I(X;Y) = H(X) + H(Y) - H(X,Y)
```
where H(X), H(Y) are individual entropies and H(X,Y) is joint entropy.

## Performance

The implementation provides significant performance improvements over traditional k-nearest neighbor approaches:

- Pre-processing: O(n²) for n points
- Partition computation: O(n log n) per dimension
- Grid creation: O(n)
- Overall entropy computation: O(n² + nd log n) for d dimensions

## Citation

If you use this crate in your research, please cite the original paper:

```bibtex
@article{diggans2024generalizing,
  title={Generalizing Geometric Partition Entropy for the Estimation of Mutual Information in the Presence of Informative Outliers},
  author={Diggans, C. Tyler and AlMomani, Abd AlRahman R.},
  journal={arXiv preprint arXiv:2410.17367},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under either of
 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)


## Acknowledgments

- C. Tyler Diggans and Abd AlRahman R. AlMomani for the original research and algorithm development
- The Rust community for excellent scientific computing tools