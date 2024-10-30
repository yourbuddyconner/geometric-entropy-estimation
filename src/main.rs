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
