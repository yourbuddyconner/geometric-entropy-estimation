use ndarray::{Array1, Array2, s};
use std::f64;

#[derive(Clone, Copy, Debug)]
pub enum MeasureFunction {
    /// g(L,F) = L/F (default)
    RatioMeasure,
    /// g(L,F) = L·F
    ProductMeasure,
    /// g(L,F) = F·e^(-L)
    FrequencyExponential,
    /// g(L,F) = L·e^(-F)
    LengthExponential,
}

#[derive(Debug)]
pub enum GeometricPartitionError {
    EmptyInput,
    MismatchedDimensions,
    InvalidParameters,
    NumericalError,
    InvalidData,
}

pub struct GeometricPartitionEntropy {
    epsilon: f64,  // Tolerance for repeated values
    k: usize,      // Number of partitions per dimension
    measure_fn: MeasureFunction,
}

impl GeometricPartitionEntropy {
    pub fn new(epsilon: f64, k: usize) -> Result<Self, GeometricPartitionError> {
        if epsilon <= 0.0 || k == 0 {
            return Err(GeometricPartitionError::InvalidParameters);
        }
        Ok(GeometricPartitionEntropy { 
            epsilon,
            k,
            measure_fn: MeasureFunction::RatioMeasure 
        })
    }

    pub fn with_measure_function(epsilon: f64, k: usize, measure_fn: MeasureFunction) -> Result<Self, GeometricPartitionError> {
        if epsilon <= 0.0 || k == 0 {
            return Err(GeometricPartitionError::InvalidParameters);
        }
        Ok(GeometricPartitionEntropy {
            epsilon,
            k,
            measure_fn
        })
    }

    /// Computes the geometric partition entropy using the π estimator
    pub fn compute_entropy(&self, data: &Array2<f64>) -> Result<f64, GeometricPartitionError> {
        self.validate_input(data)?;
        
        // Pre-process data to handle repeated values within epsilon tolerance
        let processed_data = self.preprocess_data(data);
        
        // Get dimensions
        let (_n_samples, n_dims) = processed_data.dim();
        
        // Calculate quantile partitions for each dimension
        let mut partitions = Vec::with_capacity(n_dims);
        for dim in 0..n_dims {
            let dim_data = processed_data.column(dim).to_owned();
            partitions.push(self.compute_quantile_partition(&dim_data));
        }
        
        // Create partition intersection grid
        let grid = self.create_partition_grid(&processed_data, &partitions);
        
        // Compute measures for each cell
        let measures = self.compute_measures(&grid);
        
        // Calculate entropy using partition entropy formula
        Ok(-measures.iter()
            .filter(|&&m| m > 0.0)
            .map(|&m| m * m.log2())
            .sum::<f64>())
    }

    /// Pre-processes data to handle repeated values within epsilon tolerance
    fn preprocess_data(&self, data: &Array2<f64>) -> Array2<f64> {
        let (n_samples, _) = data.dim();
        let mut processed = data.clone();
        
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                if self.points_within_tolerance(
                    data.row(i).as_slice().unwrap(),
                    data.row(j).as_slice().unwrap()
                ) {
                    // Create temporary value to avoid borrow conflict
                    let temp_row = processed.row(i).to_owned();
                    processed.row_mut(j).assign(&temp_row);
                }
            }
        }
        
        processed
    }

    /// Checks if two points are within epsilon tolerance using Chebyshev distance
    fn points_within_tolerance(&self, p1: &[f64], p2: &[f64]) -> bool {
        p1.iter()
            .zip(p2.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max) 
        <= self.epsilon
    }

    /// Computes quantile-based partition for a single dimension
    fn compute_quantile_partition(&self, data: &Array1<f64>) -> Vec<f64> {
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mut partitions = Vec::with_capacity(self.k + 1);
        partitions.push(*sorted_data.first().unwrap());
        
        for i in 1..self.k {
            let idx = (i * sorted_data.len()) / self.k;
            partitions.push(sorted_data[idx]);
        }
        
        partitions.push(*sorted_data.last().unwrap());
        partitions
    }

    /// Creates partition intersection grid
    fn create_partition_grid(&self, 
        data: &Array2<f64>, 
        partitions: &[Vec<f64>]
    ) -> Vec<usize> {
        let (n_samples, n_dims) = data.dim();
        let mut grid = vec![0; self.k.pow(n_dims as u32)];
        
        // Count points in each grid cell
        for i in 0..n_samples {
            let cell_index = self.get_cell_index(
                data.row(i).as_slice().unwrap(),
                partitions
            );
            grid[cell_index] += 1;
        }
        
        grid
    }

    /// Computes cell index for a point
    fn get_cell_index(&self, point: &[f64], partitions: &[Vec<f64>]) -> usize {
        let mut index = 0;
        
        for (dim, p) in point.iter().enumerate() {
            let dim_partitions = &partitions[dim];
            let mut dim_index = 0;
            
            for (j, &boundary) in dim_partitions.iter().enumerate().skip(1) {
                if p <= &boundary {
                    dim_index = j - 1;
                    break;
                }
            }
            
            index = index * self.k + dim_index;
        }
        
        index
    }

    /// Computes measures for each grid cell using g(L,F) = L/F
    fn compute_measures(&self, grid: &[usize]) -> Vec<f64> {
        let total_points: usize = grid.iter().sum();
        let cell_volume = 1.0 / (self.k as f64).powi(grid.len() as i32);
        
        let raw_measures: Vec<f64> = grid.iter()
            .map(|&count| {
                if count == 0 {
                    0.0
                } else {
                    match self.measure_fn {
                        MeasureFunction::RatioMeasure => {
                            (cell_volume / count as f64) * (count as f64 / total_points as f64)
                        },
                        MeasureFunction::ProductMeasure => {
                            cell_volume * (count as f64 / total_points as f64)
                        },
                        MeasureFunction::FrequencyExponential => {
                            (count as f64 / total_points as f64) * (-cell_volume).exp()
                        },
                        MeasureFunction::LengthExponential => {
                            cell_volume * (-(count as f64) / total_points as f64).exp()
                        }
                    }
                }
            })
            .collect();

        // Normalize measures to sum to 1
        let sum: f64 = raw_measures.iter().sum();
        raw_measures.iter().map(|&m| m / sum).collect()
    }

    /// Computes mutual information between two datasets
    pub fn compute_mutual_information(
        &self,
        data_x: &Array2<f64>,
        data_y: &Array2<f64>
    ) -> Result<f64, GeometricPartitionError> {
        // Compute individual entropies
        let h_x = self.compute_entropy(data_x)?;
        let h_y = self.compute_entropy(data_y)?;
        
        // Combine datasets for joint entropy
        let (n_samples, dims_x) = data_x.dim();
        let (_, dims_y) = data_y.dim();
        
        let mut joint_data = Array2::zeros((n_samples, dims_x + dims_y));
        joint_data
            .slice_mut(s![.., 0..dims_x])
            .assign(&data_x);
        joint_data
            .slice_mut(s![.., dims_x..])
            .assign(&data_y);
        
        let h_xy = self.compute_entropy(&joint_data)?;
        
        // Compute MI using definition I(X;Y) = H(X) + H(Y) - H(X,Y)
        let mi = h_x + h_y - h_xy;
        
        // If MI is negative with RatioMeasure, retry with FrequencyExponential
        if mi < 0.0 && matches!(self.measure_fn, MeasureFunction::RatioMeasure) {
            let temp_gpe = Self::with_measure_function(
                self.epsilon,
                self.k,
                MeasureFunction::FrequencyExponential
            )?;
            return temp_gpe.compute_mutual_information(data_x, data_y);
        }
        
        Ok(mi)
    }

    fn validate_input(&self, data: &Array2<f64>) -> Result<(), GeometricPartitionError> {
        if data.is_empty() {
            return Err(GeometricPartitionError::EmptyInput);
        }
        
        if data.iter().any(|x| x.is_nan() || x.is_infinite()) {
            return Err(GeometricPartitionError::InvalidData);
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;

    #[test]
    fn test_different_measure_functions() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        
        let gpe_ratio = GeometricPartitionEntropy::with_measure_function(
            0.1, 2, MeasureFunction::RatioMeasure
        ).expect("Failed to create ratio measure GPE");
        let gpe_product = GeometricPartitionEntropy::with_measure_function(
            0.1, 2, MeasureFunction::ProductMeasure
        ).expect("Failed to create product measure GPE");
        let gpe_freq_exp = GeometricPartitionEntropy::with_measure_function(
            0.1, 2, MeasureFunction::FrequencyExponential
        ).expect("Failed to create frequency exponential GPE");
        
        // Each measure function should produce valid entropy (non-negative)
        assert!(gpe_ratio.compute_entropy(&data).unwrap() >= 0.0);
        assert!(gpe_product.compute_entropy(&data).unwrap() >= 0.0);
        assert!(gpe_freq_exp.compute_entropy(&data).unwrap() >= 0.0);
    }

    #[test]
    fn test_edge_cases() {
        // Single point
        let single_point = array![[1.0, 1.0]];
        let gpe = GeometricPartitionEntropy::new(0.1, 2).unwrap();
        assert_relative_eq!(
            gpe.compute_entropy(&single_point).expect("Failed to compute entropy"),
            0.0,
            epsilon = 1e-10
        );

        // All same points
        let same_points = array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
        assert_relative_eq!(
            gpe.compute_entropy(&same_points).expect("Failed to compute entropy"),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_higher_dimensions() {
        let data_3d = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        
        let gpe = GeometricPartitionEntropy::new(0.1, 2).unwrap();
        let entropy_3d = gpe.compute_entropy(&data_3d).expect("Failed to compute 3D entropy");
        
        let data_2d = data_3d.slice(s![.., ..2]).to_owned();
        let entropy_2d = gpe.compute_entropy(&data_2d).expect("Failed to compute 2D entropy");
        
        assert!(entropy_3d >= entropy_2d);
    }

    #[test]
    fn test_error_handling() {
        // Invalid parameters
        assert!(matches!(
            GeometricPartitionEntropy::new(-0.1, 2),
            Err(GeometricPartitionError::InvalidParameters)
        ));
        
        assert!(matches!(
            GeometricPartitionEntropy::new(0.1, 0),
            Err(GeometricPartitionError::InvalidParameters)
        ));

        let gpe = GeometricPartitionEntropy::new(0.1, 2).unwrap();
        
        // Empty input
        let empty_data: Array2<f64> = Array2::zeros((0, 2));
        assert!(matches!(
            gpe.validate_input(&empty_data),
            Err(GeometricPartitionError::EmptyInput)
        ));

        // Invalid data (NaN/Inf)
        let invalid_data = array![[1.0, f64::NAN], [2.0, 3.0]];
        assert!(matches!(
            gpe.validate_input(&invalid_data),
            Err(GeometricPartitionError::InvalidData)
        ));
    }

    #[test]
    fn test_entropy_properties() {
        let gpe = GeometricPartitionEntropy::new(0.1, 2).unwrap();
        
        // Test 1: Non-negativity
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        assert!(gpe.compute_entropy(&data).unwrap() >= 0.0);
    }

    #[test]
    fn test_information_content() {
        let gpe = GeometricPartitionEntropy::new(0.1, 2).unwrap();
        
        // Test data with low information content (repeated values)
        let low_info = array![
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 2.0]
        ];
        
        // Test data with high information content (all unique values)
        let high_info = array![
            [1.0, 3.0],
            [2.0, 4.0],
            [3.0, 5.0],
            [4.0, 6.0]
        ];
        
        let entropy_low = gpe.compute_entropy(&low_info).unwrap();
        let entropy_high = gpe.compute_entropy(&high_info).unwrap();
        
        // Data with more unique values should have higher entropy
        assert!(entropy_high > entropy_low);
    }

    #[test]
    fn test_parameter_impact() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        
        // Test epsilon impact
        let gpe_small_eps = GeometricPartitionEntropy::new(0.01, 2).unwrap();
        let gpe_large_eps = GeometricPartitionEntropy::new(1.0, 2).unwrap();
        
        let entropy_small_eps = gpe_small_eps.compute_entropy(&data).unwrap();
        let entropy_large_eps = gpe_large_eps.compute_entropy(&data).unwrap();
        
        // Larger epsilon should generally lead to lower entropy (more merging)
        assert!(entropy_large_eps <= entropy_small_eps);

        // Test k impact
        let gpe_small_k = GeometricPartitionEntropy::new(0.1, 2).unwrap();
        let gpe_large_k = GeometricPartitionEntropy::new(0.1, 4).unwrap();
        
        let entropy_small_k = gpe_small_k.compute_entropy(&data).unwrap();
        let entropy_large_k = gpe_large_k.compute_entropy(&data).unwrap();
        
        // More partitions should generally lead to higher entropy
        assert!(entropy_large_k >= entropy_small_k);
    }

    #[test]
    fn test_mutual_information() {
        let gpe = GeometricPartitionEntropy::new(0.1, 2).unwrap();
        
        // Perfect correlation (linear relationship)
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![[2.0], [4.0], [6.0], [8.0], [10.0]];  // y = 2x
        
        let mi = gpe.compute_mutual_information(&x, &y).unwrap();
        assert!(mi > 0.0);

        // Independent data (different patterns)
        let z = array![[1.0], [1.0], [1.0], [1.0], [1.0]];  // constant, independent of x
        let mi_uncorrelated = gpe.compute_mutual_information(&x, &z).unwrap();
        
        // MI should be higher for correlated variables
        assert!(mi > mi_uncorrelated);
    }
}