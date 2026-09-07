use crate::error::{invalid, KernelResult};
use crate::linalg::{
    center_rows, checked_finite, checked_matrix, checked_output, checked_vector, checked_work,
    matrix_rank, mean_rows, orthonormalize_rows, solve, symmetric_eigen,
    thin_covariance_components, thin_principal_components, validate_orthonormal,
};

const NEAR_ZERO_VARIANCE: f64 = 1e-12;
const RBF_GCV_GRID_POINTS: usize = 40;
const RBF_MAX_SYSTEM_DIMENSION: usize = 512;
const GRAPH_SYMMETRY_TOLERANCE: f64 = 1e-9;
const PERSISTENCE_MAX_NODES: usize = 128;
const PERSISTENCE_MAX_TRIANGLES: usize = 500_000;
const CYCLE_MIN_NODES: usize = 7;
const CYCLE_MAX_NODES: usize = 128;
const CYCLE_MAX_DEGREE: usize = 3;
const CYCLE_MAX_DEGREE_CLUSTER: usize = 4;
const CYCLE_CLOSURE_MAX: f64 = 2.0;
const CYCLE_RECALL_MIN: f64 = 0.90;
const CYCLE_CONTRAST_MIN: f64 = 1.08;
const CYCLE_LARGE_FACTOR: f64 = 2.5;
const CYCLE_GAPS_MIN: usize = 2;
const CYCLE_GAP_REG_MAX: f64 = 2.5;
const CYCLE_ANTIPODE_MIN: f64 = 2.5;
const CYCLE_BIMODAL_MIN: f64 = 3.5;
const HARMONIC_COHERENCE: f64 = 0.80;
const HARMONIC_MAX_ORDER: usize = 5;

#[derive(Debug, Clone, PartialEq)]
pub struct Centering {
    pub rows: usize,
    pub columns: usize,
    pub mean: Vec<f64>,
    pub centered: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MahalanobisWhitener {
    pub columns: usize,
    pub rank: usize,
    pub mean: Vec<f64>,
    pub basis: Vec<f64>,
    pub eigenvalues: Vec<f64>,
    pub inverse_scales: Vec<f64>,
    pub ridge: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pca {
    pub rows: usize,
    pub columns: usize,
    pub components: usize,
    pub mean: Vec<f64>,
    pub basis: Vec<f64>,
    pub eigenvalues: Vec<f64>,
    pub explained_variance: Vec<f64>,
    pub cumulative_variance: Vec<f64>,
    pub scores: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AffineFisherFit {
    pub node_count: usize,
    pub columns: usize,
    pub components: usize,
    pub centroid_mean: Vec<f64>,
    pub mean: Vec<f64>,
    pub basis: Vec<f64>,
    pub node_coordinates: Vec<f64>,
    pub mu_coordinates: Vec<f64>,
    pub whitened_gram: Vec<f64>,
    pub neutral_cross_gram: Vec<f64>,
    pub explained_variance: f64,
    pub mahalanobis_share: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KnnGraph {
    pub node_count: usize,
    pub k_nn: usize,
    pub mask: Vec<u8>,
    pub neighbor_distances: Vec<f64>,
    pub component_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedLaplacian {
    pub node_count: usize,
    pub k_nn: usize,
    pub bandwidth: f64,
    pub mask: Vec<u8>,
    pub weights: Vec<f64>,
    pub degrees: Vec<f64>,
    pub laplacian: Vec<f64>,
    pub nontrivial_eigenvalues: Vec<f64>,
    pub nontrivial_eigenvectors: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpectralEmbedding {
    pub node_count: usize,
    pub dimensions: usize,
    pub heuristic_dimensions: usize,
    pub min_dimensions: Option<usize>,
    pub pinned: bool,
    pub k_nn: usize,
    pub bandwidth: f64,
    pub gap_magnitude: f64,
    pub eigenvalues: Vec<f64>,
    pub coordinates: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcaLayout {
    pub node_count: usize,
    pub dimensions: usize,
    pub explained_variance: Vec<f64>,
    pub cumulative_variance: Vec<f64>,
    pub coordinates: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicTopology {
    pub node_count: usize,
    pub dimensions: usize,
    pub persistent_loops: usize,
    pub used_faint_cycle: bool,
    pub angles: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TopologyCandidate {
    pub name: String,
    pub fit_mode: String,
    pub intrinsic_dimensions: usize,
    pub score: f64,
    pub viable: bool,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct TopologySelection {
    pub winner_name: String,
    pub fit_mode: String,
    pub intrinsic_dimensions: usize,
    pub periodic_dimensions: usize,
    pub coordinates: Vec<f64>,
    pub embedded_coordinates: Vec<f64>,
    pub candidates: Vec<TopologyCandidate>,
    pub persistent_loops: usize,
    pub used_faint_cycle: bool,
    pub diagnostics: TopologyDiagnostics,
    pub winner_plan: Option<RbfFitPlan>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TopologyDiagnostics {
    Pca {
        per_component_variance: Vec<f64>,
        cumulative_variance: Vec<f64>,
        picked_dimensions: usize,
        threshold: f64,
    },
    Spectral {
        eigenvalues: Vec<f64>,
        picked_dimensions: usize,
        gap_magnitude: f64,
        bandwidth: f64,
        k_nn: usize,
        heuristic_dimensions: usize,
        min_dimensions: Option<usize>,
        pinned: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RbfModel {
    pub input_dimensions: usize,
    pub output_dimensions: usize,
    pub node_count: usize,
    pub nodes: Vec<f64>,
    pub coordinate_offset: Vec<f64>,
    pub coordinate_scale: Vec<f64>,
    pub weights: Vec<f64>,
    pub polynomial: Vec<f64>,
    pub lambda: f64,
    pub effective_degrees_of_freedom: f64,
    pub gcv: f64,
}

#[derive(Debug, Clone)]
pub struct GroupedReducedCovarianceAccumulator {
    columns: usize,
    components: usize,
    offsets: Vec<usize>,
    center: Vec<f64>,
    basis: Vec<f64>,
    counts: Vec<usize>,
    means: Vec<f64>,
    second_moments: Vec<f64>,
    next_row: usize,
}

impl GroupedReducedCovarianceAccumulator {
    pub fn new(
        columns: usize,
        center: &[f64],
        basis: &[f64],
        components: usize,
        offsets: &[u32],
    ) -> KernelResult<Self> {
        if columns == 0 || components == 0 {
            return Err(invalid("reduced covariance dimensions must be positive"));
        }
        checked_vector("reduced covariance center", center, columns)?;
        checked_matrix("reduced covariance basis", basis, components, columns)?;
        if offsets.len() < 2 || offsets[0] != 0 {
            return Err(invalid(
                "reduced covariance offsets must start at zero and contain a group",
            ));
        }
        if offsets.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid(
                "reduced covariance offsets must be strictly increasing",
            ));
        }
        let node_count = offsets.len() - 1;
        let mean_elements = checked_output(
            "reduced covariance running means",
            &[node_count, components],
        )?;
        let covariance_elements = checked_output(
            "reduced covariance second moments",
            &[node_count, components, components],
        )?;
        checked_work(
            "reduced covariance accumulator",
            &[
                center.len(),
                basis.len(),
                mean_elements,
                covariance_elements,
            ],
        )?;
        Ok(Self {
            columns,
            components,
            offsets: offsets.iter().map(|value| *value as usize).collect(),
            center: center.to_vec(),
            basis: basis.to_vec(),
            counts: vec![0; node_count],
            means: vec![0.0; mean_elements],
            second_moments: vec![0.0; covariance_elements],
            next_row: 0,
        })
    }

    pub fn append(&mut self, values: &[f32], start_row: usize, rows: usize) -> KernelResult<()> {
        if start_row != self.next_row {
            return Err(invalid(format!(
                "reduced covariance rows are not contiguous: got {start_row}, expected {}",
                self.next_row
            )));
        }
        let expected = rows
            .checked_mul(self.columns)
            .ok_or_else(|| invalid("reduced covariance chunk size overflows usize"))?;
        if values.len() != expected {
            return Err(invalid(format!(
                "reduced covariance chunk has {} values, expected {expected}",
                values.len()
            )));
        }
        let total_rows = *self.offsets.last().expect("validated offsets");
        if start_row
            .checked_add(rows)
            .is_none_or(|end| end > total_rows)
        {
            return Err(invalid("reduced covariance chunk exceeds the group roster"));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid("reduced covariance rows contain non-finite values"));
        }

        let mut reduced = vec![0.0; self.components];
        let mut delta = vec![0.0; self.components];
        let mut delta_after = vec![0.0; self.components];
        for local_row in 0..rows {
            let global_row = start_row + local_row;
            let group = self.offsets[1..].partition_point(|offset| *offset <= global_row);
            let row = &values[local_row * self.columns..(local_row + 1) * self.columns];
            for (component, reduced_value) in reduced.iter_mut().enumerate() {
                *reduced_value = row
                    .iter()
                    .zip(&self.center)
                    .enumerate()
                    .map(|(column, (value, center))| {
                        (f64::from(*value) - center) * self.basis[component * self.columns + column]
                    })
                    .sum();
            }

            self.counts[group] += 1;
            let count = self.counts[group] as f64;
            let mean_start = group * self.components;
            for component in 0..self.components {
                let index = mean_start + component;
                delta[component] = reduced[component] - self.means[index];
                self.means[index] += delta[component] / count;
                delta_after[component] = reduced[component] - self.means[index];
            }
            let moment_start = group * self.components * self.components;
            for (left, delta_left) in delta.iter().copied().enumerate() {
                for (right, delta_right) in delta_after.iter().copied().enumerate() {
                    self.second_moments[moment_start + left * self.components + right] +=
                        delta_left * delta_right;
                }
            }
        }
        self.next_row += rows;
        Ok(())
    }

    pub fn finalize(&self) -> KernelResult<Vec<f64>> {
        let total_rows = *self.offsets.last().expect("validated offsets");
        if self.next_row != total_rows {
            return Err(invalid(format!(
                "reduced covariance accumulator has {} of {total_rows} rows",
                self.next_row
            )));
        }
        let mut covariances = self.second_moments.clone();
        for (group, count) in self.counts.iter().copied().enumerate() {
            let start = group * self.components * self.components;
            if count <= 1 {
                covariances[start..start + self.components * self.components].fill(0.0);
                continue;
            }
            let denominator = (count - 1) as f64;
            for left in 0..self.components {
                for right in left..self.components {
                    let left_index = start + left * self.components + right;
                    let right_index = start + right * self.components + left;
                    let symmetric =
                        0.5 * (covariances[left_index] + covariances[right_index]) / denominator;
                    covariances[left_index] = symmetric;
                    covariances[right_index] = symmetric;
                }
            }
        }
        checked_finite("reduced covariances", &covariances)?;
        Ok(covariances)
    }

    pub fn node_count(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn components(&self) -> usize {
        self.components
    }
}

#[derive(Debug, Clone)]
pub struct RbfFitPlan {
    geometry: RbfGeometry,
    lambdas: Vec<f64>,
    spectral_basis: Vec<f64>,
    residual_ratios: Vec<f64>,
    residual_traces: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SigmaFieldFit {
    pub model: RbfModel,
    pub sigma_mean: f64,
    pub sigma_min: f64,
    pub sigma_max: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RbfOriginFit {
    pub coordinates: Vec<f64>,
    pub distance: f64,
}

#[derive(Debug, Clone)]
struct RbfGeometry {
    input_dimensions: usize,
    node_count: usize,
    nodes: Vec<f64>,
    coordinate_offset: Vec<f64>,
    coordinate_scale: Vec<f64>,
    kernel: Vec<f64>,
    kernel_scale: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TemplateScoreProbabilities {
    pub sum_probabilities: Vec<f64>,
    pub mean_log_probabilities: Vec<f64>,
    pub mean_probabilities: Vec<f64>,
}

pub fn fit_neutral_centering(
    values: &[f64],
    rows: usize,
    columns: usize,
) -> KernelResult<Centering> {
    checked_matrix("neutral activations", values, rows, columns)?;
    checked_work("neutral centering", &[values.len(), values.len(), columns])?;
    let mean = mean_rows(values, rows, columns);
    let centered = center_rows(values, columns, &mean);
    checked_finite("neutral mean", &mean)?;
    checked_finite("centered neutral activations", &centered)?;
    Ok(Centering {
        rows,
        columns,
        mean,
        centered,
    })
}

pub fn group_row_means(
    values: &[f64],
    rows: usize,
    columns: usize,
    offsets: &[u32],
) -> KernelResult<Vec<f64>> {
    checked_matrix("grouped activations", values, rows, columns)?;
    if offsets.len() < 2 || offsets[0] != 0 || offsets[offsets.len() - 1] as usize != rows {
        return Err(invalid(
            "group offsets must start at zero and end at the matrix row count",
        ));
    }
    if offsets.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(invalid("group offsets must be strictly increasing"));
    }
    let groups = offsets.len() - 1;
    let output_elements = checked_output("group means", &[groups, columns])?;
    checked_work("group means", &[values.len(), output_elements])?;
    let mut output = vec![0.0; output_elements];
    for group in 0..groups {
        let start = offsets[group] as usize;
        let end = offsets[group + 1] as usize;
        let count = (end - start) as f64;
        for row in start..end {
            for column in 0..columns {
                output[group * columns + column] += values[row * columns + column] / count;
            }
        }
    }
    checked_finite("group means", &output)?;
    Ok(output)
}

pub fn apply_centering(
    values: &[f64],
    rows: usize,
    columns: usize,
    mean: &[f64],
) -> KernelResult<Vec<f64>> {
    checked_matrix("values", values, rows, columns)?;
    checked_vector("neutral mean", mean, columns)?;
    checked_work("centering", &[values.len(), values.len(), mean.len()])?;
    let centered = center_rows(values, columns, mean);
    checked_finite("centered values", &centered)?;
    Ok(centered)
}

pub fn fit_mahalanobis_whitener(
    values: &[f64],
    rows: usize,
    columns: usize,
    ridge_scale: f64,
) -> KernelResult<MahalanobisWhitener> {
    if !ridge_scale.is_finite() || ridge_scale <= 0.0 {
        return Err(invalid("ridge_scale must be finite and positive"));
    }
    let centering = fit_neutral_centering(values, rows, columns)?;
    let frobenius_squared = centering
        .centered
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    if !frobenius_squared.is_finite() {
        return Err(invalid("neutral covariance scale is not finite"));
    }
    let ridge = if frobenius_squared <= NEAR_ZERO_VARIANCE {
        1.0
    } else {
        (frobenius_squared / (rows * columns) as f64) * ridge_scale
    };
    if !ridge.is_finite() || ridge <= 0.0 {
        return Err(invalid(
            "neutral covariance ridge is not finite and positive",
        ));
    }
    let (eigenvalues, basis) = if frobenius_squared <= NEAR_ZERO_VARIANCE {
        (Vec::new(), Vec::new())
    } else {
        thin_covariance_components(&centering.centered, rows, columns)?
    };
    let rank = eigenvalues.len();
    let inverse_scales: Vec<f64> = eigenvalues
        .iter()
        .map(|eigenvalue| 1.0 / (eigenvalue + ridge).sqrt())
        .collect();
    checked_finite("whitener eigenvalues", &eigenvalues)?;
    checked_finite("whitener basis", &basis)?;
    checked_finite("whitener inverse scales", &inverse_scales)?;
    Ok(MahalanobisWhitener {
        columns,
        rank,
        mean: centering.mean,
        basis,
        eigenvalues,
        inverse_scales,
        ridge,
    })
}

pub fn apply_mahalanobis_whitener(
    whitener: &MahalanobisWhitener,
    values: &[f64],
    rows: usize,
) -> KernelResult<Vec<f64>> {
    checked_matrix("values", values, rows, whitener.columns)?;
    checked_vector("whitener mean", &whitener.mean, whitener.columns)?;
    if !whitener.ridge.is_finite() || whitener.ridge <= 0.0 {
        return Err(invalid("whitener ridge must be finite and positive"));
    }
    if whitener.rank == 0 {
        if !whitener.basis.is_empty()
            || !whitener.eigenvalues.is_empty()
            || !whitener.inverse_scales.is_empty()
        {
            return Err(invalid("rank-zero whitener must have empty factors"));
        }
    } else {
        validate_orthonormal(&whitener.basis, whitener.rank, whitener.columns)?;
        checked_vector("whitener eigenvalues", &whitener.eigenvalues, whitener.rank)?;
        checked_vector(
            "whitener inverse scales",
            &whitener.inverse_scales,
            whitener.rank,
        )?;
        if whitener.eigenvalues.iter().any(|value| *value <= 0.0)
            || whitener.inverse_scales.iter().any(|value| *value <= 0.0)
        {
            return Err(invalid("whitener factors must be positive"));
        }
    }
    let output_elements = checked_output("whitened values", &[rows, whitener.columns])?;
    checked_work(
        "Mahalanobis whitening",
        &[values.len(), output_elements, whitener.basis.len()],
    )?;
    let complement_scale = 1.0 / whitener.ridge.sqrt();
    let mut output = vec![0.0; output_elements];
    for row in 0..rows {
        for column in 0..whitener.columns {
            output[row * whitener.columns + column] = (values[row * whitener.columns + column]
                - whitener.mean[column])
                * complement_scale;
        }
        for component in 0..whitener.rank {
            let coordinate = (0..whitener.columns)
                .map(|column| {
                    (values[row * whitener.columns + column] - whitener.mean[column])
                        * whitener.basis[component * whitener.columns + column]
                })
                .sum::<f64>();
            let correction = coordinate * (whitener.inverse_scales[component] - complement_scale);
            for column in 0..whitener.columns {
                output[row * whitener.columns + column] +=
                    correction * whitener.basis[component * whitener.columns + column];
            }
        }
    }
    checked_finite("whitened values", &output)?;
    Ok(output)
}

fn apply_mahalanobis_inverse_centered(
    whitener: &MahalanobisWhitener,
    values: &[f64],
    rows: usize,
) -> KernelResult<Vec<f64>> {
    checked_matrix("centered values", values, rows, whitener.columns)?;
    validate_whitener(whitener)?;
    let mut output = values
        .iter()
        .map(|value| value / whitener.ridge)
        .collect::<Vec<_>>();
    for row in 0..rows {
        for component in 0..whitener.rank {
            let coordinate = (0..whitener.columns)
                .map(|column| {
                    values[row * whitener.columns + column]
                        * whitener.basis[component * whitener.columns + column]
                })
                .sum::<f64>();
            let correction = coordinate
                * (1.0 / (whitener.eigenvalues[component] + whitener.ridge) - 1.0 / whitener.ridge);
            for column in 0..whitener.columns {
                output[row * whitener.columns + column] +=
                    correction * whitener.basis[component * whitener.columns + column];
            }
        }
    }
    checked_finite("inverse-whitened values", &output)?;
    Ok(output)
}

fn validate_whitener(whitener: &MahalanobisWhitener) -> KernelResult<()> {
    checked_vector("whitener mean", &whitener.mean, whitener.columns)?;
    if !whitener.ridge.is_finite() || whitener.ridge <= 0.0 {
        return Err(invalid("whitener ridge must be finite and positive"));
    }
    if whitener.rank == 0 {
        if !whitener.basis.is_empty()
            || !whitener.eigenvalues.is_empty()
            || !whitener.inverse_scales.is_empty()
        {
            return Err(invalid("rank-zero whitener must have empty factors"));
        }
        return Ok(());
    }
    validate_orthonormal(&whitener.basis, whitener.rank, whitener.columns)?;
    checked_vector("whitener eigenvalues", &whitener.eigenvalues, whitener.rank)?;
    checked_vector(
        "whitener inverse scales",
        &whitener.inverse_scales,
        whitener.rank,
    )?;
    if whitener.eigenvalues.iter().any(|value| *value <= 0.0)
        || whitener.inverse_scales.iter().any(|value| *value <= 0.0)
    {
        return Err(invalid("whitener factors must be positive"));
    }
    Ok(())
}

pub fn fit_affine_fisher(
    centroids: &[f64],
    node_count: usize,
    columns: usize,
    whitener: &MahalanobisWhitener,
    max_components: usize,
    orient_to: usize,
) -> KernelResult<AffineFisherFit> {
    checked_matrix("node centroids", centroids, node_count, columns)?;
    if node_count < 2 {
        return Err(invalid("an affine Fisher fit needs at least two nodes"));
    }
    if columns != whitener.columns {
        return Err(invalid("centroid width does not match the whitener"));
    }
    if max_components == 0 {
        return Err(invalid("max_components must be positive"));
    }
    if orient_to >= node_count {
        return Err(invalid("orient_to is outside the node roster"));
    }
    validate_whitener(whitener)?;
    let centroid_mean = mean_rows(centroids, node_count, columns);
    let centered = center_rows(centroids, columns, &centroid_mean);
    let inverse_rows = apply_mahalanobis_inverse_centered(whitener, &centered, node_count)?;
    let neutral_offset: Vec<f64> = whitener
        .mean
        .iter()
        .zip(&centroid_mean)
        .map(|(neutral, centroid)| neutral - centroid)
        .collect();
    let neutral_cross_gram = inverse_rows
        .chunks_exact(columns)
        .map(|row| {
            row.iter()
                .zip(&neutral_offset)
                .map(|(left, right)| left * right)
                .sum()
        })
        .collect::<Vec<f64>>();
    let mut whitened_gram = vec![0.0; checked_output("whitened Gram", &[node_count, node_count])?];
    for left in 0..node_count {
        for right in left..node_count {
            let value = (0..columns)
                .map(|column| {
                    centered[left * columns + column] * inverse_rows[right * columns + column]
                })
                .sum::<f64>();
            whitened_gram[left * node_count + right] = value;
            whitened_gram[right * node_count + left] = value;
        }
    }
    let (eigenvalues, eigenvectors) = symmetric_eigen(&whitened_gram, node_count)?;
    let leading = eigenvalues.first().copied().unwrap_or(0.0).max(0.0);
    if leading <= f64::EPSILON {
        return Err(invalid("node centroids have zero whitened spread"));
    }
    let stable_rank = eigenvalues
        .iter()
        .filter(|value| value.max(0.0) > 1e-6 * leading.max(1e-12))
        .count();
    let components = max_components.min(node_count - 1).min(stable_rank).max(1);
    let mut directions = vec![0.0; checked_output("Fisher directions", &[components, columns])?];
    for component in 0..components {
        for column in 0..columns {
            directions[component * columns + column] = (0..node_count)
                .map(|row| {
                    eigenvectors[row * node_count + component]
                        * inverse_rows[row * columns + column]
                })
                .sum::<f64>();
        }
    }
    let mut basis = orthonormalize_rows(&directions, components, columns)?;
    for component in 0..components {
        let projection = (0..columns)
            .map(|column| {
                basis[component * columns + column]
                    * (centroids[orient_to * columns + column] - centroid_mean[column])
            })
            .sum::<f64>();
        if projection < 0.0 {
            for column in 0..columns {
                basis[component * columns + column] = -basis[component * columns + column];
            }
        }
    }
    let mut mean = vec![0.0; columns];
    for component in 0..components {
        let coordinate = (0..columns)
            .map(|column| whitener.mean[column] * basis[component * columns + column])
            .sum::<f64>();
        for column in 0..columns {
            mean[column] += coordinate * basis[component * columns + column];
        }
    }
    let mut node_coordinates = vec![0.0; node_count * components];
    let mut mu_coordinates = vec![0.0; node_count * components];
    for node in 0..node_count {
        for component in 0..components {
            node_coordinates[node * components + component] = (0..columns)
                .map(|column| {
                    (centroids[node * columns + column] - whitener.mean[column])
                        * basis[component * columns + column]
                })
                .sum::<f64>();
            mu_coordinates[node * components + component] = (0..columns)
                .map(|column| {
                    centered[node * columns + column] * basis[component * columns + column]
                })
                .sum::<f64>();
        }
    }
    let inverse_basis = apply_mahalanobis_inverse_centered(whitener, &basis, components)?;
    let mut reduced_metric = vec![0.0; components * components];
    for left in 0..components {
        for right in 0..components {
            reduced_metric[left * components + right] = (0..columns)
                .map(|column| {
                    basis[left * columns + column] * inverse_basis[right * columns + column]
                })
                .sum::<f64>();
        }
    }
    let share_squared = (0..node_count)
        .map(|node| {
            (0..components)
                .map(|left| {
                    (0..components)
                        .map(|right| {
                            mu_coordinates[node * components + left]
                                * reduced_metric[left * components + right]
                                * mu_coordinates[node * components + right]
                        })
                        .sum::<f64>()
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        .max(0.0);
    let total = eigenvalues.iter().map(|value| value.max(0.0)).sum::<f64>();
    let retained = eigenvalues
        .iter()
        .take(components)
        .map(|value| value.max(0.0))
        .sum::<f64>();
    let explained_variance = if total > 1e-12 { retained / total } else { 1.0 };
    checked_finite("affine Fisher basis", &basis)?;
    checked_finite("affine Fisher coordinates", &node_coordinates)?;
    Ok(AffineFisherFit {
        node_count,
        columns,
        components,
        centroid_mean,
        mean,
        basis,
        node_coordinates,
        mu_coordinates,
        whitened_gram,
        neutral_cross_gram,
        explained_variance,
        mahalanobis_share: share_squared.sqrt(),
    })
}

pub fn fit_pca(
    values: &[f64],
    rows: usize,
    columns: usize,
    max_components: usize,
    variance_threshold: f64,
) -> KernelResult<Pca> {
    if max_components == 0 {
        return Err(invalid("max_components must be positive"));
    }
    if !(variance_threshold.is_finite() && 0.0 < variance_threshold && variance_threshold <= 1.0) {
        return Err(invalid(
            "variance_threshold must be finite and in the interval (0, 1]",
        ));
    }
    let centering = fit_neutral_centering(values, rows, columns)?;
    let (all_eigenvalues, all_basis) =
        thin_principal_components(&centering.centered, rows, columns)?;
    let total = all_eigenvalues.iter().sum::<f64>();
    let cap = max_components.min(all_eigenvalues.len());
    let all_explained: Vec<f64> = all_eigenvalues
        .iter()
        .map(|eigenvalue| eigenvalue / total)
        .collect();
    let mut cumulative = 0.0;
    let mut components = cap;
    for (index, fraction) in all_explained.iter().take(cap).enumerate() {
        cumulative += fraction;
        if cumulative + 1e-15 >= variance_threshold {
            components = index + 1;
            break;
        }
    }
    let basis = all_basis[..components * columns].to_vec();
    let eigenvalues = all_eigenvalues[..components].to_vec();
    let explained_variance = all_explained[..components].to_vec();
    let mut cumulative_variance = Vec::with_capacity(components);
    cumulative = 0.0;
    for fraction in &explained_variance {
        cumulative += fraction;
        cumulative_variance.push(cumulative);
    }
    let score_elements = checked_output("PCA scores", &[rows, components])?;
    checked_work(
        "PCA result",
        &[centering.centered.len(), basis.len(), score_elements],
    )?;
    let mut scores = vec![0.0; score_elements];
    for row in 0..rows {
        for component in 0..components {
            scores[row * components + component] = (0..columns)
                .map(|column| {
                    centering.centered[row * columns + column] * basis[component * columns + column]
                })
                .sum();
        }
    }
    checked_finite("PCA basis", &basis)?;
    checked_finite("PCA eigenvalues", &eigenvalues)?;
    checked_finite("PCA explained variance", &explained_variance)?;
    checked_finite("PCA cumulative variance", &cumulative_variance)?;
    checked_finite("PCA scores", &scores)?;
    Ok(Pca {
        rows,
        columns,
        components,
        mean: centering.mean,
        basis,
        eigenvalues,
        explained_variance,
        cumulative_variance,
        scores,
    })
}

pub fn position(
    mean: &[f64],
    basis: &[f64],
    components: usize,
    coordinates: &[f64],
) -> KernelResult<Vec<f64>> {
    let columns = mean.len();
    checked_vector("mean", mean, columns)?;
    validate_orthonormal(basis, components, columns)?;
    checked_vector("coordinates", coordinates, components)?;
    let mut output = mean.to_vec();
    for component in 0..components {
        for column in 0..columns {
            output[column] += coordinates[component] * basis[component * columns + column];
        }
    }
    checked_finite("position", &output)?;
    Ok(output)
}

pub fn project_rows(
    values: &[f64],
    rows: usize,
    mean: &[f64],
    basis: &[f64],
    components: usize,
) -> KernelResult<Vec<f64>> {
    let columns = mean.len();
    checked_matrix("values", values, rows, columns)?;
    checked_vector("mean", mean, columns)?;
    validate_orthonormal(basis, components, columns)?;
    checked_work("row projection", &[values.len(), values.len(), basis.len()])?;
    let mut output = vec![0.0; values.len()];
    for row in 0..rows {
        output[row * columns..(row + 1) * columns].copy_from_slice(mean);
        for component in 0..components {
            let coordinate = (0..columns)
                .map(|column| {
                    (values[row * columns + column] - mean[column])
                        * basis[component * columns + column]
                })
                .sum::<f64>();
            for column in 0..columns {
                output[row * columns + column] += coordinate * basis[component * columns + column];
            }
        }
    }
    checked_finite("projected values", &output)?;
    Ok(output)
}

pub fn ablate_rows(
    values: &[f64],
    rows: usize,
    mean: &[f64],
    basis: &[f64],
    components: usize,
    coefficients: &[f64],
) -> KernelResult<Vec<f64>> {
    let columns = mean.len();
    checked_matrix("values", values, rows, columns)?;
    checked_vector("mean", mean, columns)?;
    validate_orthonormal(basis, components, columns)?;
    checked_vector("ablation coefficients", coefficients, components)?;
    checked_work("row ablation", &[values.len(), values.len(), basis.len()])?;
    let mut output = values.to_vec();
    for row in 0..rows {
        for component in 0..components {
            let coordinate = (0..columns)
                .map(|column| {
                    (values[row * columns + column] - mean[column])
                        * basis[component * columns + column]
                })
                .sum::<f64>();
            for column in 0..columns {
                output[row * columns + column] -=
                    coefficients[component] * coordinate * basis[component * columns + column];
            }
        }
    }
    checked_finite("ablated values", &output)?;
    Ok(output)
}

pub fn pearson_correlations(values: &[f64], rows: usize, columns: usize) -> KernelResult<Vec<f64>> {
    if rows < 2 {
        return Err(invalid("correlations require at least two rows"));
    }
    checked_matrix("values", values, rows, columns)?;
    let output_elements = checked_output("correlation matrix", &[columns, columns])?;
    checked_work(
        "Pearson correlations",
        &[values.len(), values.len(), output_elements],
    )?;
    let mean = mean_rows(values, rows, columns);
    let centered = center_rows(values, columns, &mean);
    let mut norms = vec![0.0_f64; columns];
    for column in 0..columns {
        norms[column] = (0..rows)
            .map(|row| centered[row * columns + column].powi(2))
            .sum::<f64>()
            .sqrt();
        if norms[column] <= f64::EPSILON {
            return Err(invalid(format!(
                "column {column} has zero variance and no Pearson correlation"
            )));
        }
    }
    let mut output = vec![0.0; output_elements];
    for left in 0..columns {
        for right in left..columns {
            let numerator = (0..rows)
                .map(|row| centered[row * columns + left] * centered[row * columns + right])
                .sum::<f64>();
            let value = (numerator / (norms[left] * norms[right])).clamp(-1.0, 1.0);
            output[left * columns + right] = value;
            output[right * columns + left] = value;
        }
    }
    checked_finite("correlation matrix", &output)?;
    Ok(output)
}

pub fn pairwise_distances(values: &[f64], rows: usize, columns: usize) -> KernelResult<Vec<f64>> {
    checked_matrix("values", values, rows, columns)?;
    let output_elements = checked_output("pairwise distances", &[rows, rows])?;
    checked_work("pairwise distances", &[values.len(), output_elements])?;
    let mut output = vec![0.0; output_elements];
    for left in 0..rows {
        for right in (left + 1)..rows {
            let squared = (0..columns)
                .map(|column| {
                    let difference =
                        values[left * columns + column] - values[right * columns + column];
                    difference * difference
                })
                .sum::<f64>();
            let distance = squared.sqrt();
            output[left * rows + right] = distance;
            output[right * rows + left] = distance;
        }
    }
    checked_finite("pairwise distances", &output)?;
    Ok(output)
}

pub fn build_knn_graph(
    distances: &[f64],
    node_count: usize,
    k_nn: usize,
) -> KernelResult<KnnGraph> {
    if node_count < 2 {
        return Err(invalid("k-NN graph requires at least two nodes"));
    }
    if k_nn == 0 {
        return Err(invalid("k_nn must be positive"));
    }
    checked_matrix(
        "pairwise distance matrix",
        distances,
        node_count,
        node_count,
    )?;
    checked_work(
        "k-NN graph",
        &[distances.len(), distances.len(), distances.len()],
    )?;
    for row in 0..node_count {
        let diagonal = distances[row * node_count + row];
        if diagonal.abs() > GRAPH_SYMMETRY_TOLERANCE {
            return Err(invalid(format!(
                "pairwise distance matrix diagonal {row} is not zero"
            )));
        }
        for column in (row + 1)..node_count {
            let left = distances[row * node_count + column];
            let right = distances[column * node_count + row];
            if left < 0.0 || right < 0.0 {
                return Err(invalid("pairwise distances must be non-negative"));
            }
            let scale = left.abs().max(right.abs()).max(1.0);
            if (left - right).abs() > GRAPH_SYMMETRY_TOLERANCE * scale {
                return Err(invalid("pairwise distance matrix must be symmetric"));
            }
        }
    }

    let k_nn = k_nn.min(node_count - 1);
    let mut directed = vec![false; node_count * node_count];
    for row in 0..node_count {
        let mut neighbors: Vec<usize> = (0..node_count).filter(|column| *column != row).collect();
        neighbors.sort_by(|left, right| {
            distances[row * node_count + *left]
                .total_cmp(&distances[row * node_count + *right])
                .then_with(|| left.cmp(right))
        });
        for column in neighbors.into_iter().take(k_nn) {
            directed[row * node_count + column] = true;
        }
    }

    let mut mask = vec![0_u8; node_count * node_count];
    let mut neighbor_distances = Vec::new();
    for row in 0..node_count {
        for column in 0..node_count {
            if row != column
                && (directed[row * node_count + column] || directed[column * node_count + row])
            {
                mask[row * node_count + column] = 1;
                neighbor_distances.push(distances[row * node_count + column]);
            }
        }
    }
    checked_output("k-NN neighbor distances", &[neighbor_distances.len()])?;
    let component_count = connected_components(&mask, node_count);
    Ok(KnnGraph {
        node_count,
        k_nn,
        mask,
        neighbor_distances,
        component_count,
    })
}

pub fn build_normalized_laplacian(
    gram: &[f64],
    node_count: usize,
    k_nn: Option<usize>,
    bandwidth: Option<f64>,
) -> KernelResult<NormalizedLaplacian> {
    if node_count < 4 {
        return Err(invalid(format!(
            "spectral embedding needs at least four centroids, got {node_count}"
        )));
    }
    checked_matrix("spectral Gram", gram, node_count, node_count)?;
    let matrix_elements = checked_output("spectral graph matrix", &[node_count, node_count])?;
    checked_work(
        "normalized-Laplacian construction",
        &[
            gram.len(),
            matrix_elements,
            matrix_elements,
            matrix_elements,
            matrix_elements,
            node_count,
        ],
    )?;
    let gram = coerce_fp32("fp32 spectral Gram", gram)?;

    let mut symmetric_gram = vec![0.0; matrix_elements];
    for row in 0..node_count {
        for column in 0..node_count {
            symmetric_gram[row * node_count + column] =
                fp32(0.5 * (gram[row * node_count + column] + gram[column * node_count + row]));
        }
    }
    let mut distances = vec![0.0; matrix_elements];
    for row in 0..node_count {
        for column in (row + 1)..node_count {
            let diagonal_sum = fp32(
                symmetric_gram[row * node_count + row]
                    + symmetric_gram[column * node_count + column],
            );
            let doubled_cross = fp32(2.0 * symmetric_gram[row * node_count + column]);
            let squared = fp32(diagonal_sum - doubled_cross).max(0.0);
            let distance = fp32(squared.sqrt());
            distances[row * node_count + column] = distance;
            distances[column * node_count + row] = distance;
        }
    }
    checked_finite("spectral pairwise distances", &distances)?;

    let default_k_nn = 5_usize.max((node_count as f64).ln().ceil() as usize);
    let graph = build_knn_graph(
        &distances,
        node_count,
        k_nn.unwrap_or(default_k_nn).clamp(1, node_count - 1),
    )?;
    if graph.component_count > 1 {
        return Err(invalid(format!(
            "spectral embedding: k-NN graph has {} connected components (need 1). Raise k_nn or switch to PCA.",
            graph.component_count
        )));
    }

    let bandwidth = match bandwidth {
        Some(value) if value.is_finite() && value > 0.0 => value,
        Some(_) => return Err(invalid("bandwidth must be finite and positive")),
        None => {
            if graph.neighbor_distances.is_empty() {
                return Err(invalid("spectral embedding: k-NN graph has no edges"));
            }
            let mut sorted = graph.neighbor_distances.clone();
            sorted.sort_by(f64::total_cmp);
            let median = sorted[(sorted.len() - 1) / 2];
            if median <= 0.0 {
                1e-6
            } else {
                median
            }
        }
    };
    let bandwidth = f64::from(bandwidth as f32);
    let denominator = fp32(2.0 * bandwidth * bandwidth);
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(invalid("heat-kernel bandwidth scale is not finite"));
    }

    let mut weights = vec![0.0; matrix_elements];
    let mut degrees = vec![0.0; node_count];
    for row in 0..node_count {
        for column in 0..node_count {
            if graph.mask[row * node_count + column] != 0 {
                let distance = distances[row * node_count + column];
                let squared = fp32(distance * distance);
                let weight = fp32(fp32(-squared / denominator).exp());
                weights[row * node_count + column] = weight;
                degrees[row] = fp32(degrees[row] + weight);
            }
        }
    }
    if degrees.iter().any(|degree| *degree <= 0.0) {
        return Err(invalid(
            "spectral embedding: graph contains an isolated node",
        ));
    }

    let inverse_sqrt_degrees: Vec<f64> = degrees
        .iter()
        .map(|degree| fp32(fp32(degree.max(1e-12).sqrt()).recip()))
        .collect();
    let mut laplacian = vec![0.0; matrix_elements];
    for row in 0..node_count {
        laplacian[row * node_count + row] = 1.0;
        for column in (row + 1)..node_count {
            let value = fp32(
                fp32(-weights[row * node_count + column] * inverse_sqrt_degrees[row])
                    * inverse_sqrt_degrees[column],
            );
            laplacian[row * node_count + column] = value;
            laplacian[column * node_count + row] = value;
        }
    }
    checked_finite("spectral adjacency weights", &weights)?;
    checked_finite("spectral degrees", &degrees)?;
    checked_finite("normalized Laplacian", &laplacian)?;

    let (descending_values, descending_vectors) = symmetric_eigen(&laplacian, node_count)?;
    let mut nontrivial_eigenvalues = Vec::with_capacity(node_count - 1);
    let mut nontrivial_eigenvectors = vec![0.0; node_count * (node_count - 1)];
    for output_column in 0..(node_count - 1) {
        let input_column = node_count - 2 - output_column;
        nontrivial_eigenvalues.push(fp32(descending_values[input_column]));
        for row in 0..node_count {
            nontrivial_eigenvectors[row * (node_count - 1) + output_column] =
                fp32(descending_vectors[row * node_count + input_column]);
        }
    }
    checked_finite("nontrivial Laplacian eigenvalues", &nontrivial_eigenvalues)?;
    checked_finite(
        "nontrivial Laplacian eigenvectors",
        &nontrivial_eigenvectors,
    )?;
    Ok(NormalizedLaplacian {
        node_count,
        k_nn: graph.k_nn,
        bandwidth,
        mask: graph.mask,
        weights,
        degrees,
        laplacian,
        nontrivial_eigenvalues,
        nontrivial_eigenvectors,
    })
}

pub fn derive_spectral_embedding(
    gram: &[f64],
    node_count: usize,
    max_dimensions: usize,
    min_dimensions: Option<usize>,
    k_nn: Option<usize>,
    bandwidth: Option<f64>,
) -> KernelResult<SpectralEmbedding> {
    if max_dimensions == 0 {
        return Err(invalid("max_dimensions must be positive"));
    }
    if matches!(min_dimensions, Some(0)) {
        return Err(invalid("min_dimensions must be positive when provided"));
    }
    let eigensystem = build_normalized_laplacian(gram, node_count, k_nn, bandwidth)?;
    spectral_embedding_from_eigensystem(&eigensystem, max_dimensions, min_dimensions)
}

fn spectral_embedding_from_eigensystem(
    eigensystem: &NormalizedLaplacian,
    max_dimensions: usize,
    min_dimensions: Option<usize>,
) -> KernelResult<SpectralEmbedding> {
    if max_dimensions == 0 {
        return Err(invalid("max_dimensions must be positive"));
    }
    if matches!(min_dimensions, Some(0)) {
        return Err(invalid("min_dimensions must be positive when provided"));
    }
    let node_count = eigensystem.node_count;
    let cap = max_dimensions
        .min(eigensystem.nontrivial_eigenvalues.len().saturating_sub(1))
        .min(node_count - 2)
        .max(1);
    let heuristic_dimensions = if cap == 1 {
        1
    } else {
        let mut best_index = 0;
        let mut best_ratio = f64::NEG_INFINITY;
        for index in 0..cap {
            let denominator = eigensystem.nontrivial_eigenvalues[index].max(1e-12);
            let ratio = eigensystem.nontrivial_eigenvalues[index + 1] / denominator;
            if ratio > best_ratio {
                best_ratio = ratio;
                best_index = index;
            }
        }
        best_index + 1
    };
    let dimensions = min_dimensions.map_or(heuristic_dimensions, |floor| {
        heuristic_dimensions.max(floor.min(cap))
    });
    let pinned = dimensions != heuristic_dimensions;
    let gap_magnitude = if dimensions < eigensystem.nontrivial_eigenvalues.len() {
        fp32(
            eigensystem.nontrivial_eigenvalues[dimensions]
                - eigensystem.nontrivial_eigenvalues[dimensions - 1],
        )
    } else {
        0.0
    };
    let diagnostic_count = max_dimensions
        .saturating_add(5)
        .min(eigensystem.nontrivial_eigenvalues.len());
    let eigenvalues = eigensystem.nontrivial_eigenvalues[..diagnostic_count].to_vec();
    let coordinate_elements = checked_output("spectral coordinates", &[node_count, dimensions])?;
    checked_work(
        "spectral embedding",
        &[
            eigensystem.nontrivial_eigenvectors.len(),
            coordinate_elements,
            eigenvalues.len(),
        ],
    )?;
    let full_columns = node_count - 1;
    let mut coordinates = vec![0.0; coordinate_elements];
    for row in 0..node_count {
        for dimension in 0..dimensions {
            coordinates[row * dimensions + dimension] =
                eigensystem.nontrivial_eigenvectors[row * full_columns + dimension];
        }
    }
    checked_finite("spectral coordinates", &coordinates)?;
    Ok(SpectralEmbedding {
        node_count,
        dimensions,
        heuristic_dimensions,
        min_dimensions,
        pinned,
        k_nn: eigensystem.k_nn,
        bandwidth: eigensystem.bandwidth,
        gap_magnitude,
        eigenvalues,
        coordinates,
    })
}

pub fn derive_pca_layout(
    gram: &[f64],
    node_count: usize,
    max_dimensions: usize,
    variance_threshold: f64,
) -> KernelResult<PcaLayout> {
    if node_count < 2 {
        return Err(invalid(format!(
            "PCA coordinate derivation needs at least two centroids, got {node_count}"
        )));
    }
    if max_dimensions == 0 {
        return Err(invalid("max_dimensions must be positive"));
    }
    if !(variance_threshold.is_finite() && variance_threshold > 0.0 && variance_threshold <= 1.0) {
        return Err(invalid("variance_threshold must be finite and in (0, 1]"));
    }
    checked_matrix("PCA layout Gram", gram, node_count, node_count)?;
    let matrix_elements = checked_output("PCA layout matrix", &[node_count, node_count])?;
    checked_work(
        "PCA layout",
        &[
            gram.len(),
            matrix_elements,
            matrix_elements,
            matrix_elements,
        ],
    )?;
    let gram = coerce_fp32("fp32 PCA layout Gram", gram)?;
    let mut symmetric = vec![0.0; matrix_elements];
    for row in 0..node_count {
        for column in row..node_count {
            let value =
                fp32(0.5 * (gram[row * node_count + column] + gram[column * node_count + row]));
            symmetric[row * node_count + column] = value;
            symmetric[column * node_count + row] = value;
        }
    }
    let (raw_eigenvalues, raw_eigenvectors) = symmetric_eigen(&symmetric, node_count)?;
    let eigenvalues: Vec<f64> = raw_eigenvalues
        .iter()
        .map(|value| fp32(value.max(0.0)))
        .collect();
    let total = eigenvalues.iter().sum::<f64>().max(1e-12);
    let cap = max_dimensions.min(node_count);
    let mut explained_variance = Vec::with_capacity(cap);
    let mut cumulative_variance = Vec::with_capacity(cap);
    let mut cumulative = 0.0;
    for eigenvalue in eigenvalues.iter().take(cap) {
        let fraction = fp32(*eigenvalue / total);
        cumulative = fp32(cumulative + fraction);
        explained_variance.push(fraction);
        cumulative_variance.push(cumulative);
    }
    let dimensions = cumulative_variance
        .iter()
        .position(|value| *value >= variance_threshold)
        .map_or(cap, |index| index + 1)
        .clamp(1, cap);
    let coordinate_elements = checked_output("PCA layout coordinates", &[node_count, dimensions])?;
    let mut coordinates = vec![0.0; coordinate_elements];
    for row in 0..node_count {
        for dimension in 0..dimensions {
            coordinates[row * dimensions + dimension] = fp32(
                raw_eigenvectors[row * node_count + dimension] * eigenvalues[dimension].sqrt(),
            );
        }
    }
    checked_finite("PCA layout coordinates", &coordinates)?;
    Ok(PcaLayout {
        node_count,
        dimensions,
        explained_variance,
        cumulative_variance,
        coordinates,
    })
}

fn pairwise_distances_from_gram(gram: &[f64], node_count: usize) -> KernelResult<Vec<f64>> {
    checked_matrix("topology Gram", gram, node_count, node_count)?;
    let matrix_elements = checked_output("topology distance matrix", &[node_count, node_count])?;
    checked_work(
        "topology distance construction",
        &[gram.len(), matrix_elements, matrix_elements],
    )?;
    let gram = coerce_fp32("fp32 topology Gram", gram)?;
    let mut symmetric = vec![0.0; matrix_elements];
    for row in 0..node_count {
        for column in row..node_count {
            let value =
                fp32(0.5 * (gram[row * node_count + column] + gram[column * node_count + row]));
            symmetric[row * node_count + column] = value;
            symmetric[column * node_count + row] = value;
        }
    }
    let mut distances = vec![0.0; matrix_elements];
    for row in 0..node_count {
        for column in (row + 1)..node_count {
            let diagonal_sum =
                fp32(symmetric[row * node_count + row] + symmetric[column * node_count + column]);
            let doubled_cross = fp32(2.0 * symmetric[row * node_count + column]);
            let distance = fp32(fp32(diagonal_sum - doubled_cross).max(0.0).sqrt());
            distances[row * node_count + column] = distance;
            distances[column * node_count + row] = distance;
        }
    }
    checked_finite("topology distances", &distances)?;
    Ok(distances)
}

fn validate_distance_matrix(distances: &[f64], node_count: usize) -> KernelResult<()> {
    checked_matrix("topology distances", distances, node_count, node_count)?;
    for row in 0..node_count {
        if distances[row * node_count + row].abs() > GRAPH_SYMMETRY_TOLERANCE {
            return Err(invalid("topology distance diagonal must be zero"));
        }
        for column in (row + 1)..node_count {
            let left = distances[row * node_count + column];
            let right = distances[column * node_count + row];
            if left < 0.0 || right < 0.0 {
                return Err(invalid("topology distances must be non-negative"));
            }
            let scale = left.abs().max(right.abs()).max(1.0);
            if (left - right).abs() > GRAPH_SYMMETRY_TOLERANCE * scale {
                return Err(invalid("topology distance matrix must be symmetric"));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct PersistenceEdge {
    left: usize,
    right: usize,
    length: f64,
}

#[derive(Debug, Clone)]
struct PersistenceTriangle {
    filtration: f64,
    left: usize,
    middle: usize,
    right: usize,
}

pub fn count_persistent_loops(
    distances: &[f64],
    node_count: usize,
    persistence_fraction: f64,
    max_dimensions: usize,
) -> KernelResult<usize> {
    validate_distance_matrix(distances, node_count)?;
    if !persistence_fraction.is_finite() || persistence_fraction < 0.0 {
        return Err(invalid(
            "persistence_fraction must be finite and non-negative",
        ));
    }
    if max_dimensions == 0 {
        return Err(invalid("max_dimensions must be positive"));
    }
    if node_count < 4 {
        return Ok(0);
    }
    if node_count > PERSISTENCE_MAX_NODES {
        return Err(invalid(format!(
            "persistent topology supports at most {PERSISTENCE_MAX_NODES} nodes, got {node_count}"
        )));
    }
    let mut complete_edges = Vec::with_capacity(node_count * (node_count - 1) / 2);
    for left in 0..node_count {
        for right in (left + 1)..node_count {
            complete_edges.push(PersistenceEdge {
                left,
                right,
                length: distances[left * node_count + right],
            });
        }
    }
    complete_edges.sort_by(|left, right| {
        left.length
            .total_cmp(&right.length)
            .then_with(|| left.left.cmp(&right.left))
            .then_with(|| left.right.cmp(&right.right))
    });
    let mut parents: Vec<usize> = (0..node_count).collect();
    let mut connectivity_scale = 0.0;
    let mut joined = 0;
    for edge in &complete_edges {
        if union_components(&mut parents, edge.left, edge.right) {
            connectivity_scale = edge.length;
            joined += 1;
            if joined == node_count - 1 {
                break;
            }
        }
    }
    if connectivity_scale <= 0.0 {
        return Ok(0);
    }
    let epsilon_max = 2.0 * connectivity_scale;
    if complete_edges.iter().all(|edge| edge.length <= epsilon_max) {
        return Ok(0);
    }
    let pairs = rips_h1_persistence(
        distances,
        node_count,
        epsilon_max,
        PERSISTENCE_MAX_TRIANGLES,
    )?;
    let threshold = persistence_fraction * connectivity_scale;
    Ok(pairs
        .iter()
        .filter(|(birth, death)| {
            *death >= (1.0 + 0.5 * persistence_fraction) * connectivity_scale
                && death.min(epsilon_max) - *birth >= threshold.max(persistence_fraction * birth)
        })
        .count()
        .min(max_dimensions))
}

fn rips_h1_persistence(
    distances: &[f64],
    node_count: usize,
    epsilon_max: f64,
    max_triangles: usize,
) -> KernelResult<Vec<(f64, f64)>> {
    let mut edges = Vec::new();
    for left in 0..node_count {
        for right in (left + 1)..node_count {
            let length = distances[left * node_count + right];
            if length <= epsilon_max {
                edges.push(PersistenceEdge {
                    left,
                    right,
                    length,
                });
            }
        }
    }
    edges.sort_by(|left, right| {
        left.length
            .total_cmp(&right.length)
            .then_with(|| left.left.cmp(&right.left))
            .then_with(|| left.right.cmp(&right.right))
    });
    let edge_count = edges.len();
    let mut edge_ids = vec![usize::MAX; node_count * node_count];
    for (index, edge) in edges.iter().enumerate() {
        edge_ids[edge.left * node_count + edge.right] = index;
        edge_ids[edge.right * node_count + edge.left] = index;
    }
    let mut triangles = Vec::new();
    for left in 0..node_count {
        for middle in (left + 1)..node_count {
            if edge_ids[left * node_count + middle] == usize::MAX {
                continue;
            }
            for right in (middle + 1)..node_count {
                if edge_ids[left * node_count + right] == usize::MAX
                    || edge_ids[middle * node_count + right] == usize::MAX
                {
                    continue;
                }
                if triangles.len() == max_triangles {
                    return Err(invalid(
                        "persistent-homology triangle budget exceeded; topology is unresolved",
                    ));
                }
                triangles.push(PersistenceTriangle {
                    filtration: distances[left * node_count + middle]
                        .max(distances[left * node_count + right])
                        .max(distances[middle * node_count + right]),
                    left,
                    middle,
                    right,
                });
            }
        }
    }
    triangles.sort_by(|left, right| {
        left.filtration
            .total_cmp(&right.filtration)
            .then_with(|| left.left.cmp(&right.left))
            .then_with(|| left.middle.cmp(&right.middle))
            .then_with(|| left.right.cmp(&right.right))
    });
    let word_count = edge_count.div_ceil(64);
    let reduced_bitset_elements = edge_count
        .checked_mul(word_count)
        .ok_or_else(|| invalid("persistent-homology bitset work size overflows usize"))?;
    checked_work(
        "persistent-homology reduction",
        &[
            edges.len(),
            edge_ids.len(),
            triangles.len(),
            edge_count,
            edge_count,
            reduced_bitset_elements,
        ],
    )?;

    let mut parents: Vec<usize> = (0..node_count).collect();
    let mut positive = vec![false; edge_count];
    for (index, edge) in edges.iter().enumerate() {
        if !union_components(&mut parents, edge.left, edge.right) {
            positive[index] = true;
        }
    }

    let mut reduced_by_pivot: Vec<Option<Vec<u64>>> = vec![None; edge_count];
    let mut killed = vec![None; edge_count];
    for triangle in triangles {
        let mut column = vec![0_u64; word_count];
        for (left, right) in [
            (triangle.left, triangle.middle),
            (triangle.left, triangle.right),
            (triangle.middle, triangle.right),
        ] {
            let edge = edge_ids[left * node_count + right];
            column[edge / 64] |= 1_u64 << (edge % 64);
        }
        while let Some(pivot) = highest_set_bit(&column) {
            match &reduced_by_pivot[pivot] {
                Some(reduced) => {
                    for (value, pivot_value) in column.iter_mut().zip(reduced) {
                        *value ^= *pivot_value;
                    }
                }
                None => break,
            }
        }
        let Some(pivot) = highest_set_bit(&column) else {
            continue;
        };
        reduced_by_pivot[pivot] = Some(column);
        if positive[pivot] && killed[pivot].is_none() {
            killed[pivot] = Some(triangle.filtration);
        }
    }
    Ok(positive
        .iter()
        .enumerate()
        .filter_map(|(index, is_positive)| {
            is_positive.then_some((edges[index].length, killed[index].unwrap_or(f64::INFINITY)))
        })
        .collect())
}

fn highest_set_bit(column: &[u64]) -> Option<usize> {
    column
        .iter()
        .enumerate()
        .rev()
        .find(|(_, word)| **word != 0)
        .map(|(word_index, word)| word_index * 64 + (63 - word.leading_zeros() as usize))
}

pub fn detect_faint_cycle(distances: &[f64], node_count: usize) -> KernelResult<Option<Vec<f64>>> {
    validate_distance_matrix(distances, node_count)?;
    if !(CYCLE_MIN_NODES..=CYCLE_MAX_NODES).contains(&node_count) {
        return Ok(None);
    }
    let mut nearest_two = vec![[0_usize; 2]; node_count];
    for row in 0..node_count {
        let mut candidates: Vec<usize> = (0..node_count).filter(|node| *node != row).collect();
        candidates.sort_by(|left, right| {
            distances[row * node_count + *left]
                .total_cmp(&distances[row * node_count + *right])
                .then_with(|| left.cmp(right))
        });
        nearest_two[row] = [candidates[0], candidates[1]];
    }
    let mut undirected_edges = vec![false; node_count * node_count];
    let mut degrees = vec![0_usize; node_count];
    for (node, neighbors) in nearest_two.iter().enumerate() {
        for neighbor in neighbors {
            let left = node.min(*neighbor);
            let right = node.max(*neighbor);
            let index = left * node_count + right;
            if !undirected_edges[index] {
                undirected_edges[index] = true;
                degrees[left] += 1;
                degrees[right] += 1;
            }
        }
    }
    let max_degree = degrees.iter().copied().max().unwrap_or(0);
    if max_degree > CYCLE_MAX_DEGREE_CLUSTER {
        return Ok(None);
    }
    let tour = nearest_neighbor_tour(distances, node_count);
    let mut positions = vec![0_usize; node_count];
    for (index, node) in tour.iter().enumerate() {
        positions[*node] = index;
    }
    let tour_edges: Vec<f64> = (0..node_count)
        .map(|index| distances[tour[index] * node_count + tour[(index + 1) % node_count]])
        .collect();
    let mut sorted_edges = tour_edges.clone();
    sorted_edges.sort_by(f64::total_cmp);
    let median = if node_count % 2 == 1 {
        sorted_edges[node_count / 2]
    } else {
        0.5 * (sorted_edges[node_count / 2 - 1] + sorted_edges[node_count / 2])
    };
    if median <= 0.0 {
        return Ok(None);
    }

    let mut separation_one_sum = 0.0;
    let mut separation_one_count = 0_usize;
    let mut separation_two_sum = 0.0;
    let mut separation_two_count = 0_usize;
    for left in 0..node_count {
        for right in (left + 1)..node_count {
            let direct = positions[left].abs_diff(positions[right]);
            let separation = direct.min(node_count - direct);
            if separation == 1 {
                separation_one_sum += distances[left * node_count + right];
                separation_one_count += 1;
            } else if separation == 2 {
                separation_two_sum += distances[left * node_count + right];
                separation_two_count += 1;
            }
        }
    }
    if separation_one_count == 0 || separation_two_count == 0 {
        return Ok(None);
    }
    let contrast = (separation_two_sum / separation_two_count as f64)
        / (separation_one_sum / separation_one_count as f64).max(1e-9);
    if contrast < CYCLE_CONTRAST_MIN {
        return Ok(None);
    }

    let hits = tour
        .iter()
        .enumerate()
        .map(|(index, node)| {
            let before = tour[(index + node_count - 1) % node_count];
            let after = tour[(index + 1) % node_count];
            usize::from(nearest_two[*node].contains(&before))
                + usize::from(nearest_two[*node].contains(&after))
        })
        .sum::<usize>();
    let recall = hits as f64 / (2 * node_count) as f64;
    let closure = sorted_edges[node_count - 1] / median;
    let uniform =
        max_degree <= CYCLE_MAX_DEGREE && closure < CYCLE_CLOSURE_MAX && recall >= CYCLE_RECALL_MIN;

    let small_count = (node_count / 2).max(1);
    let small_scale = sorted_edges[..small_count][small_count / 2];
    let gaps: Vec<f64> = tour_edges
        .iter()
        .copied()
        .filter(|edge| *edge > CYCLE_LARGE_FACTOR * small_scale)
        .collect();
    let half = node_count / 2;
    let near = tour_edges.iter().sum::<f64>() / node_count as f64;
    let far = (0..node_count)
        .map(|index| distances[tour[index] * node_count + tour[(index + half) % node_count]])
        .sum::<f64>()
        / node_count as f64;
    let antipode = far / near.max(1e-9);
    let clustered = if gaps.len() >= CYCLE_GAPS_MIN {
        let minimum = gaps.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = gaps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        minimum >= CYCLE_BIMODAL_MIN * small_scale
            && maximum / minimum <= CYCLE_GAP_REG_MAX
            && antipode >= CYCLE_ANTIPODE_MIN
    } else {
        false
    };
    if !uniform && !clustered {
        return Ok(None);
    }
    let mut angles = vec![0.0; node_count];
    for (index, node) in tour.iter().enumerate() {
        angles[*node] = fp32(2.0 * std::f64::consts::PI * index as f64 / node_count as f64);
    }
    Ok(Some(angles))
}

fn nearest_neighbor_tour(distances: &[f64], node_count: usize) -> Vec<usize> {
    let mut best_tour = Vec::new();
    let mut best_length = f64::INFINITY;
    for start in 0..node_count {
        let mut visited = vec![false; node_count];
        visited[start] = true;
        let mut tour = vec![start];
        while tour.len() < node_count {
            let current = *tour.last().unwrap_or(&start);
            let next = (0..node_count)
                .filter(|candidate| !visited[*candidate])
                .min_by(|left, right| {
                    distances[current * node_count + *left]
                        .total_cmp(&distances[current * node_count + *right])
                        .then_with(|| left.cmp(right))
                })
                .unwrap_or(start);
            visited[next] = true;
            tour.push(next);
        }
        let length = (0..node_count)
            .map(|index| distances[tour[index] * node_count + tour[(index + 1) % node_count]])
            .sum::<f64>();
        if length < best_length {
            best_length = length;
            best_tour = tour;
        }
    }
    let mut improved = true;
    while improved {
        improved = false;
        for left_index in 0..(node_count - 1) {
            for right_index in (left_index + 2)..node_count {
                if left_index == 0 && right_index == node_count - 1 {
                    continue;
                }
                let left = best_tour[left_index];
                let left_next = best_tour[left_index + 1];
                let right = best_tour[right_index];
                let right_next = best_tour[(right_index + 1) % node_count];
                let current = distances[left * node_count + left_next]
                    + distances[right * node_count + right_next];
                let swapped = distances[left * node_count + right]
                    + distances[left_next * node_count + right_next];
                if current > swapped + 1e-9 {
                    best_tour[(left_index + 1)..=right_index].reverse();
                    improved = true;
                }
            }
        }
    }
    best_tour
}

fn is_angular_harmonic(candidate: &[f64], accepted: &[Vec<f64>]) -> bool {
    accepted.iter().any(|angle| {
        (1..=HARMONIC_MAX_ORDER).any(|order| {
            let mut same_real = 0.0;
            let mut same_imag = 0.0;
            let mut opposite_real = 0.0;
            let mut opposite_imag = 0.0;
            for (candidate, accepted) in candidate.iter().zip(angle) {
                let same = candidate - order as f64 * accepted;
                same_real += same.cos();
                same_imag += same.sin();
                let opposite = candidate + order as f64 * accepted;
                opposite_real += opposite.cos();
                opposite_imag += opposite.sin();
            }
            let count = candidate.len() as f64;
            let same = same_real.hypot(same_imag) / count;
            let opposite = opposite_real.hypot(opposite_imag) / count;
            same.max(opposite) >= HARMONIC_COHERENCE
        })
    })
}

pub fn detect_periodic_topology(
    gram: &[f64],
    node_count: usize,
    max_dimensions: usize,
    k_nn: Option<usize>,
    bandwidth: Option<f64>,
    persistence_fraction: f64,
) -> KernelResult<Option<PeriodicTopology>> {
    if max_dimensions == 0 {
        return Err(invalid("max_dimensions must be positive"));
    }
    if node_count > PERSISTENCE_MAX_NODES {
        return Err(invalid(format!(
            "periodic topology supports at most {PERSISTENCE_MAX_NODES} nodes, got {node_count}"
        )));
    }
    let eigensystem = build_normalized_laplacian(gram, node_count, k_nn, bandwidth)?;
    let distances = pairwise_distances_from_gram(gram, node_count)?;
    periodic_topology_from_eigensystem(
        &distances,
        &eigensystem,
        max_dimensions,
        persistence_fraction,
    )
}

fn ascending_eigen(matrix: &[f64], size: usize) -> KernelResult<(Vec<f64>, Vec<f64>)> {
    let (mut values, vectors) = symmetric_eigen(matrix, size)?;
    values.reverse();
    let vectors = vectors
        .chunks_exact(size)
        .flat_map(|row| row.iter().rev().copied())
        .collect();
    Ok((values, vectors))
}

fn untangle_periodic_eigenspaces(
    values: &[f64],
    vectors: &[f64],
    rows: usize,
) -> KernelResult<Vec<f64>> {
    let columns = values.len();
    let mut result = vectors.to_vec();
    let mut start = 0;
    while start < columns {
        let mut end = start + 1;
        while end < columns && values[end] <= 1.05 * values[start] {
            end += 1;
        }
        let width = end - start;
        if (4..=8).contains(&width) && width.is_multiple_of(2) {
            let pairs: Vec<(usize, usize)> = (0..width)
                .flat_map(|i| (i..width).map(move |j| (i, j)))
                .collect();
            let size = pairs.len();
            let mut products = vec![0.0; rows * size];
            for (index, &(i, j)) in pairs.iter().enumerate() {
                let factor = if i == j { 1.0 } else { 2.0_f64.sqrt() };
                let mut mean = 0.0;
                for row in 0..rows {
                    let product = vectors[row * columns + start + i]
                        * vectors[row * columns + start + j]
                        * factor;
                    products[row * size + index] = product;
                    mean += product / rows as f64;
                }
                for row in 0..rows {
                    products[row * size + index] -= mean;
                }
            }
            let mut covariance = vec![0.0; size * size];
            for i in 0..size {
                for j in 0..size {
                    covariance[i * size + j] = (0..rows)
                        .map(|row| products[row * size + i] * products[row * size + j])
                        .sum();
                }
            }
            let (variances, forms) = ascending_eigen(&covariance, size)?;
            for column in 0..size {
                if variances[column] > 1e-4 * variances[size - 1] {
                    break;
                }
                let mut quadratic = vec![0.0; width * width];
                for (index, &(i, j)) in pairs.iter().enumerate() {
                    let value =
                        forms[index * size + column] / if i == j { 1.0 } else { 2.0_f64.sqrt() };
                    quadratic[i * width + j] = value;
                    quadratic[j * width + i] = value;
                }
                let (spectrum, rotation) = ascending_eigen(&quadratic, width)?;
                let spread = spectrum[width - 1] - spectrum[0];
                let within = (0..width)
                    .step_by(2)
                    .map(|i| spectrum[i + 1] - spectrum[i])
                    .fold(0.0, f64::max);
                let between = (1..width - 1)
                    .step_by(2)
                    .map(|i| spectrum[i + 1] - spectrum[i])
                    .fold(f64::INFINITY, f64::min);
                if spread >= 1e-5 && within < 0.1 * spread && between > 0.1 * spread {
                    for row in 0..rows {
                        for j in 0..width {
                            result[row * columns + start + j] = (0..width)
                                .map(|i| {
                                    vectors[row * columns + start + i] * rotation[i * width + j]
                                })
                                .sum();
                        }
                    }
                    break;
                }
            }
        }
        start = end;
    }
    Ok(result)
}

fn distance_eigensystem(distances: &[f64], rows: usize) -> KernelResult<(Vec<f64>, Vec<f64>)> {
    let means: Vec<f64> = distances
        .chunks_exact(rows)
        .map(|row| row.iter().map(|value| value * value).sum::<f64>() / rows as f64)
        .collect();
    let mean = means.iter().sum::<f64>() / rows as f64;
    let mut gram = vec![0.0; rows * rows];
    for i in 0..rows {
        for j in 0..rows {
            gram[i * rows + j] =
                -0.5 * (distances[i * rows + j].powi(2) - means[i] - means[j] + mean);
        }
    }
    ascending_eigen(&gram, rows)
}

fn periodic_mds_angles(
    distances: &[f64],
    rows: usize,
    dimensions: usize,
    values: &[f64],
    vectors: &[f64],
) -> KernelResult<Option<Vec<f64>>> {
    let columns = 2 * dimensions;
    if columns >= rows {
        return Ok(None);
    }
    if values[rows - columns] <= 1e-7 * values[rows - 1] {
        return Ok(None);
    }
    let block: Vec<f64> = vectors
        .chunks_exact(rows)
        .flat_map(|row| row[rows - columns..].iter().copied())
        .collect();
    let block = untangle_periodic_eigenspaces(&vec![1.0; columns], &block, rows)?;
    let mut angles = vec![0.0; rows * dimensions];
    for row in 0..rows {
        for axis in 0..dimensions {
            angles[row * dimensions + axis] =
                fp32(block[row * columns + 2 * axis + 1].atan2(block[row * columns + 2 * axis]));
        }
    }
    Ok(periodic_neighborhoods_preserved(distances, &angles, rows, dimensions).then_some(angles))
}

fn periodic_neighborhoods_preserved(
    distances: &[f64],
    angles: &[f64],
    node_count: usize,
    dimensions: usize,
) -> bool {
    if node_count < 7 || angles.iter().any(|value| !value.is_finite()) {
        return false;
    }
    let period = 2.0 * std::f64::consts::PI;
    for axis in 0..dimensions {
        let mut ordered: Vec<f64> = (0..node_count)
            .map(|row| angles[row * dimensions + axis].rem_euclid(period))
            .collect();
        ordered.sort_by(f64::total_cmp);
        ordered.push(ordered[0] + period);
        if ordered
            .windows(2)
            .any(|pair| pair[1] - pair[0] >= std::f64::consts::PI)
        {
            return false;
        }
    }
    let diameter = distances.iter().copied().fold(0.0, f64::max);
    let neighbors = (2 * dimensions).min(node_count - 1);
    let mut bad_rows = 0;
    for row in 0..node_count {
        let mut source = Vec::with_capacity(node_count - 1);
        let mut mapped = Vec::with_capacity(node_count - 1);
        for other in 0..node_count {
            if row == other {
                continue;
            }
            let distance = distances[row * node_count + other];
            if distance <= 1e-7 * diameter {
                return false;
            }
            source.push(distance);
            let squared = (0..dimensions)
                .map(|axis| {
                    let a = angles[row * dimensions + axis];
                    let b = angles[other * dimensions + axis];
                    (a.cos() - b.cos()).powi(2) + (a.sin() - b.sin()).powi(2)
                })
                .sum::<f64>();
            mapped.push(squared.sqrt());
        }
        let mut source_sorted = source.clone();
        let mut mapped_sorted = mapped.clone();
        source_sorted.sort_by(f64::total_cmp);
        mapped_sorted.sort_by(f64::total_cmp);
        let radius = source_sorted[neighbors - 1];
        let mapped_radius = mapped_sorted[neighbors - 1];
        let collapsed_radius =
            0.2 * std::f64::consts::PI / (node_count as f64).powf(1.0 / dimensions as f64);
        // Include ties so row order cannot conceal a collapsed chart.
        if source.iter().zip(&mapped).any(|(original, projected)| {
            (*projected <= mapped_radius + 1e-6 && *original > 2.0 * radius)
                || (*projected < collapsed_radius && *original > radius)
        }) {
            bad_rows += 1;
        }
    }
    bad_rows as f64 / node_count as f64 <= 0.1
}

fn periodic_topology_from_eigensystem(
    distances: &[f64],
    eigensystem: &NormalizedLaplacian,
    max_dimensions: usize,
    persistence_fraction: f64,
) -> KernelResult<Option<PeriodicTopology>> {
    let node_count = eigensystem.node_count;
    let persistent_loops = count_persistent_loops(
        distances,
        node_count,
        persistence_fraction,
        max_dimensions.saturating_add(1),
    )?;
    if persistent_loops > max_dimensions {
        return Ok(None);
    }
    if persistent_loops == 0 {
        let Some(angles) = detect_faint_cycle(distances, node_count)? else {
            return Ok(None);
        };
        let (values, _) = distance_eigensystem(distances, node_count)?;
        if values
            .iter()
            .filter(|value| **value > 1e-5 * values[node_count - 1])
            .count()
            < 2
        {
            return Ok(None);
        }
        return Ok(Some(PeriodicTopology {
            node_count,
            dimensions: 1,
            persistent_loops: 0,
            used_faint_cycle: true,
            angles,
        }));
    }
    let (values, vectors) = distance_eigensystem(distances, node_count)?;
    if values
        .iter()
        .filter(|value| **value > 1e-5 * values[node_count - 1])
        .count()
        <= persistent_loops
    {
        return Ok(None);
    }
    if let Some(angles) =
        periodic_mds_angles(distances, node_count, persistent_loops, &values, &vectors)?
    {
        return Ok(Some(PeriodicTopology {
            node_count,
            dimensions: persistent_loops,
            persistent_loops,
            used_faint_cycle: false,
            angles,
        }));
    }
    let full_columns = node_count - 1;
    let eigenvectors = untangle_periodic_eigenspaces(
        &eigensystem.nontrivial_eigenvalues,
        &eigensystem.nontrivial_eigenvectors,
        node_count,
    )?;
    let mut accepted: Vec<Vec<f64>> = Vec::new();
    let mut pair = 0;
    while pair * 2 + 1 < full_columns && accepted.len() < persistent_loops {
        let mut angle = vec![0.0; node_count];
        let mut radii = Vec::with_capacity(node_count);
        for (row, value) in angle.iter_mut().enumerate() {
            let cosine = eigenvectors[row * full_columns + pair * 2];
            let sine = eigenvectors[row * full_columns + pair * 2 + 1];
            *value = fp32(sine.atan2(cosine));
            radii.push(cosine * cosine + sine * sine);
        }
        let mean_radius = radii.iter().sum::<f64>() / node_count as f64;
        let deviation = (radii
            .iter()
            .map(|radius| (radius - mean_radius).powi(2))
            .sum::<f64>()
            / node_count as f64)
            .sqrt();
        if mean_radius <= 0.0
            || radii.iter().any(|radius| *radius < 0.1 * mean_radius)
            || deviation > 0.35 * mean_radius
        {
            return Ok(None);
        }
        if !is_angular_harmonic(&angle, &accepted) {
            accepted.push(angle);
        }
        pair += 1;
    }
    if accepted.len() != persistent_loops {
        return Ok(None);
    }
    let dimensions = accepted.len();
    let mut angles = vec![0.0; node_count * dimensions];
    for row in 0..node_count {
        for dimension in 0..dimensions {
            angles[row * dimensions + dimension] = accepted[dimension][row];
        }
    }
    if !periodic_neighborhoods_preserved(distances, &angles, node_count, dimensions) {
        return Ok(None);
    }
    Ok(Some(PeriodicTopology {
        node_count,
        dimensions,
        persistent_loops,
        used_faint_cycle: false,
        angles,
    }))
}

fn connected_components(mask: &[u8], node_count: usize) -> usize {
    let mut parents: Vec<usize> = (0..node_count).collect();
    for row in 0..node_count {
        for column in (row + 1)..node_count {
            if mask[row * node_count + column] != 0 {
                union_components(&mut parents, row, column);
            }
        }
    }
    let mut roots = Vec::with_capacity(node_count);
    for node in 0..node_count {
        let root = find_component(&mut parents, node);
        if !roots.contains(&root) {
            roots.push(root);
        }
    }
    roots.len()
}

fn find_component(parents: &mut [usize], mut node: usize) -> usize {
    while parents[node] != node {
        parents[node] = parents[parents[node]];
        node = parents[node];
    }
    node
}

fn union_components(parents: &mut [usize], left: usize, right: usize) -> bool {
    let left_root = find_component(parents, left);
    let right_root = find_component(parents, right);
    if left_root != right_root {
        parents[left_root] = right_root;
        true
    } else {
        false
    }
}

fn fp32(value: f64) -> f64 {
    f64::from(value as f32)
}

#[allow(clippy::too_many_arguments)]
pub fn select_topology_from_targets(
    consensus_gram: &[f64],
    node_count: usize,
    targets: &[f64],
    target_offsets: &[u32],
    max_dimensions: usize,
    variance_threshold: f64,
    requested_fit_mode: &str,
    min_dimensions: Option<usize>,
    k_nn: Option<usize>,
    bandwidth: Option<f64>,
    persistence_fraction: f64,
    smoothing: Option<f64>,
) -> KernelResult<TopologySelection> {
    if !matches!(requested_fit_mode, "pca" | "spectral" | "auto") {
        return Err(invalid("requested_fit_mode must be pca, spectral, or auto"));
    }
    if matches!(min_dimensions, Some(0)) {
        return Err(invalid("min_dimensions must be positive when provided"));
    }
    if node_count == 0 {
        return Err(invalid("topology selection needs at least one node"));
    }
    if target_offsets.len() < 2 {
        return Err(invalid("topology targets need at least one layer"));
    }
    if target_offsets[0] != 0 || target_offsets[target_offsets.len() - 1] as usize != targets.len()
    {
        return Err(invalid(
            "topology target offsets must start at zero and end at the target length",
        ));
    }
    if target_offsets.windows(2).any(|pair| {
        pair[0] >= pair[1] || !((pair[1] - pair[0]) as usize).is_multiple_of(node_count)
    }) {
        return Err(invalid(
            "each topology target layer must contain a non-empty node matrix",
        ));
    }
    checked_vector("topology targets", targets, targets.len())?;
    if !persistence_fraction.is_finite() || persistence_fraction < 0.0 {
        return Err(invalid(
            "persistence_fraction must be finite and non-negative",
        ));
    }
    if smoothing.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err(invalid(
            "topology smoothing must be finite and non-negative",
        ));
    }
    let targets = coerce_fp32("fp32 topology targets", targets)?;
    let flat = derive_pca_layout(
        consensus_gram,
        node_count,
        max_dimensions,
        variance_threshold,
    )?;
    let flat_score = ols_gcv_score(
        &flat.coordinates,
        node_count,
        flat.dimensions,
        &targets,
        target_offsets,
    )?;
    let pca_diagnostics = TopologyDiagnostics::Pca {
        per_component_variance: flat.explained_variance.clone(),
        cumulative_variance: flat.cumulative_variance.clone(),
        picked_dimensions: flat.dimensions,
        threshold: variance_threshold,
    };
    let mut candidates = vec![TopologyCandidate {
        name: "flat-pca".to_string(),
        fit_mode: "pca".to_string(),
        intrinsic_dimensions: flat.dimensions,
        score: flat_score,
        viable: true,
        reason: String::new(),
    }];

    if requested_fit_mode == "pca" {
        return Ok(TopologySelection {
            winner_name: "flat-pca".to_string(),
            fit_mode: "pca".to_string(),
            intrinsic_dimensions: flat.dimensions,
            periodic_dimensions: 0,
            embedded_coordinates: flat.coordinates.clone(),
            coordinates: flat.coordinates,
            candidates,
            persistent_loops: 0,
            used_faint_cycle: false,
            diagnostics: pca_diagnostics,
            winner_plan: None,
        });
    }

    let eigensystem = build_normalized_laplacian(consensus_gram, node_count, k_nn, bandwidth);
    let mut curved: Option<(SpectralEmbedding, f64, RbfFitPlan)> = None;
    let mut spectral_diagnostics: Option<TopologyDiagnostics> = None;
    match &eigensystem {
        Ok(eigen) => {
            let spectral_min_dimensions = if requested_fit_mode == "auto" {
                Some(flat.dimensions)
            } else {
                min_dimensions
            };
            let embedding =
                spectral_embedding_from_eigensystem(eigen, max_dimensions, spectral_min_dimensions);
            let attempted = embedding.and_then(|embedding| {
                spectral_diagnostics = Some(TopologyDiagnostics::Spectral {
                    eigenvalues: embedding.eigenvalues.clone(),
                    picked_dimensions: embedding.dimensions,
                    gap_magnitude: embedding.gap_magnitude,
                    bandwidth: embedding.bandwidth,
                    k_nn: embedding.k_nn,
                    heuristic_dimensions: embedding.heuristic_dimensions,
                    min_dimensions: embedding.min_dimensions,
                    pinned: embedding.pinned,
                });
                if 2 * embedding.dimensions + 1 > node_count {
                    return Err(invalid(format!(
                        "poisedness floor 2n+1={} exceeds K={node_count}",
                        2 * embedding.dimensions + 1
                    )));
                }
                let plan =
                    prepare_rbf_fit_plan(&embedding.coordinates, node_count, embedding.dimensions)?;
                let score = rbf_gcv_score(&plan, &targets, target_offsets, smoothing)?;
                Ok((embedding, score, plan))
            });
            match attempted {
                Ok((embedding, score, plan)) => {
                    candidates.push(TopologyCandidate {
                        name: "spectral".to_string(),
                        fit_mode: "spectral".to_string(),
                        intrinsic_dimensions: embedding.dimensions,
                        score,
                        viable: true,
                        reason: String::new(),
                    });
                    curved = Some((embedding, score, plan));
                }
                Err(error) => candidates.push(TopologyCandidate {
                    name: "spectral".to_string(),
                    fit_mode: "spectral".to_string(),
                    intrinsic_dimensions: flat.dimensions,
                    score: f64::INFINITY,
                    viable: false,
                    reason: error.to_string(),
                }),
            }
        }
        Err(error) => candidates.push(TopologyCandidate {
            name: "spectral".to_string(),
            fit_mode: "spectral".to_string(),
            intrinsic_dimensions: flat.dimensions,
            score: f64::INFINITY,
            viable: false,
            reason: error.to_string(),
        }),
    }

    let mut periodic: Option<(PeriodicTopology, Vec<f64>, f64, RbfFitPlan)> = None;
    if requested_fit_mode == "auto" && node_count <= PERSISTENCE_MAX_NODES {
        if let Ok(eigen) = &eigensystem {
            let attempted = (|| -> KernelResult<_> {
                let distances = pairwise_distances_from_gram(consensus_gram, node_count)?;
                let Some(topology) = periodic_topology_from_eigensystem(
                    &distances,
                    eigen,
                    max_dimensions,
                    persistence_fraction,
                )?
                else {
                    return Ok(None);
                };
                if 2 * topology.dimensions + 1 > node_count {
                    return Ok(None);
                }
                let embedded =
                    embed_periodic_angles(&topology.angles, node_count, topology.dimensions)?;
                let plan = prepare_rbf_fit_plan(&embedded, node_count, 2 * topology.dimensions)?;
                let score = rbf_gcv_score(&plan, &targets, target_offsets, smoothing)?;
                Ok(Some((topology, embedded, score, plan)))
            })();
            if let Ok(Some((topology, embedded, score, plan))) = attempted {
                let detected_count = if topology.persistent_loops == 0 {
                    topology.dimensions
                } else {
                    topology.persistent_loops
                };
                let mut null_score = 0.0;
                for offsets in target_offsets.windows(2) {
                    let layer = &targets[offsets[0] as usize..offsets[1] as usize];
                    let columns = layer.len() / node_count;
                    for column in 0..columns {
                        let mean = layer
                            .chunks_exact(columns)
                            .map(|row| row[column])
                            .sum::<f64>()
                            / node_count as f64;
                        let rss = layer
                            .chunks_exact(columns)
                            .map(|row| (row[column] - mean).powi(2))
                            .sum::<f64>();
                        null_score += node_count as f64 * rss / (node_count - 1).pow(2) as f64;
                    }
                }
                let viable = score.is_finite() && null_score > 0.0 && score <= null_score;
                candidates.push(TopologyCandidate {
                    name: format!("torus-T{}", topology.dimensions),
                    fit_mode: "spectral".to_string(),
                    intrinsic_dimensions: topology.dimensions,
                    score,
                    viable,
                    reason: if viable {
                        format!("Periodic chart candidate; H1 evidence = {detected_count}; not a topology certificate")
                    } else {
                        "Periodic fit does not improve on the mean-only prediction baseline".to_string()
                    },
                });
                if viable {
                    periodic = Some((topology, embedded, score, plan));
                }
            } else if let Err(error) = attempted {
                candidates.push(TopologyCandidate {
                    name: "periodic-unresolved".to_string(),
                    fit_mode: "spectral".to_string(),
                    intrinsic_dimensions: flat.dimensions,
                    score: f64::INFINITY,
                    viable: false,
                    reason: error.to_string(),
                });
            }
        }
    }

    let (
        winner_name,
        fit_mode,
        intrinsic_dimensions,
        periodic_dimensions,
        coordinates,
        embedded_coordinates,
        persistent_loops,
        used_faint_cycle,
        winner_plan,
    ) = match requested_fit_mode {
        "pca" => select_flat_winner(flat),
        "spectral" => select_spectral_winner(curved)?,
        "auto" => {
            if let Some((topology, embedded, score, plan)) = periodic {
                if score.is_finite() {
                    (
                        format!("torus-T{}", topology.dimensions),
                        "spectral".to_string(),
                        topology.dimensions,
                        topology.dimensions,
                        topology.angles,
                        embedded,
                        topology.persistent_loops,
                        topology.used_faint_cycle,
                        Some(plan),
                    )
                } else {
                    select_nonperiodic_winner(flat, flat_score, curved)
                }
            } else {
                select_nonperiodic_winner(flat, flat_score, curved)
            }
        }
        _ => unreachable!(),
    };

    candidates.sort_by(|left, right| {
        right
            .viable
            .cmp(&left.viable)
            .then_with(|| left.score.total_cmp(&right.score))
    });
    let diagnostics = if fit_mode == "pca" {
        pca_diagnostics
    } else {
        spectral_diagnostics.ok_or_else(|| {
            invalid("the selected spectral topology has no coordinate diagnostics")
        })?
    };
    Ok(TopologySelection {
        winner_name,
        fit_mode,
        intrinsic_dimensions,
        periodic_dimensions,
        coordinates,
        embedded_coordinates,
        candidates,
        persistent_loops,
        used_faint_cycle,
        diagnostics,
        winner_plan,
    })
}

#[allow(clippy::type_complexity)]
fn select_flat_winner(
    flat: PcaLayout,
) -> (
    String,
    String,
    usize,
    usize,
    Vec<f64>,
    Vec<f64>,
    usize,
    bool,
    Option<RbfFitPlan>,
) {
    (
        "flat-pca".to_string(),
        "pca".to_string(),
        flat.dimensions,
        0,
        flat.coordinates.clone(),
        flat.coordinates,
        0,
        false,
        None,
    )
}

#[allow(clippy::type_complexity)]
fn select_spectral_winner(
    curved: Option<(SpectralEmbedding, f64, RbfFitPlan)>,
) -> KernelResult<(
    String,
    String,
    usize,
    usize,
    Vec<f64>,
    Vec<f64>,
    usize,
    bool,
    Option<RbfFitPlan>,
)> {
    let Some((embedding, _score, plan)) = curved else {
        return Err(invalid("the requested spectral topology is not viable"));
    };
    Ok((
        "spectral".to_string(),
        "spectral".to_string(),
        embedding.dimensions,
        0,
        embedding.coordinates.clone(),
        embedding.coordinates,
        0,
        false,
        Some(plan),
    ))
}

#[allow(clippy::type_complexity)]
fn select_nonperiodic_winner(
    flat: PcaLayout,
    flat_score: f64,
    curved: Option<(SpectralEmbedding, f64, RbfFitPlan)>,
) -> (
    String,
    String,
    usize,
    usize,
    Vec<f64>,
    Vec<f64>,
    usize,
    bool,
    Option<RbfFitPlan>,
) {
    if let Some((embedding, score, plan)) = curved {
        if score < flat_score {
            return (
                "spectral".to_string(),
                "spectral".to_string(),
                embedding.dimensions,
                0,
                embedding.coordinates.clone(),
                embedding.coordinates,
                0,
                false,
                Some(plan),
            );
        }
    }
    select_flat_winner(flat)
}

fn ols_gcv_score(
    coordinates: &[f64],
    node_count: usize,
    coordinate_dimensions: usize,
    targets: &[f64],
    target_offsets: &[u32],
) -> KernelResult<f64> {
    checked_matrix(
        "OLS topology coordinates",
        coordinates,
        node_count,
        coordinate_dimensions,
    )?;
    let design_columns = coordinate_dimensions + 1;
    let mut design = vec![0.0; node_count * design_columns];
    for row in 0..node_count {
        design[row * design_columns] = 1.0;
        design[(row * design_columns + 1)..((row + 1) * design_columns)].copy_from_slice(
            &coordinates[row * coordinate_dimensions..(row + 1) * coordinate_dimensions],
        );
    }
    let rank = matrix_rank(&design, node_count, design_columns)?;
    if rank >= node_count {
        return Ok(f64::INFINITY);
    }
    let mut normal = vec![0.0; design_columns * design_columns];
    for left in 0..design_columns {
        for right in left..design_columns {
            let value = (0..node_count)
                .map(|row| {
                    design[row * design_columns + left] * design[row * design_columns + right]
                })
                .sum::<f64>();
            normal[left * design_columns + right] = value;
            normal[right * design_columns + left] = value;
        }
    }
    let (eigenvalues, eigenvectors) = symmetric_eigen(&normal, design_columns)?;
    let leading_singular = eigenvalues[0].max(0.0).sqrt();
    let tolerance =
        node_count.max(design_columns) as f64 * f64::from(f32::EPSILON) * leading_singular;
    let mut sample_basis = Vec::with_capacity(rank * node_count);
    for component in 0..design_columns {
        let eigenvalue = eigenvalues[component].max(0.0);
        if eigenvalue.sqrt() <= tolerance {
            continue;
        }
        let inverse_scale = 1.0 / eigenvalue.sqrt();
        for row in 0..node_count {
            let value = (0..design_columns)
                .map(|column| {
                    design[row * design_columns + column]
                        * eigenvectors[column * design_columns + component]
                })
                .sum::<f64>()
                * inverse_scale;
            sample_basis.push(value);
        }
    }
    let actual_rank = sample_basis.len() / node_count;
    let slack = node_count - actual_rank;
    if slack == 0 {
        return Ok(f64::INFINITY);
    }
    let mut total = 0.0;
    for offsets in target_offsets.windows(2) {
        let start = offsets[0] as usize;
        let end = offsets[1] as usize;
        let values = &targets[start..end];
        let target_dimensions = values.len() / node_count;
        let mut rss = 0.0;
        for output in 0..target_dimensions {
            let projections: Vec<f64> = sample_basis
                .chunks_exact(node_count)
                .map(|basis| {
                    (0..node_count)
                        .map(|row| basis[row] * values[row * target_dimensions + output])
                        .sum::<f64>()
                })
                .collect();
            for row in 0..node_count {
                let fitted = sample_basis
                    .chunks_exact(node_count)
                    .zip(&projections)
                    .map(|(basis, projection)| basis[row] * projection)
                    .sum::<f64>();
                let residual = values[row * target_dimensions + output] - fitted;
                rss += residual * residual;
            }
        }
        total += node_count as f64 * rss / (slack * slack) as f64;
    }
    Ok(total)
}

fn rbf_gcv_score(
    plan: &RbfFitPlan,
    targets: &[f64],
    target_offsets: &[u32],
    smoothing: Option<f64>,
) -> KernelResult<f64> {
    let mut total = 0.0;
    for offsets in target_offsets.windows(2) {
        let start = offsets[0] as usize;
        let end = offsets[1] as usize;
        let values = &targets[start..end];
        let target_dimensions = values.len() / plan.node_count();
        total += plan.gcv_score(values, target_dimensions, smoothing)?;
    }
    Ok(total)
}

fn embed_periodic_angles(
    angles: &[f64],
    node_count: usize,
    dimensions: usize,
) -> KernelResult<Vec<f64>> {
    checked_matrix("periodic angles", angles, node_count, dimensions)?;
    let embedded_dimensions = 2 * dimensions;
    let mut embedded = vec![0.0; node_count * embedded_dimensions];
    for row in 0..node_count {
        for dimension in 0..dimensions {
            let angle = angles[row * dimensions + dimension];
            embedded[row * embedded_dimensions + 2 * dimension] = fp32(angle.cos());
            embedded[row * embedded_dimensions + 2 * dimension + 1] = fp32(angle.sin());
        }
    }
    checked_finite("embedded periodic coordinates", &embedded)?;
    Ok(embedded)
}

pub fn fit_rbf_smoothed(
    raw_nodes: &[f64],
    node_count: usize,
    input_dimensions: usize,
    values: &[f64],
    output_dimensions: usize,
    smoothing: f64,
) -> KernelResult<RbfModel> {
    prepare_rbf_fit_plan(raw_nodes, node_count, input_dimensions)?.fit_smoothed(
        values,
        output_dimensions,
        smoothing,
    )
}

pub fn fit_rbf_auto_smoothed(
    raw_nodes: &[f64],
    node_count: usize,
    input_dimensions: usize,
    values: &[f64],
    output_dimensions: usize,
) -> KernelResult<RbfModel> {
    prepare_rbf_fit_plan(raw_nodes, node_count, input_dimensions)?
        .fit_auto_smoothed(values, output_dimensions)
}

pub fn prepare_rbf_fit_plan(
    raw_nodes: &[f64],
    node_count: usize,
    input_dimensions: usize,
) -> KernelResult<RbfFitPlan> {
    let geometry = prepare_rbf_geometry(raw_nodes, node_count, input_dimensions)?;
    let polynomial_columns = input_dimensions + 1;
    let null_dimensions = node_count - polynomial_columns;
    let lambdas: Vec<f64> = (0..RBF_GCV_GRID_POINTS)
        .map(|index| {
            let exponent = -6.0 + 9.0 * index as f64 / (RBF_GCV_GRID_POINTS - 1) as f64;
            geometry.kernel_scale * 10.0_f64.powf(exponent)
        })
        .collect();
    checked_finite("RBF GCV lambda grid", &lambdas)?;
    if null_dimensions == 0 {
        return Ok(RbfFitPlan {
            geometry,
            lambdas,
            spectral_basis: Vec::new(),
            residual_ratios: Vec::new(),
            residual_traces: vec![0.0; RBF_GCV_GRID_POINTS],
        });
    }

    let orthonormal_elements = checked_output("RBF orthonormal basis", &[node_count, node_count])?;
    let null_basis_elements =
        checked_output("RBF null-space basis", &[null_dimensions, node_count])?;
    let reduced_elements =
        checked_output("RBF reduced kernel", &[null_dimensions, null_dimensions])?;
    let spectral_elements =
        checked_output("RBF GCV spectral basis", &[node_count, null_dimensions])?;
    let ratio_elements = checked_output(
        "RBF GCV residual ratios",
        &[RBF_GCV_GRID_POINTS, null_dimensions],
    )?;
    checked_work(
        "RBF fit plan",
        &[
            geometry.nodes.len(),
            geometry.kernel.len(),
            orthonormal_elements,
            null_basis_elements,
            spectral_elements,
            reduced_elements,
            reduced_elements,
            spectral_elements,
            ratio_elements,
            RBF_GCV_GRID_POINTS,
        ],
    )?;

    let null_basis = rbf_null_space_basis(
        &geometry.nodes,
        node_count,
        input_dimensions,
        null_dimensions,
    )?;
    let mut kernel_times_null = vec![0.0; spectral_elements];
    for row in 0..node_count {
        for component in 0..null_dimensions {
            kernel_times_null[row * null_dimensions + component] = (0..node_count)
                .map(|source| {
                    geometry.kernel[row * node_count + source]
                        * null_basis[component * node_count + source]
                })
                .sum();
        }
    }
    let mut reduced_kernel = vec![0.0; reduced_elements];
    for left in 0..null_dimensions {
        for right in left..null_dimensions {
            let value = (0..node_count)
                .map(|row| {
                    null_basis[left * node_count + row]
                        * kernel_times_null[row * null_dimensions + right]
                })
                .sum::<f64>();
            let transpose = (0..node_count)
                .map(|row| {
                    null_basis[right * node_count + row]
                        * kernel_times_null[row * null_dimensions + left]
                })
                .sum::<f64>();
            let symmetric = 0.5 * (value + transpose);
            reduced_kernel[left * null_dimensions + right] = symmetric;
            reduced_kernel[right * null_dimensions + left] = symmetric;
        }
    }
    let (mut gamma, eigenvectors) = symmetric_eigen(&reduced_kernel, null_dimensions)?;
    for value in &mut gamma {
        *value = value.max(0.0);
    }
    let mut spectral_basis = vec![0.0; spectral_elements];
    for row in 0..node_count {
        for eigenvector in 0..null_dimensions {
            spectral_basis[row * null_dimensions + eigenvector] = (0..null_dimensions)
                .map(|component| {
                    null_basis[component * node_count + row]
                        * eigenvectors[component * null_dimensions + eigenvector]
                })
                .sum();
        }
    }
    let mut residual_ratios = vec![0.0; ratio_elements];
    let mut residual_traces = vec![0.0; RBF_GCV_GRID_POINTS];
    for (grid_index, lambda) in lambdas.iter().enumerate() {
        for (component, eigenvalue) in gamma.iter().enumerate() {
            let ratio = lambda / (eigenvalue + lambda);
            residual_ratios[grid_index * null_dimensions + component] = ratio;
            residual_traces[grid_index] += ratio;
        }
    }
    checked_finite("RBF GCV spectral basis", &spectral_basis)?;
    checked_finite("RBF GCV residual ratios", &residual_ratios)?;
    checked_finite("RBF GCV residual traces", &residual_traces)?;
    Ok(RbfFitPlan {
        geometry,
        lambdas,
        spectral_basis,
        residual_ratios,
        residual_traces,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn restore_rbf_fit_plan(
    input_dimensions: usize,
    node_count: usize,
    nodes: &[f64],
    coordinate_offset: &[f64],
    coordinate_scale: &[f64],
    kernel: &[f64],
    kernel_scale: f64,
    lambdas: &[f64],
    spectral_basis: &[f64],
    residual_ratios: &[f64],
    residual_traces: &[f64],
) -> KernelResult<RbfFitPlan> {
    let polynomial_columns = input_dimensions
        .checked_add(1)
        .ok_or_else(|| invalid("RBF plan polynomial dimension overflows usize"))?;
    if input_dimensions == 0 || node_count < polynomial_columns {
        return Err(invalid("serialized RBF plan has invalid dimensions"));
    }
    let system_size = node_count
        .checked_add(polynomial_columns)
        .ok_or_else(|| invalid("RBF plan system dimension overflows usize"))?;
    if system_size > RBF_MAX_SYSTEM_DIMENSION {
        return Err(invalid(format!(
            "RBF system dimension {system_size} exceeds the {RBF_MAX_SYSTEM_DIMENSION}-dimension compute ceiling"
        )));
    }
    let null_dimensions = node_count - polynomial_columns;
    checked_matrix(
        "serialized RBF plan nodes",
        nodes,
        node_count,
        input_dimensions,
    )?;
    checked_vector(
        "serialized RBF plan coordinate offset",
        coordinate_offset,
        input_dimensions,
    )?;
    checked_vector(
        "serialized RBF plan coordinate scale",
        coordinate_scale,
        input_dimensions,
    )?;
    checked_matrix("serialized RBF plan kernel", kernel, node_count, node_count)?;
    checked_vector("serialized RBF plan lambdas", lambdas, RBF_GCV_GRID_POINTS)?;
    if null_dimensions == 0 {
        if !spectral_basis.is_empty() || !residual_ratios.is_empty() {
            return Err(invalid(
                "serialized full-polynomial RBF plan must have empty spectral arrays",
            ));
        }
    } else {
        checked_matrix(
            "serialized RBF plan spectral basis",
            spectral_basis,
            node_count,
            null_dimensions,
        )?;
        checked_matrix(
            "serialized RBF plan residual ratios",
            residual_ratios,
            RBF_GCV_GRID_POINTS,
            null_dimensions,
        )?;
    }
    checked_vector(
        "serialized RBF plan residual traces",
        residual_traces,
        RBF_GCV_GRID_POINTS,
    )?;
    if !kernel_scale.is_finite() || kernel_scale <= 0.0 {
        return Err(invalid(
            "serialized RBF plan kernel scale must be finite and positive",
        ));
    }
    if coordinate_scale.iter().any(|value| *value <= 0.0)
        || lambdas.iter().any(|value| *value <= 0.0)
        || residual_ratios.iter().any(|value| *value < 0.0)
        || residual_traces.iter().any(|value| *value < 0.0)
    {
        return Err(invalid("serialized RBF plan contains an invalid scale"));
    }
    if lambdas.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(invalid(
            "serialized RBF plan lambdas must be strictly increasing",
        ));
    }
    Ok(RbfFitPlan {
        geometry: RbfGeometry {
            input_dimensions,
            node_count,
            nodes: nodes.to_vec(),
            coordinate_offset: coordinate_offset.to_vec(),
            coordinate_scale: coordinate_scale.to_vec(),
            kernel: kernel.to_vec(),
            kernel_scale,
        },
        lambdas: lambdas.to_vec(),
        spectral_basis: spectral_basis.to_vec(),
        residual_ratios: residual_ratios.to_vec(),
        residual_traces: residual_traces.to_vec(),
    })
}

impl RbfFitPlan {
    pub fn node_count(&self) -> usize {
        self.geometry.node_count
    }

    pub fn input_dimensions(&self) -> usize {
        self.geometry.input_dimensions
    }

    pub fn nodes(&self) -> &[f64] {
        &self.geometry.nodes
    }

    pub fn coordinate_offset(&self) -> &[f64] {
        &self.geometry.coordinate_offset
    }

    pub fn coordinate_scale(&self) -> &[f64] {
        &self.geometry.coordinate_scale
    }

    pub fn kernel(&self) -> &[f64] {
        &self.geometry.kernel
    }

    pub fn kernel_scale(&self) -> f64 {
        self.geometry.kernel_scale
    }

    pub fn lambdas(&self) -> &[f64] {
        &self.lambdas
    }

    pub fn spectral_basis(&self) -> &[f64] {
        &self.spectral_basis
    }

    pub fn residual_ratios(&self) -> &[f64] {
        &self.residual_ratios
    }

    pub fn residual_traces(&self) -> &[f64] {
        &self.residual_traces
    }

    pub fn fit_smoothed(
        &self,
        values: &[f64],
        output_dimensions: usize,
        smoothing: f64,
    ) -> KernelResult<RbfModel> {
        if !smoothing.is_finite() || smoothing < 0.0 {
            return Err(invalid("smoothing must be finite and non-negative"));
        }
        checked_matrix(
            "RBF values",
            values,
            self.geometry.node_count,
            output_dimensions,
        )?;
        let values = coerce_fp32("fp32 RBF values", values)?;
        let lambda = smoothing * self.geometry.kernel_scale;
        if !lambda.is_finite() {
            return Err(invalid("scaled RBF lambda is not finite"));
        }
        let system = build_rbf_system(
            &self.geometry.kernel,
            &self.geometry.nodes,
            self.geometry.node_count,
            self.geometry.input_dimensions,
            lambda,
        );
        let effective_degrees_of_freedom = if smoothing == 0.0 {
            self.geometry.node_count as f64
        } else {
            rbf_smoother_stats(
                &system,
                &self.geometry.kernel,
                &self.geometry.nodes,
                self.geometry.node_count,
                self.geometry.input_dimensions,
                None,
                output_dimensions,
            )?
            .0
        };
        solve_rbf_geometry(
            &self.geometry,
            &values,
            output_dimensions,
            system,
            lambda,
            effective_degrees_of_freedom,
            -1.0,
        )
    }

    pub fn gcv_score(
        &self,
        values: &[f64],
        output_dimensions: usize,
        smoothing: Option<f64>,
    ) -> KernelResult<f64> {
        let Some(smoothing) = smoothing else {
            return Ok(self.fit_auto_smoothed(values, output_dimensions)?.gcv);
        };
        if !smoothing.is_finite() || smoothing < 0.0 {
            return Err(invalid("smoothing must be finite and non-negative"));
        }
        checked_matrix(
            "RBF values",
            values,
            self.geometry.node_count,
            output_dimensions,
        )?;
        if smoothing == 0.0 {
            return Ok(f64::INFINITY);
        }
        let values = coerce_fp32("fp32 RBF values", values)?;
        let lambda = smoothing * self.geometry.kernel_scale;
        if !lambda.is_finite() {
            return Err(invalid("scaled RBF lambda is not finite"));
        }
        let system = build_rbf_system(
            &self.geometry.kernel,
            &self.geometry.nodes,
            self.geometry.node_count,
            self.geometry.input_dimensions,
            lambda,
        );
        let (effective_degrees_of_freedom, residual_sum) = rbf_smoother_stats(
            &system,
            &self.geometry.kernel,
            &self.geometry.nodes,
            self.geometry.node_count,
            self.geometry.input_dimensions,
            Some(&values),
            output_dimensions,
        )?;
        Ok(gcv_value(
            residual_sum,
            effective_degrees_of_freedom,
            self.geometry.node_count,
        ))
    }

    pub fn fit_auto_smoothed(
        &self,
        values: &[f64],
        output_dimensions: usize,
    ) -> KernelResult<RbfModel> {
        checked_matrix(
            "RBF values",
            values,
            self.geometry.node_count,
            output_dimensions,
        )?;
        let values = coerce_fp32("fp32 RBF values", values)?;
        let null_dimensions = self.geometry.node_count - self.geometry.input_dimensions - 1;
        let (lambda, effective_degrees_of_freedom, gcv) = if null_dimensions == 0 {
            (
                self.lambdas[0],
                self.geometry.node_count as f64,
                f64::INFINITY,
            )
        } else {
            let projected_elements = checked_output(
                "RBF GCV projected values",
                &[null_dimensions, output_dimensions],
            )?;
            checked_work(
                "RBF GCV fit",
                &[
                    values.len(),
                    self.spectral_basis.len(),
                    self.residual_ratios.len(),
                    projected_elements,
                    null_dimensions,
                ],
            )?;
            let mut energy = vec![0.0; null_dimensions];
            for (component, component_energy) in energy.iter_mut().enumerate() {
                for output in 0..output_dimensions {
                    let projection = (0..self.geometry.node_count)
                        .map(|row| {
                            self.spectral_basis[row * null_dimensions + component]
                                * values[row * output_dimensions + output]
                        })
                        .sum::<f64>();
                    *component_energy += projection * projection;
                }
            }
            let mut best_index = 0;
            let mut best_gcv = f64::INFINITY;
            for grid_index in 0..RBF_GCV_GRID_POINTS {
                let residual_trace = self.residual_traces[grid_index];
                if residual_trace <= 0.0 {
                    continue;
                }
                let residual_sum = (0..null_dimensions)
                    .map(|component| {
                        let ratio = self.residual_ratios[grid_index * null_dimensions + component];
                        ratio * ratio * energy[component]
                    })
                    .sum::<f64>();
                let candidate = self.geometry.node_count as f64 * residual_sum
                    / (residual_trace * residual_trace);
                if candidate < best_gcv {
                    best_index = grid_index;
                    best_gcv = candidate;
                }
            }
            (
                self.lambdas[best_index],
                self.geometry.node_count as f64 - self.residual_traces[best_index],
                best_gcv,
            )
        };
        let system = build_rbf_system(
            &self.geometry.kernel,
            &self.geometry.nodes,
            self.geometry.node_count,
            self.geometry.input_dimensions,
            lambda,
        );
        solve_rbf_geometry(
            &self.geometry,
            &values,
            output_dimensions,
            system,
            lambda,
            effective_degrees_of_freedom,
            gcv,
        )
    }
}

fn gcv_value(residual_sum: f64, effective_degrees_of_freedom: f64, node_count: usize) -> f64 {
    let slack = node_count as f64 - effective_degrees_of_freedom;
    if slack <= 0.0 {
        f64::INFINITY
    } else {
        node_count as f64 * residual_sum / (slack * slack)
    }
}

fn prepare_rbf_geometry(
    raw_nodes: &[f64],
    node_count: usize,
    input_dimensions: usize,
) -> KernelResult<RbfGeometry> {
    checked_matrix("RBF nodes", raw_nodes, node_count, input_dimensions)?;
    let polynomial_columns = input_dimensions
        .checked_add(1)
        .ok_or_else(|| invalid("RBF polynomial dimension overflows usize"))?;
    if node_count < polynomial_columns {
        return Err(invalid(format!(
            "RBF needs at least {} nodes for {input_dimensions} input dimensions",
            input_dimensions + 1
        )));
    }

    let system_size = node_count
        .checked_add(polynomial_columns)
        .ok_or_else(|| invalid("RBF system dimension overflows usize"))?;
    if system_size > RBF_MAX_SYSTEM_DIMENSION {
        return Err(invalid(format!(
            "RBF system dimension {system_size} exceeds the {RBF_MAX_SYSTEM_DIMENSION}-dimension compute ceiling"
        )));
    }
    let raw_nodes = coerce_fp32("fp32 RBF nodes", raw_nodes)?;
    let node_mean = mean_rows(&raw_nodes, node_count, input_dimensions);
    let centered = center_rows(&raw_nodes, input_dimensions, &node_mean);
    if matrix_rank(&centered, node_count, input_dimensions)? != input_dimensions {
        return Err(invalid(
            "RBF nodes are not affinely poised across every input dimension under the Python fp32 rank policy",
        ));
    }

    let mut coordinate_offset = vec![f64::INFINITY; input_dimensions];
    let mut maxima = vec![f64::NEG_INFINITY; input_dimensions];
    for row in raw_nodes.chunks_exact(input_dimensions) {
        for (dimension, value) in row.iter().enumerate() {
            coordinate_offset[dimension] = coordinate_offset[dimension].min(*value);
            maxima[dimension] = maxima[dimension].max(*value);
        }
    }
    let coordinate_scale: Vec<f64> = maxima
        .iter()
        .zip(&coordinate_offset)
        .map(|(maximum, minimum)| (maximum - minimum).max(1e-9))
        .collect();
    let nodes: Vec<f64> = raw_nodes
        .chunks_exact(input_dimensions)
        .flat_map(|row| {
            row.iter().enumerate().map(|(dimension, value)| {
                (value - coordinate_offset[dimension]) / coordinate_scale[dimension]
            })
        })
        .collect();
    checked_finite("normalized RBF nodes", &nodes)?;

    let kernel_elements = checked_output("RBF kernel", &[node_count, node_count])?;
    let system_elements = checked_output("RBF saddle system", &[system_size, system_size])?;
    checked_work(
        "RBF geometry",
        &[
            raw_nodes.len(),
            centered.len(),
            nodes.len(),
            kernel_elements,
            system_elements,
        ],
    )?;
    let mut kernel = vec![0.0; kernel_elements];
    let mut off_diagonal_sum = 0.0;
    for left in 0..node_count {
        for right in (left + 1)..node_count {
            let squared = (0..input_dimensions)
                .map(|dimension| {
                    let difference = nodes[left * input_dimensions + dimension]
                        - nodes[right * input_dimensions + dimension];
                    difference * difference
                })
                .sum::<f64>();
            let value = squared.sqrt().powi(3);
            kernel[left * node_count + right] = value;
            kernel[right * node_count + left] = value;
            off_diagonal_sum += 2.0 * value.abs();
        }
    }
    let off_diagonal_count = kernel_elements - node_count;
    let kernel_scale = if off_diagonal_count == 0 {
        1.0
    } else {
        (off_diagonal_sum / off_diagonal_count as f64).max(f64::EPSILON)
    };
    checked_finite("RBF coordinate offset", &coordinate_offset)?;
    checked_finite("RBF coordinate scale", &coordinate_scale)?;
    checked_finite("RBF kernel", &kernel)?;
    Ok(RbfGeometry {
        input_dimensions,
        node_count,
        nodes,
        coordinate_offset,
        coordinate_scale,
        kernel,
        kernel_scale,
    })
}

fn coerce_fp32(name: &str, values: &[f64]) -> KernelResult<Vec<f64>> {
    let coerced: Vec<f64> = values
        .iter()
        .map(|value| f64::from(*value as f32))
        .collect();
    checked_finite(name, &coerced)?;
    Ok(coerced)
}

fn rbf_null_space_basis(
    nodes: &[f64],
    node_count: usize,
    input_dimensions: usize,
    null_dimensions: usize,
) -> KernelResult<Vec<f64>> {
    let mut basis = Vec::with_capacity(node_count * node_count);
    let mut constant = vec![1.0; node_count];
    append_orthonormal(&mut basis, &mut constant, node_count)?;
    for dimension in 0..input_dimensions {
        let mut column: Vec<f64> = (0..node_count)
            .map(|row| nodes[row * input_dimensions + dimension])
            .collect();
        append_orthonormal(&mut basis, &mut column, node_count)?;
    }
    for seed in 0..node_count {
        if basis.len() == node_count * node_count {
            break;
        }
        let mut candidate = vec![0.0; node_count];
        candidate[seed] = 1.0;
        let _ = append_orthonormal(&mut basis, &mut candidate, node_count);
    }
    let polynomial_columns = input_dimensions + 1;
    if basis.len() != node_count * node_count {
        return Err(invalid(
            "RBF null-space basis construction did not span the node space",
        ));
    }
    let null_basis = basis[polynomial_columns * node_count..].to_vec();
    if null_basis.len() != null_dimensions * node_count {
        return Err(invalid("RBF null-space basis has an invalid shape"));
    }
    checked_finite("RBF null-space basis", &null_basis)?;
    Ok(null_basis)
}

fn append_orthonormal(
    basis: &mut Vec<f64>,
    candidate: &mut [f64],
    width: usize,
) -> KernelResult<()> {
    for _ in 0..2 {
        for component in basis.chunks_exact(width) {
            let projection = candidate
                .iter()
                .zip(component)
                .map(|(value, basis_value)| value * basis_value)
                .sum::<f64>();
            for (value, basis_value) in candidate.iter_mut().zip(component) {
                *value -= projection * basis_value;
            }
        }
    }
    let norm = candidate
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= 1e-10 {
        return Err(invalid("RBF orthonormal basis candidate is rank deficient"));
    }
    for value in candidate.iter_mut() {
        *value /= norm;
    }
    basis.extend_from_slice(candidate);
    Ok(())
}

fn solve_rbf_geometry(
    geometry: &RbfGeometry,
    values: &[f64],
    output_dimensions: usize,
    system: Vec<f64>,
    lambda: f64,
    effective_degrees_of_freedom: f64,
    gcv: f64,
) -> KernelResult<RbfModel> {
    let system_size = geometry.node_count + geometry.input_dimensions + 1;
    let rhs_elements = checked_output("RBF right side", &[system_size, output_dimensions])?;
    checked_work(
        "RBF fit",
        &[
            geometry.nodes.len(),
            geometry.kernel.len(),
            values.len(),
            system.len(),
            system.len(),
            rhs_elements,
            rhs_elements,
        ],
    )?;
    let mut right_side = vec![0.0; rhs_elements];
    right_side[..values.len()].copy_from_slice(values);
    let solution = solve(system, system_size, right_side, output_dimensions)?;
    let weights = solution[..geometry.node_count * output_dimensions].to_vec();
    let polynomial = solution[geometry.node_count * output_dimensions..].to_vec();
    checked_finite("RBF weights", &weights)?;
    checked_finite("RBF polynomial", &polynomial)?;
    if !effective_degrees_of_freedom.is_finite() {
        return Err(invalid("RBF effective degrees of freedom is not finite"));
    }
    if !gcv.is_finite() && gcv != f64::INFINITY {
        return Err(invalid("RBF GCV score is invalid"));
    }
    Ok(RbfModel {
        input_dimensions: geometry.input_dimensions,
        output_dimensions,
        node_count: geometry.node_count,
        nodes: geometry.nodes.clone(),
        coordinate_offset: geometry.coordinate_offset.clone(),
        coordinate_scale: geometry.coordinate_scale.clone(),
        weights,
        polynomial,
        lambda,
        effective_degrees_of_freedom,
        gcv,
    })
}

fn build_rbf_system(
    kernel: &[f64],
    nodes: &[f64],
    node_count: usize,
    input_dimensions: usize,
    lambda: f64,
) -> Vec<f64> {
    let polynomial_columns = input_dimensions + 1;
    let system_size = node_count + polynomial_columns;
    let mut system = vec![0.0; system_size * system_size];
    for row in 0..node_count {
        for column in 0..node_count {
            system[row * system_size + column] = kernel[row * node_count + column];
        }
        system[row * system_size + row] += lambda;
        system[row * system_size + node_count] = 1.0;
        system[node_count * system_size + row] = 1.0;
        for dimension in 0..input_dimensions {
            let value = nodes[row * input_dimensions + dimension];
            system[row * system_size + node_count + 1 + dimension] = value;
            system[(node_count + 1 + dimension) * system_size + row] = value;
        }
    }
    system
}

fn rbf_smoother_stats(
    system: &[f64],
    kernel: &[f64],
    nodes: &[f64],
    node_count: usize,
    input_dimensions: usize,
    values: Option<&[f64]>,
    output_dimensions: usize,
) -> KernelResult<(f64, f64)> {
    let system_size = node_count + input_dimensions + 1;
    let mut identity_rhs = vec![0.0; system_size * node_count];
    for index in 0..node_count {
        identity_rhs[index * node_count + index] = 1.0;
    }
    let solution = solve(system.to_vec(), system_size, identity_rhs, node_count)?;
    let weights = &solution[..node_count * node_count];
    let polynomial = &solution[node_count * node_count..];
    let mut trace = 0.0;
    let mut residual_sum = 0.0;
    for row in 0..node_count {
        let mut fitted = vec![0.0; output_dimensions];
        for source in 0..node_count {
            let mut smoother = polynomial[source];
            for node in 0..node_count {
                smoother += kernel[row * node_count + node] * weights[node * node_count + source];
            }
            for dimension in 0..input_dimensions {
                smoother += nodes[row * input_dimensions + dimension]
                    * polynomial[(dimension + 1) * node_count + source];
            }
            if row == source {
                trace += smoother;
            }
            if let Some(values) = values {
                for output in 0..output_dimensions {
                    fitted[output] += smoother * values[source * output_dimensions + output];
                }
            }
        }
        if let Some(values) = values {
            for output in 0..output_dimensions {
                let residual = values[row * output_dimensions + output] - fitted[output];
                residual_sum += residual * residual;
            }
        }
    }
    Ok((trace, residual_sum))
}

pub fn evaluate_rbf(model: &RbfModel, queries: &[f64], rows: usize) -> KernelResult<Vec<f64>> {
    checked_matrix("RBF queries", queries, rows, model.input_dimensions)?;
    checked_matrix(
        "stored RBF nodes",
        &model.nodes,
        model.node_count,
        model.input_dimensions,
    )?;
    checked_vector(
        "stored RBF coordinate offset",
        &model.coordinate_offset,
        model.input_dimensions,
    )?;
    checked_vector(
        "stored RBF coordinate scale",
        &model.coordinate_scale,
        model.input_dimensions,
    )?;
    if model.coordinate_scale.iter().any(|scale| *scale <= 0.0) {
        return Err(invalid("stored RBF coordinate scale must be positive"));
    }
    checked_matrix(
        "stored RBF weights",
        &model.weights,
        model.node_count,
        model.output_dimensions,
    )?;
    let polynomial_rows = model
        .input_dimensions
        .checked_add(1)
        .ok_or_else(|| invalid("stored RBF polynomial dimension overflows usize"))?;
    checked_matrix(
        "stored RBF polynomial",
        &model.polynomial,
        polynomial_rows,
        model.output_dimensions,
    )?;
    let output_elements = checked_output("RBF result", &[rows, model.output_dimensions])?;
    checked_work(
        "RBF evaluation",
        &[
            queries.len(),
            output_elements,
            model.nodes.len(),
            model.weights.len(),
        ],
    )?;
    let mut output = vec![0.0; output_elements];
    for row in 0..rows {
        let normalized: Vec<f64> = (0..model.input_dimensions)
            .map(|dimension| {
                (queries[row * model.input_dimensions + dimension]
                    - model.coordinate_offset[dimension])
                    / model.coordinate_scale[dimension]
            })
            .collect();
        for output_dimension in 0..model.output_dimensions {
            let mut value = model.polynomial[output_dimension];
            for (dimension, coordinate) in normalized.iter().enumerate() {
                value += coordinate
                    * model.polynomial
                        [(dimension + 1) * model.output_dimensions + output_dimension];
            }
            for node in 0..model.node_count {
                let squared = (0..model.input_dimensions)
                    .map(|dimension| {
                        let difference = normalized[dimension]
                            - model.nodes[node * model.input_dimensions + dimension];
                        difference * difference
                    })
                    .sum::<f64>();
                value += squared.sqrt().powi(3)
                    * model.weights[node * model.output_dimensions + output_dimension];
            }
            output[row * model.output_dimensions + output_dimension] = value;
        }
    }
    checked_finite("RBF result", &output)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn fit_sigma_field(
    surface: &RbfModel,
    covariances: &[f64],
    coordinates: &[f64],
    embedded_coordinates: &[f64],
    intrinsic_dimensions: usize,
    periodic_dimensions: usize,
    smoothing: Option<f64>,
    floor_fraction: f64,
    plan: Option<&RbfFitPlan>,
) -> KernelResult<SigmaFieldFit> {
    let node_count = surface.node_count;
    let components = surface.output_dimensions;
    let embedded_dimensions = surface.input_dimensions;
    let covariance_columns = components
        .checked_mul(components)
        .ok_or_else(|| invalid("sigma-field covariance dimensions overflow usize"))?;
    checked_matrix(
        "sigma-field covariances",
        covariances,
        node_count,
        covariance_columns,
    )?;
    checked_matrix(
        "sigma-field coordinates",
        coordinates,
        node_count,
        intrinsic_dimensions,
    )?;
    checked_matrix(
        "sigma-field embedded coordinates",
        embedded_coordinates,
        node_count,
        embedded_dimensions,
    )?;
    if intrinsic_dimensions == 0 || intrinsic_dimensions > components {
        return Err(invalid(
            "sigma-field intrinsic dimensions must fit the reduced subspace",
        ));
    }
    if periodic_dimensions == 0 {
        if embedded_dimensions != intrinsic_dimensions {
            return Err(invalid(
                "non-periodic sigma-field coordinates must use an identity embedding",
            ));
        }
    } else if periodic_dimensions != intrinsic_dimensions
        || embedded_dimensions
            != intrinsic_dimensions
                .checked_mul(2)
                .ok_or_else(|| invalid("sigma-field periodic dimensions overflow usize"))?
    {
        return Err(invalid(
            "periodic sigma-field coordinates must use paired cosine/sine dimensions",
        ));
    }
    if !floor_fraction.is_finite() || floor_fraction <= 0.0 {
        return Err(invalid(
            "sigma-field floor fraction must be finite and positive",
        ));
    }
    if smoothing.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err(invalid(
            "sigma-field smoothing must be finite and non-negative",
        ));
    }
    if let Some(plan) = plan {
        if plan.node_count() != node_count || plan.input_dimensions() != embedded_dimensions {
            return Err(invalid(
                "sigma-field RBF plan does not match the selected topology",
            ));
        }
    }
    evaluate_rbf(surface, embedded_coordinates, node_count)?;

    let mut variances = vec![0.0; node_count];
    for node in 0..node_count {
        let query =
            &embedded_coordinates[node * embedded_dimensions..(node + 1) * embedded_dimensions];
        let mut normalized = vec![0.0; embedded_dimensions];
        for dimension in 0..embedded_dimensions {
            normalized[dimension] = (query[dimension] - surface.coordinate_offset[dimension])
                / surface.coordinate_scale[dimension];
        }
        let mut embedded_jacobian = vec![0.0; components * embedded_dimensions];
        for output in 0..components {
            for dimension in 0..embedded_dimensions {
                let mut derivative = surface.polynomial[(dimension + 1) * components + output];
                for source in 0..node_count {
                    let mut squared = 0.0;
                    for (axis, normalized_value) in normalized.iter().copied().enumerate() {
                        let difference =
                            normalized_value - surface.nodes[source * embedded_dimensions + axis];
                        squared += difference * difference;
                    }
                    let radius = squared.sqrt();
                    derivative += 3.0
                        * radius
                        * (normalized[dimension]
                            - surface.nodes[source * embedded_dimensions + dimension])
                        * surface.weights[source * components + output];
                }
                embedded_jacobian[output * embedded_dimensions + dimension] =
                    derivative / surface.coordinate_scale[dimension];
            }
        }

        let mut tangent = vec![0.0; components * intrinsic_dimensions];
        if periodic_dimensions == 0 {
            tangent.copy_from_slice(&embedded_jacobian);
        } else {
            for output in 0..components {
                for dimension in 0..intrinsic_dimensions {
                    let angle = coordinates[node * intrinsic_dimensions + dimension];
                    tangent[output * intrinsic_dimensions + dimension] = -angle.sin()
                        * embedded_jacobian[output * embedded_dimensions + 2 * dimension]
                        + angle.cos()
                            * embedded_jacobian[output * embedded_dimensions + 2 * dimension + 1];
                }
            }
        }

        let covariance = &covariances[node * covariance_columns..(node + 1) * covariance_columns];
        variances[node] =
            off_surface_variance(covariance, &tangent, components, intrinsic_dimensions)?;
    }

    let mut ordered = variances.clone();
    ordered.sort_by(f64::total_cmp);
    let median = ordered[(node_count - 1) / 2].max(1e-12);
    let floor = floor_fraction * median;
    let sigma: Vec<f64> = variances
        .into_iter()
        .map(|variance| fp32(variance.max(floor).sqrt()))
        .collect();
    checked_finite("sigma-field standard deviations", &sigma)?;
    let log_sigma: Vec<f64> = sigma.iter().map(|value| fp32(value.ln())).collect();
    let model = match (plan, smoothing) {
        (Some(plan), Some(value)) => plan.fit_smoothed(&log_sigma, 1, value)?,
        (Some(plan), None) => plan.fit_auto_smoothed(&log_sigma, 1)?,
        (None, Some(value)) => fit_rbf_smoothed(
            embedded_coordinates,
            node_count,
            embedded_dimensions,
            &log_sigma,
            1,
            value,
        )?,
        (None, None) => fit_rbf_auto_smoothed(
            embedded_coordinates,
            node_count,
            embedded_dimensions,
            &log_sigma,
            1,
        )?,
    };
    let sigma_mean = sigma.iter().sum::<f64>() / node_count as f64;
    let sigma_min = sigma.iter().copied().fold(f64::INFINITY, f64::min);
    let sigma_max = sigma.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(SigmaFieldFit {
        model,
        sigma_mean,
        sigma_min,
        sigma_max,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn invert_rbf_origin(
    surface: &RbfModel,
    coordinates: &[f64],
    embedded_coordinates: &[f64],
    intrinsic_dimensions: usize,
    periodic_dimensions: usize,
    max_iterations: usize,
    restart_count: usize,
    damping: f64,
) -> KernelResult<RbfOriginFit> {
    let node_count = surface.node_count;
    if max_iterations == 0 || restart_count == 0 {
        return Err(invalid(
            "RBF origin inversion needs positive iterations and restarts",
        ));
    }
    if !damping.is_finite() || damping < 0.0 {
        return Err(invalid(
            "RBF origin inversion damping must be finite and non-negative",
        ));
    }
    validate_authoring_embedding(
        coordinates,
        embedded_coordinates,
        node_count,
        intrinsic_dimensions,
        periodic_dimensions,
        surface.input_dimensions,
    )?;
    let node_values = evaluate_rbf(surface, embedded_coordinates, node_count)?;
    let mut seeds: Vec<usize> = (0..node_count).collect();
    seeds.sort_by(|left, right| {
        let norm = |node: usize| {
            node_values[node * surface.output_dimensions..(node + 1) * surface.output_dimensions]
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
        };
        norm(*left)
            .total_cmp(&norm(*right))
            .then_with(|| left.cmp(right))
    });

    let mut best_coordinates = Vec::new();
    let mut best_distance = f64::INFINITY;
    for seed in seeds.into_iter().take(restart_count.min(node_count)) {
        let mut position =
            coordinates[seed * intrinsic_dimensions..(seed + 1) * intrinsic_dimensions].to_vec();
        for _ in 0..max_iterations {
            let embedded = embed_authoring_point(&position, periodic_dimensions);
            let residual = evaluate_rbf(surface, &embedded, 1)?;
            let embedded_jacobian = rbf_raw_jacobian(surface, &embedded)?;
            let tangent = authoring_jacobian(
                &embedded_jacobian,
                surface.output_dimensions,
                surface.input_dimensions,
                &position,
                periodic_dimensions,
            );
            let mut normal = vec![0.0; intrinsic_dimensions * intrinsic_dimensions];
            let mut gradient = vec![0.0; intrinsic_dimensions];
            for left in 0..intrinsic_dimensions {
                gradient[left] = (0..surface.output_dimensions)
                    .map(|output| tangent[output * intrinsic_dimensions + left] * residual[output])
                    .sum();
                for right in 0..intrinsic_dimensions {
                    normal[left * intrinsic_dimensions + right] = (0..surface.output_dimensions)
                        .map(|output| {
                            tangent[output * intrinsic_dimensions + left]
                                * tangent[output * intrinsic_dimensions + right]
                        })
                        .sum();
                }
                let diagonal = normal[left * intrinsic_dimensions + left];
                normal[left * intrinsic_dimensions + left] += damping * diagonal.max(1e-9) + 1e-9;
            }
            let step = solve(normal, intrinsic_dimensions, gradient, 1)?;
            for dimension in 0..intrinsic_dimensions {
                position[dimension] -= step[dimension];
                if periodic_dimensions > 0 {
                    position[dimension] =
                        position[dimension].rem_euclid(2.0 * std::f64::consts::PI);
                }
            }
        }
        let embedded = embed_authoring_point(&position, periodic_dimensions);
        let distance = evaluate_rbf(surface, &embedded, 1)?
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if distance < best_distance {
            best_distance = distance;
            best_coordinates = position;
        }
    }
    checked_vector(
        "RBF origin coordinates",
        &best_coordinates,
        intrinsic_dimensions,
    )?;
    if !best_distance.is_finite() {
        return Err(invalid("RBF origin distance is not finite"));
    }
    Ok(RbfOriginFit {
        coordinates: best_coordinates,
        distance: best_distance,
    })
}

fn validate_authoring_embedding(
    coordinates: &[f64],
    embedded_coordinates: &[f64],
    node_count: usize,
    intrinsic_dimensions: usize,
    periodic_dimensions: usize,
    embedded_dimensions: usize,
) -> KernelResult<()> {
    checked_matrix(
        "authoring coordinates",
        coordinates,
        node_count,
        intrinsic_dimensions,
    )?;
    checked_matrix(
        "embedded authoring coordinates",
        embedded_coordinates,
        node_count,
        embedded_dimensions,
    )?;
    if periodic_dimensions == 0 && embedded_dimensions != intrinsic_dimensions {
        return Err(invalid(
            "non-periodic authoring coordinates must use an identity embedding",
        ));
    }
    if periodic_dimensions > 0
        && (periodic_dimensions != intrinsic_dimensions
            || embedded_dimensions
                != intrinsic_dimensions
                    .checked_mul(2)
                    .ok_or_else(|| invalid("periodic authoring dimensions overflow usize"))?)
    {
        return Err(invalid(
            "periodic authoring coordinates must use paired cosine/sine dimensions",
        ));
    }
    Ok(())
}

fn embed_authoring_point(coordinates: &[f64], periodic_dimensions: usize) -> Vec<f64> {
    if periodic_dimensions == 0 {
        return coordinates.to_vec();
    }
    coordinates
        .iter()
        .flat_map(|coordinate| [fp32(coordinate.cos()), fp32(coordinate.sin())])
        .collect()
}

fn rbf_raw_jacobian(surface: &RbfModel, query: &[f64]) -> KernelResult<Vec<f64>> {
    checked_vector("RBF Jacobian query", query, surface.input_dimensions)?;
    let mut normalized = vec![0.0; surface.input_dimensions];
    for dimension in 0..surface.input_dimensions {
        normalized[dimension] = (query[dimension] - surface.coordinate_offset[dimension])
            / surface.coordinate_scale[dimension];
    }
    let mut jacobian = vec![0.0; surface.output_dimensions * surface.input_dimensions];
    for output in 0..surface.output_dimensions {
        for dimension in 0..surface.input_dimensions {
            let mut derivative =
                surface.polynomial[(dimension + 1) * surface.output_dimensions + output];
            for source in 0..surface.node_count {
                let radius = normalized
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(axis, value)| {
                        let difference =
                            value - surface.nodes[source * surface.input_dimensions + axis];
                        difference * difference
                    })
                    .sum::<f64>()
                    .sqrt();
                derivative += 3.0
                    * radius
                    * (normalized[dimension]
                        - surface.nodes[source * surface.input_dimensions + dimension])
                    * surface.weights[source * surface.output_dimensions + output];
            }
            jacobian[output * surface.input_dimensions + dimension] =
                derivative / surface.coordinate_scale[dimension];
        }
    }
    checked_finite("RBF Jacobian", &jacobian)?;
    Ok(jacobian)
}

fn authoring_jacobian(
    embedded_jacobian: &[f64],
    output_dimensions: usize,
    embedded_dimensions: usize,
    coordinates: &[f64],
    periodic_dimensions: usize,
) -> Vec<f64> {
    if periodic_dimensions == 0 {
        return embedded_jacobian.to_vec();
    }
    let intrinsic_dimensions = coordinates.len();
    let mut tangent = vec![0.0; output_dimensions * intrinsic_dimensions];
    for output in 0..output_dimensions {
        for dimension in 0..intrinsic_dimensions {
            tangent[output * intrinsic_dimensions + dimension] = -coordinates[dimension].sin()
                * embedded_jacobian[output * embedded_dimensions + 2 * dimension]
                + coordinates[dimension].cos()
                    * embedded_jacobian[output * embedded_dimensions + 2 * dimension + 1];
        }
    }
    tangent
}

fn off_surface_variance(
    covariance: &[f64],
    tangent: &[f64],
    components: usize,
    intrinsic_dimensions: usize,
) -> KernelResult<f64> {
    let mut tangent_gram = vec![0.0; intrinsic_dimensions * intrinsic_dimensions];
    let mut covariance_tangent = vec![0.0; components * intrinsic_dimensions];
    for row in 0..components {
        for dimension in 0..intrinsic_dimensions {
            covariance_tangent[row * intrinsic_dimensions + dimension] = (0..components)
                .map(|column| {
                    covariance[row * components + column]
                        * tangent[column * intrinsic_dimensions + dimension]
                })
                .sum();
        }
    }
    for left in 0..intrinsic_dimensions {
        for right in left..intrinsic_dimensions {
            let value = (0..components)
                .map(|row| {
                    tangent[row * intrinsic_dimensions + left]
                        * tangent[row * intrinsic_dimensions + right]
                })
                .sum();
            tangent_gram[left * intrinsic_dimensions + right] = value;
            tangent_gram[right * intrinsic_dimensions + left] = value;
        }
    }
    let (eigenvalues, eigenvectors) = symmetric_eigen(&tangent_gram, intrinsic_dimensions)?;
    let largest_singular = eigenvalues[0].max(0.0).sqrt();
    let singular_tolerance =
        largest_singular * components.max(intrinsic_dimensions) as f64 * f64::EPSILON;
    let mut tangent_trace = 0.0;
    let mut rank = 0usize;
    for (component, eigenvalue) in eigenvalues.iter().copied().enumerate() {
        if eigenvalue.max(0.0).sqrt() <= singular_tolerance {
            continue;
        }
        rank += 1;
        let mut projected = 0.0;
        for row in 0..components {
            let tangent_vector = (0..intrinsic_dimensions)
                .map(|dimension| {
                    tangent[row * intrinsic_dimensions + dimension]
                        * eigenvectors[dimension * intrinsic_dimensions + component]
                })
                .sum::<f64>();
            let covariance_vector = (0..intrinsic_dimensions)
                .map(|dimension| {
                    covariance_tangent[row * intrinsic_dimensions + dimension]
                        * eigenvectors[dimension * intrinsic_dimensions + component]
                })
                .sum::<f64>();
            projected += tangent_vector * covariance_vector;
        }
        tangent_trace += projected / eigenvalue;
    }
    let total_trace: f64 = (0..components)
        .map(|index| covariance[index * components + index])
        .sum();
    let normal_dimensions = components - rank;
    let variance = if normal_dimensions > 0 {
        (total_trace - tangent_trace) / normal_dimensions as f64
    } else {
        total_trace / components as f64
    };
    if !variance.is_finite() {
        return Err(invalid("sigma-field variance is not finite"));
    }
    Ok(variance.max(0.0))
}

pub fn normalize_template_scores(
    sum_log_probabilities: &[f64],
    token_counts: &[u32],
) -> KernelResult<TemplateScoreProbabilities> {
    if sum_log_probabilities.is_empty() {
        return Err(invalid("template scores must not be empty"));
    }
    checked_finite("sum log probabilities", sum_log_probabilities)?;
    if token_counts.len() != sum_log_probabilities.len() {
        return Err(invalid(format!(
            "token_counts has {} values, expected {}",
            token_counts.len(),
            sum_log_probabilities.len()
        )));
    }
    checked_output("template-score result", &[sum_log_probabilities.len()])?;
    let mean_log_probabilities: Vec<f64> = sum_log_probabilities
        .iter()
        .zip(token_counts)
        .map(|(score, count)| {
            if *count == 0 {
                *score
            } else {
                score / f64::from(*count)
            }
        })
        .collect();
    checked_finite("mean log probabilities", &mean_log_probabilities)?;
    let excluded: Vec<bool> = token_counts.iter().map(|count| *count == 0).collect();
    let result = TemplateScoreProbabilities {
        sum_probabilities: restricted_softmax(sum_log_probabilities, &excluded)?,
        mean_probabilities: restricted_softmax(&mean_log_probabilities, &excluded)?,
        mean_log_probabilities,
    };
    checked_finite("sum probabilities", &result.sum_probabilities)?;
    checked_finite("mean probabilities", &result.mean_probabilities)?;
    Ok(result)
}

fn restricted_softmax(values: &[f64], excluded: &[bool]) -> KernelResult<Vec<f64>> {
    checked_finite("softmax values", values)?;
    if excluded.len() != values.len() {
        return Err(invalid("softmax exclusion mask length mismatch"));
    }
    let maximum = values
        .iter()
        .zip(excluded)
        .filter_map(|(value, excluded)| (!excluded).then_some(*value))
        .fold(f64::NEG_INFINITY, f64::max);
    if maximum == f64::NEG_INFINITY {
        return Ok(vec![0.0; values.len()]);
    }
    let mut output: Vec<f64> = values
        .iter()
        .zip(excluded)
        .map(|(value, excluded)| {
            if *excluded {
                0.0
            } else {
                (value - maximum).exp()
            }
        })
        .collect();
    let total = output.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(invalid("softmax normalization is not finite"));
    }
    for value in &mut output {
        *value /= total;
    }
    Ok(output)
}
