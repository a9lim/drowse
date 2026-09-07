mod error;
mod linalg;

pub mod kernel;

use kernel::{
    AffineFisherFit as NativeAffineFisherFit, Centering as NativeCentering,
    GroupedReducedCovarianceAccumulator as NativeGroupedReducedCovarianceAccumulator,
    KnnGraph as NativeKnnGraph, MahalanobisWhitener as NativeMahalanobisWhitener,
    NormalizedLaplacian as NativeNormalizedLaplacian, Pca as NativePca,
    PeriodicTopology as NativePeriodicTopology, RbfFitPlan as NativeRbfFitPlan,
    RbfModel as NativeRbfModel, RbfOriginFit as NativeRbfOriginFit,
    SigmaFieldFit as NativeSigmaFieldFit, SpectralEmbedding as NativeSpectralEmbedding,
    TemplateScoreProbabilities as NativeTemplateScoreProbabilities,
    TopologyDiagnostics as NativeTopologyDiagnostics, TopologySelection as NativeTopologySelection,
};
use wasm_bindgen::prelude::*;

fn js_error(error: error::KernelError) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[wasm_bindgen]
pub struct CenteringResult(NativeCentering);

#[wasm_bindgen]
impl CenteringResult {
    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.0.rows
    }

    #[wasm_bindgen(getter)]
    pub fn columns(&self) -> usize {
        self.0.columns
    }

    pub fn mean(&self) -> Vec<f64> {
        self.0.mean.clone()
    }

    pub fn centered(&self) -> Vec<f64> {
        self.0.centered.clone()
    }
}

#[wasm_bindgen(js_name = fitNeutralCentering)]
pub fn fit_neutral_centering(
    values: &[f64],
    rows: usize,
    columns: usize,
) -> Result<CenteringResult, JsValue> {
    kernel::fit_neutral_centering(values, rows, columns)
        .map(CenteringResult)
        .map_err(js_error)
}

#[wasm_bindgen(js_name = groupRowMeans)]
pub fn group_row_means(
    values: &[f64],
    rows: usize,
    columns: usize,
    offsets: &[u32],
) -> Result<Vec<f64>, JsValue> {
    kernel::group_row_means(values, rows, columns, offsets).map_err(js_error)
}

#[wasm_bindgen]
pub struct GroupedReducedCovarianceAccumulator(NativeGroupedReducedCovarianceAccumulator);

#[wasm_bindgen]
impl GroupedReducedCovarianceAccumulator {
    #[wasm_bindgen(constructor)]
    pub fn new(
        columns: usize,
        center: &[f64],
        basis: &[f64],
        components: usize,
        offsets: &[u32],
    ) -> Result<GroupedReducedCovarianceAccumulator, JsValue> {
        NativeGroupedReducedCovarianceAccumulator::new(columns, center, basis, components, offsets)
            .map(GroupedReducedCovarianceAccumulator)
            .map_err(js_error)
    }

    #[wasm_bindgen(getter, js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.0.node_count()
    }

    #[wasm_bindgen(getter)]
    pub fn components(&self) -> usize {
        self.0.components()
    }

    pub fn append(&mut self, values: &[f32], start_row: usize, rows: usize) -> Result<(), JsValue> {
        self.0.append(values, start_row, rows).map_err(js_error)
    }

    pub fn finalize(&self) -> Result<Vec<f64>, JsValue> {
        self.0.finalize().map_err(js_error)
    }
}

#[wasm_bindgen(js_name = applyCentering)]
pub fn apply_centering(
    values: &[f64],
    rows: usize,
    columns: usize,
    mean: &[f64],
) -> Result<Vec<f64>, JsValue> {
    kernel::apply_centering(values, rows, columns, mean).map_err(js_error)
}

#[wasm_bindgen]
pub struct MahalanobisWhitener(NativeMahalanobisWhitener);

#[wasm_bindgen]
impl MahalanobisWhitener {
    #[wasm_bindgen(constructor)]
    pub fn new(
        columns: usize,
        rank: usize,
        mean: &[f64],
        basis: &[f64],
        eigenvalues: &[f64],
        inverse_scales: &[f64],
        ridge: f64,
    ) -> MahalanobisWhitener {
        MahalanobisWhitener(NativeMahalanobisWhitener {
            columns,
            rank,
            mean: mean.to_vec(),
            basis: basis.to_vec(),
            eigenvalues: eigenvalues.to_vec(),
            inverse_scales: inverse_scales.to_vec(),
            ridge,
        })
    }

    #[wasm_bindgen(getter)]
    pub fn columns(&self) -> usize {
        self.0.columns
    }

    #[wasm_bindgen(getter)]
    pub fn rank(&self) -> usize {
        self.0.rank
    }

    #[wasm_bindgen(getter)]
    pub fn ridge(&self) -> f64 {
        self.0.ridge
    }

    pub fn mean(&self) -> Vec<f64> {
        self.0.mean.clone()
    }

    pub fn basis(&self) -> Vec<f64> {
        self.0.basis.clone()
    }

    pub fn eigenvalues(&self) -> Vec<f64> {
        self.0.eigenvalues.clone()
    }

    #[wasm_bindgen(js_name = inverseScales)]
    pub fn inverse_scales(&self) -> Vec<f64> {
        self.0.inverse_scales.clone()
    }

    pub fn transform(&self, values: &[f64], rows: usize) -> Result<Vec<f64>, JsValue> {
        kernel::apply_mahalanobis_whitener(&self.0, values, rows).map_err(js_error)
    }
}

#[wasm_bindgen(js_name = fitMahalanobisWhitener)]
pub fn fit_mahalanobis_whitener(
    values: &[f64],
    rows: usize,
    columns: usize,
    ridge_scale: f64,
) -> Result<MahalanobisWhitener, JsValue> {
    kernel::fit_mahalanobis_whitener(values, rows, columns, ridge_scale)
        .map(MahalanobisWhitener)
        .map_err(js_error)
}

#[wasm_bindgen]
pub struct AffineFisherFit(NativeAffineFisherFit);

#[wasm_bindgen]
impl AffineFisherFit {
    #[wasm_bindgen(getter, js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.0.node_count
    }
    #[wasm_bindgen(getter)]
    pub fn columns(&self) -> usize {
        self.0.columns
    }
    #[wasm_bindgen(getter)]
    pub fn components(&self) -> usize {
        self.0.components
    }
    #[wasm_bindgen(getter, js_name = explainedVariance)]
    pub fn explained_variance(&self) -> f64 {
        self.0.explained_variance
    }
    #[wasm_bindgen(getter, js_name = mahalanobisShare)]
    pub fn mahalanobis_share(&self) -> f64 {
        self.0.mahalanobis_share
    }
    #[wasm_bindgen(js_name = centroidMean)]
    pub fn centroid_mean(&self) -> Vec<f64> {
        self.0.centroid_mean.clone()
    }
    pub fn mean(&self) -> Vec<f64> {
        self.0.mean.clone()
    }
    pub fn basis(&self) -> Vec<f64> {
        self.0.basis.clone()
    }
    #[wasm_bindgen(js_name = nodeCoordinates)]
    pub fn node_coordinates(&self) -> Vec<f64> {
        self.0.node_coordinates.clone()
    }
    #[wasm_bindgen(js_name = muCoordinates)]
    pub fn mu_coordinates(&self) -> Vec<f64> {
        self.0.mu_coordinates.clone()
    }
    #[wasm_bindgen(js_name = whitenedGram)]
    pub fn whitened_gram(&self) -> Vec<f64> {
        self.0.whitened_gram.clone()
    }
    #[wasm_bindgen(js_name = neutralCrossGram)]
    pub fn neutral_cross_gram(&self) -> Vec<f64> {
        self.0.neutral_cross_gram.clone()
    }
}

#[wasm_bindgen(js_name = fitAffineFisher)]
pub fn fit_affine_fisher(
    centroids: &[f64],
    node_count: usize,
    columns: usize,
    whitener: &MahalanobisWhitener,
    max_components: usize,
    orient_to: usize,
) -> Result<AffineFisherFit, JsValue> {
    kernel::fit_affine_fisher(
        centroids,
        node_count,
        columns,
        &whitener.0,
        max_components,
        orient_to,
    )
    .map(AffineFisherFit)
    .map_err(js_error)
}

#[wasm_bindgen]
pub struct PcaResult(NativePca);

#[wasm_bindgen]
impl PcaResult {
    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.0.rows
    }

    #[wasm_bindgen(getter)]
    pub fn columns(&self) -> usize {
        self.0.columns
    }

    #[wasm_bindgen(getter)]
    pub fn components(&self) -> usize {
        self.0.components
    }

    pub fn mean(&self) -> Vec<f64> {
        self.0.mean.clone()
    }

    pub fn basis(&self) -> Vec<f64> {
        self.0.basis.clone()
    }

    pub fn eigenvalues(&self) -> Vec<f64> {
        self.0.eigenvalues.clone()
    }

    #[wasm_bindgen(js_name = explainedVariance)]
    pub fn explained_variance(&self) -> Vec<f64> {
        self.0.explained_variance.clone()
    }

    #[wasm_bindgen(js_name = cumulativeVariance)]
    pub fn cumulative_variance(&self) -> Vec<f64> {
        self.0.cumulative_variance.clone()
    }

    pub fn scores(&self) -> Vec<f64> {
        self.0.scores.clone()
    }
}

#[wasm_bindgen(js_name = fitPca)]
pub fn fit_pca(
    values: &[f64],
    rows: usize,
    columns: usize,
    max_components: usize,
    variance_threshold: f64,
) -> Result<PcaResult, JsValue> {
    kernel::fit_pca(values, rows, columns, max_components, variance_threshold)
        .map(PcaResult)
        .map_err(js_error)
}

#[wasm_bindgen(js_name = manifoldPosition)]
pub fn manifold_position(
    mean: &[f64],
    basis: &[f64],
    components: usize,
    coordinates: &[f64],
) -> Result<Vec<f64>, JsValue> {
    kernel::position(mean, basis, components, coordinates).map_err(js_error)
}

#[wasm_bindgen(js_name = euclideanProjectRows)]
pub fn project_rows(
    values: &[f64],
    rows: usize,
    mean: &[f64],
    basis: &[f64],
    components: usize,
) -> Result<Vec<f64>, JsValue> {
    kernel::project_rows(values, rows, mean, basis, components).map_err(js_error)
}

#[wasm_bindgen(js_name = euclideanAblateRows)]
pub fn ablate_rows(
    values: &[f64],
    rows: usize,
    mean: &[f64],
    basis: &[f64],
    components: usize,
    coefficients: &[f64],
) -> Result<Vec<f64>, JsValue> {
    kernel::ablate_rows(values, rows, mean, basis, components, coefficients).map_err(js_error)
}

#[wasm_bindgen(js_name = euclideanPearsonCorrelations)]
pub fn pearson_correlations(
    values: &[f64],
    rows: usize,
    columns: usize,
) -> Result<Vec<f64>, JsValue> {
    kernel::pearson_correlations(values, rows, columns).map_err(js_error)
}

#[wasm_bindgen(js_name = euclideanPairwiseDistances)]
pub fn pairwise_distances(
    values: &[f64],
    rows: usize,
    columns: usize,
) -> Result<Vec<f64>, JsValue> {
    kernel::pairwise_distances(values, rows, columns).map_err(js_error)
}

#[wasm_bindgen]
pub struct KnnGraph(NativeKnnGraph);

#[wasm_bindgen]
impl KnnGraph {
    #[wasm_bindgen(getter, js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.0.node_count
    }

    #[wasm_bindgen(getter, js_name = kNn)]
    pub fn k_nn(&self) -> usize {
        self.0.k_nn
    }

    #[wasm_bindgen(getter, js_name = componentCount)]
    pub fn component_count(&self) -> usize {
        self.0.component_count
    }

    pub fn mask(&self) -> Vec<u8> {
        self.0.mask.clone()
    }

    #[wasm_bindgen(js_name = neighborDistances)]
    pub fn neighbor_distances(&self) -> Vec<f64> {
        self.0.neighbor_distances.clone()
    }
}

#[wasm_bindgen(js_name = buildKnnGraph)]
pub fn build_knn_graph(
    distances: &[f64],
    node_count: usize,
    k_nn: usize,
) -> Result<KnnGraph, JsValue> {
    kernel::build_knn_graph(distances, node_count, k_nn)
        .map(KnnGraph)
        .map_err(js_error)
}

#[wasm_bindgen]
pub struct NormalizedLaplacian(NativeNormalizedLaplacian);

#[wasm_bindgen]
impl NormalizedLaplacian {
    #[wasm_bindgen(getter, js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.0.node_count
    }

    #[wasm_bindgen(getter, js_name = kNn)]
    pub fn k_nn(&self) -> usize {
        self.0.k_nn
    }

    #[wasm_bindgen(getter)]
    pub fn bandwidth(&self) -> f64 {
        self.0.bandwidth
    }

    pub fn mask(&self) -> Vec<u8> {
        self.0.mask.clone()
    }

    pub fn weights(&self) -> Vec<f64> {
        self.0.weights.clone()
    }

    pub fn degrees(&self) -> Vec<f64> {
        self.0.degrees.clone()
    }

    pub fn laplacian(&self) -> Vec<f64> {
        self.0.laplacian.clone()
    }

    #[wasm_bindgen(js_name = nontrivialEigenvalues)]
    pub fn nontrivial_eigenvalues(&self) -> Vec<f64> {
        self.0.nontrivial_eigenvalues.clone()
    }

    #[wasm_bindgen(js_name = nontrivialEigenvectors)]
    pub fn nontrivial_eigenvectors(&self) -> Vec<f64> {
        self.0.nontrivial_eigenvectors.clone()
    }
}

#[wasm_bindgen(js_name = buildNormalizedLaplacian)]
pub fn build_normalized_laplacian(
    gram: &[f64],
    node_count: usize,
    k_nn: Option<usize>,
    bandwidth: Option<f64>,
) -> Result<NormalizedLaplacian, JsValue> {
    kernel::build_normalized_laplacian(gram, node_count, k_nn, bandwidth)
        .map(NormalizedLaplacian)
        .map_err(js_error)
}

#[wasm_bindgen]
pub struct SpectralEmbedding(NativeSpectralEmbedding);

#[wasm_bindgen]
impl SpectralEmbedding {
    #[wasm_bindgen(getter, js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.0.node_count
    }

    #[wasm_bindgen(getter)]
    pub fn dimensions(&self) -> usize {
        self.0.dimensions
    }

    #[wasm_bindgen(getter, js_name = heuristicDimensions)]
    pub fn heuristic_dimensions(&self) -> usize {
        self.0.heuristic_dimensions
    }

    #[wasm_bindgen(getter, js_name = minDimensions)]
    pub fn min_dimensions(&self) -> Option<usize> {
        self.0.min_dimensions
    }

    #[wasm_bindgen(getter)]
    pub fn pinned(&self) -> bool {
        self.0.pinned
    }

    #[wasm_bindgen(getter, js_name = kNn)]
    pub fn k_nn(&self) -> usize {
        self.0.k_nn
    }

    #[wasm_bindgen(getter)]
    pub fn bandwidth(&self) -> f64 {
        self.0.bandwidth
    }

    #[wasm_bindgen(getter, js_name = gapMagnitude)]
    pub fn gap_magnitude(&self) -> f64 {
        self.0.gap_magnitude
    }

    pub fn eigenvalues(&self) -> Vec<f64> {
        self.0.eigenvalues.clone()
    }

    pub fn coordinates(&self) -> Vec<f64> {
        self.0.coordinates.clone()
    }
}

#[wasm_bindgen(js_name = deriveSpectralEmbedding)]
pub fn derive_spectral_embedding(
    gram: &[f64],
    node_count: usize,
    max_dimensions: usize,
    min_dimensions: Option<usize>,
    k_nn: Option<usize>,
    bandwidth: Option<f64>,
) -> Result<SpectralEmbedding, JsValue> {
    kernel::derive_spectral_embedding(
        gram,
        node_count,
        max_dimensions,
        min_dimensions,
        k_nn,
        bandwidth,
    )
    .map(SpectralEmbedding)
    .map_err(js_error)
}

#[wasm_bindgen]
pub struct PeriodicTopology(NativePeriodicTopology);

#[wasm_bindgen]
impl PeriodicTopology {
    #[wasm_bindgen(getter, js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.0.node_count
    }

    #[wasm_bindgen(getter)]
    pub fn dimensions(&self) -> usize {
        self.0.dimensions
    }

    #[wasm_bindgen(getter, js_name = persistentLoops)]
    pub fn persistent_loops(&self) -> usize {
        self.0.persistent_loops
    }

    #[wasm_bindgen(getter, js_name = usedFaintCycle)]
    pub fn used_faint_cycle(&self) -> bool {
        self.0.used_faint_cycle
    }

    pub fn angles(&self) -> Vec<f64> {
        self.0.angles.clone()
    }
}

#[wasm_bindgen(js_name = detectPeriodicTopology)]
pub fn detect_periodic_topology(
    gram: &[f64],
    node_count: usize,
    max_dimensions: usize,
    k_nn: Option<usize>,
    bandwidth: Option<f64>,
    persistence_fraction: f64,
) -> Result<Option<PeriodicTopology>, JsValue> {
    kernel::detect_periodic_topology(
        gram,
        node_count,
        max_dimensions,
        k_nn,
        bandwidth,
        persistence_fraction,
    )
    .map(|result| result.map(PeriodicTopology))
    .map_err(js_error)
}

#[wasm_bindgen]
pub struct TopologySelection(NativeTopologySelection);

#[wasm_bindgen]
impl TopologySelection {
    #[wasm_bindgen(getter, js_name = winnerName)]
    pub fn winner_name(&self) -> String {
        self.0.winner_name.clone()
    }

    #[wasm_bindgen(getter, js_name = fitMode)]
    pub fn fit_mode(&self) -> String {
        self.0.fit_mode.clone()
    }

    #[wasm_bindgen(getter, js_name = intrinsicDimensions)]
    pub fn intrinsic_dimensions(&self) -> usize {
        self.0.intrinsic_dimensions
    }

    #[wasm_bindgen(getter, js_name = periodicDimensions)]
    pub fn periodic_dimensions(&self) -> usize {
        self.0.periodic_dimensions
    }

    #[wasm_bindgen(getter, js_name = persistentLoops)]
    pub fn persistent_loops(&self) -> usize {
        self.0.persistent_loops
    }

    #[wasm_bindgen(getter, js_name = usedFaintCycle)]
    pub fn used_faint_cycle(&self) -> bool {
        self.0.used_faint_cycle
    }

    pub fn coordinates(&self) -> Vec<f64> {
        self.0.coordinates.clone()
    }

    #[wasm_bindgen(js_name = embeddedCoordinates)]
    pub fn embedded_coordinates(&self) -> Vec<f64> {
        self.0.embedded_coordinates.clone()
    }

    #[wasm_bindgen(getter, js_name = candidateCount)]
    pub fn candidate_count(&self) -> usize {
        self.0.candidates.len()
    }

    #[wasm_bindgen(getter, js_name = hasWinnerPlan)]
    pub fn has_winner_plan(&self) -> bool {
        self.0.winner_plan.is_some()
    }

    #[wasm_bindgen(js_name = winnerPlan)]
    pub fn winner_plan(&self) -> Option<RbfFitPlan> {
        self.0.winner_plan.clone().map(RbfFitPlan)
    }

    #[wasm_bindgen(getter, js_name = diagnosticsKind)]
    pub fn diagnostics_kind(&self) -> String {
        match self.0.diagnostics {
            NativeTopologyDiagnostics::Pca { .. } => "pca",
            NativeTopologyDiagnostics::Spectral { .. } => "spectral",
        }
        .to_string()
    }

    #[wasm_bindgen(getter, js_name = diagnosticsPickedDimensions)]
    pub fn diagnostics_picked_dimensions(&self) -> usize {
        match &self.0.diagnostics {
            NativeTopologyDiagnostics::Pca {
                picked_dimensions, ..
            }
            | NativeTopologyDiagnostics::Spectral {
                picked_dimensions, ..
            } => *picked_dimensions,
        }
    }

    #[wasm_bindgen(js_name = diagnosticsPerComponentVariance)]
    pub fn diagnostics_per_component_variance(&self) -> Vec<f64> {
        match &self.0.diagnostics {
            NativeTopologyDiagnostics::Pca {
                per_component_variance,
                ..
            } => per_component_variance.clone(),
            NativeTopologyDiagnostics::Spectral { .. } => Vec::new(),
        }
    }

    #[wasm_bindgen(js_name = diagnosticsCumulativeVariance)]
    pub fn diagnostics_cumulative_variance(&self) -> Vec<f64> {
        match &self.0.diagnostics {
            NativeTopologyDiagnostics::Pca {
                cumulative_variance,
                ..
            } => cumulative_variance.clone(),
            NativeTopologyDiagnostics::Spectral { .. } => Vec::new(),
        }
    }

    #[wasm_bindgen(getter, js_name = diagnosticsThreshold)]
    pub fn diagnostics_threshold(&self) -> Option<f64> {
        match self.0.diagnostics {
            NativeTopologyDiagnostics::Pca { threshold, .. } => Some(threshold),
            NativeTopologyDiagnostics::Spectral { .. } => None,
        }
    }

    #[wasm_bindgen(js_name = diagnosticsEigenvalues)]
    pub fn diagnostics_eigenvalues(&self) -> Vec<f64> {
        match &self.0.diagnostics {
            NativeTopologyDiagnostics::Spectral { eigenvalues, .. } => eigenvalues.clone(),
            NativeTopologyDiagnostics::Pca { .. } => Vec::new(),
        }
    }

    #[wasm_bindgen(getter, js_name = diagnosticsGapMagnitude)]
    pub fn diagnostics_gap_magnitude(&self) -> Option<f64> {
        match self.0.diagnostics {
            NativeTopologyDiagnostics::Spectral { gap_magnitude, .. } => Some(gap_magnitude),
            NativeTopologyDiagnostics::Pca { .. } => None,
        }
    }

    #[wasm_bindgen(getter, js_name = diagnosticsBandwidth)]
    pub fn diagnostics_bandwidth(&self) -> Option<f64> {
        match self.0.diagnostics {
            NativeTopologyDiagnostics::Spectral { bandwidth, .. } => Some(bandwidth),
            NativeTopologyDiagnostics::Pca { .. } => None,
        }
    }

    #[wasm_bindgen(getter, js_name = diagnosticsKNn)]
    pub fn diagnostics_k_nn(&self) -> Option<usize> {
        match self.0.diagnostics {
            NativeTopologyDiagnostics::Spectral { k_nn, .. } => Some(k_nn),
            NativeTopologyDiagnostics::Pca { .. } => None,
        }
    }

    #[wasm_bindgen(getter, js_name = diagnosticsHeuristicDimensions)]
    pub fn diagnostics_heuristic_dimensions(&self) -> Option<usize> {
        match self.0.diagnostics {
            NativeTopologyDiagnostics::Spectral {
                heuristic_dimensions,
                ..
            } => Some(heuristic_dimensions),
            NativeTopologyDiagnostics::Pca { .. } => None,
        }
    }

    #[wasm_bindgen(getter, js_name = diagnosticsMinDimensions)]
    pub fn diagnostics_min_dimensions(&self) -> Option<usize> {
        match self.0.diagnostics {
            NativeTopologyDiagnostics::Spectral { min_dimensions, .. } => min_dimensions,
            NativeTopologyDiagnostics::Pca { .. } => None,
        }
    }

    #[wasm_bindgen(getter, js_name = diagnosticsPinned)]
    pub fn diagnostics_pinned(&self) -> Option<bool> {
        match self.0.diagnostics {
            NativeTopologyDiagnostics::Spectral { pinned, .. } => Some(pinned),
            NativeTopologyDiagnostics::Pca { .. } => None,
        }
    }

    #[wasm_bindgen(js_name = fitWinnerAutoSmoothed)]
    pub fn fit_winner_auto_smoothed(
        &self,
        values: &[f64],
        output_dimensions: usize,
    ) -> Result<RbfModel, JsValue> {
        self.0
            .winner_plan
            .as_ref()
            .ok_or_else(|| JsValue::from_str("flat topology winners do not have an RBF fit plan"))?
            .fit_auto_smoothed(values, output_dimensions)
            .map(RbfModel)
            .map_err(js_error)
    }

    #[wasm_bindgen(js_name = candidateName)]
    pub fn candidate_name(&self, index: usize) -> Option<String> {
        self.0
            .candidates
            .get(index)
            .map(|candidate| candidate.name.clone())
    }

    #[wasm_bindgen(js_name = candidateFitMode)]
    pub fn candidate_fit_mode(&self, index: usize) -> Option<String> {
        self.0
            .candidates
            .get(index)
            .map(|candidate| candidate.fit_mode.clone())
    }

    #[wasm_bindgen(js_name = candidateDimensions)]
    pub fn candidate_dimensions(&self, index: usize) -> Option<usize> {
        self.0
            .candidates
            .get(index)
            .map(|candidate| candidate.intrinsic_dimensions)
    }

    #[wasm_bindgen(js_name = candidateScore)]
    pub fn candidate_score(&self, index: usize) -> Option<f64> {
        self.0
            .candidates
            .get(index)
            .map(|candidate| candidate.score)
    }

    #[wasm_bindgen(js_name = candidateViable)]
    pub fn candidate_viable(&self, index: usize) -> Option<bool> {
        self.0
            .candidates
            .get(index)
            .map(|candidate| candidate.viable)
    }

    #[wasm_bindgen(js_name = candidateReason)]
    pub fn candidate_reason(&self, index: usize) -> Option<String> {
        self.0
            .candidates
            .get(index)
            .map(|candidate| candidate.reason.clone())
    }
}

#[wasm_bindgen(js_name = selectTopologyFromTargets)]
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
) -> Result<TopologySelection, JsValue> {
    kernel::select_topology_from_targets(
        consensus_gram,
        node_count,
        targets,
        target_offsets,
        max_dimensions,
        variance_threshold,
        requested_fit_mode,
        min_dimensions,
        k_nn,
        bandwidth,
        persistence_fraction,
        smoothing,
    )
    .map(TopologySelection)
    .map_err(js_error)
}

#[wasm_bindgen]
pub struct RbfModel(NativeRbfModel);

#[wasm_bindgen]
pub struct RbfFitPlan(NativeRbfFitPlan);

#[wasm_bindgen]
impl RbfFitPlan {
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(constructor)]
    pub fn new(
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
    ) -> Result<RbfFitPlan, JsValue> {
        kernel::restore_rbf_fit_plan(
            input_dimensions,
            node_count,
            nodes,
            coordinate_offset,
            coordinate_scale,
            kernel,
            kernel_scale,
            lambdas,
            spectral_basis,
            residual_ratios,
            residual_traces,
        )
        .map(RbfFitPlan)
        .map_err(js_error)
    }

    #[wasm_bindgen(getter, js_name = inputDimensions)]
    pub fn input_dimensions(&self) -> usize {
        self.0.input_dimensions()
    }

    #[wasm_bindgen(getter, js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.0.node_count()
    }

    pub fn nodes(&self) -> Vec<f64> {
        self.0.nodes().to_vec()
    }

    #[wasm_bindgen(js_name = coordinateOffset)]
    pub fn coordinate_offset(&self) -> Vec<f64> {
        self.0.coordinate_offset().to_vec()
    }

    #[wasm_bindgen(js_name = coordinateScale)]
    pub fn coordinate_scale(&self) -> Vec<f64> {
        self.0.coordinate_scale().to_vec()
    }

    pub fn kernel(&self) -> Vec<f64> {
        self.0.kernel().to_vec()
    }

    #[wasm_bindgen(getter, js_name = kernelScale)]
    pub fn kernel_scale(&self) -> f64 {
        self.0.kernel_scale()
    }

    pub fn lambdas(&self) -> Vec<f64> {
        self.0.lambdas().to_vec()
    }

    #[wasm_bindgen(js_name = spectralBasis)]
    pub fn spectral_basis(&self) -> Vec<f64> {
        self.0.spectral_basis().to_vec()
    }

    #[wasm_bindgen(js_name = residualRatios)]
    pub fn residual_ratios(&self) -> Vec<f64> {
        self.0.residual_ratios().to_vec()
    }

    #[wasm_bindgen(js_name = residualTraces)]
    pub fn residual_traces(&self) -> Vec<f64> {
        self.0.residual_traces().to_vec()
    }

    #[wasm_bindgen(js_name = fitSmoothed)]
    pub fn fit_smoothed(
        &self,
        values: &[f64],
        output_dimensions: usize,
        smoothing: f64,
    ) -> Result<RbfModel, JsValue> {
        self.0
            .fit_smoothed(values, output_dimensions, smoothing)
            .map(RbfModel)
            .map_err(js_error)
    }

    #[wasm_bindgen(js_name = fitAutoSmoothed)]
    pub fn fit_auto_smoothed(
        &self,
        values: &[f64],
        output_dimensions: usize,
    ) -> Result<RbfModel, JsValue> {
        self.0
            .fit_auto_smoothed(values, output_dimensions)
            .map(RbfModel)
            .map_err(js_error)
    }
}

#[wasm_bindgen(js_name = prepareRbfFitPlan)]
pub fn prepare_rbf_fit_plan(
    nodes: &[f64],
    node_count: usize,
    input_dimensions: usize,
) -> Result<RbfFitPlan, JsValue> {
    kernel::prepare_rbf_fit_plan(nodes, node_count, input_dimensions)
        .map(RbfFitPlan)
        .map_err(js_error)
}

#[wasm_bindgen]
impl RbfModel {
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(constructor)]
    pub fn new(
        input_dimensions: usize,
        output_dimensions: usize,
        node_count: usize,
        nodes: &[f64],
        coordinate_offset: &[f64],
        coordinate_scale: &[f64],
        weights: &[f64],
        polynomial: &[f64],
        lambda: f64,
        effective_degrees_of_freedom: f64,
        gcv: f64,
    ) -> Result<RbfModel, JsValue> {
        let model = NativeRbfModel {
            input_dimensions,
            output_dimensions,
            node_count,
            nodes: nodes.to_vec(),
            coordinate_offset: coordinate_offset.to_vec(),
            coordinate_scale: coordinate_scale.to_vec(),
            weights: weights.to_vec(),
            polynomial: polynomial.to_vec(),
            lambda,
            effective_degrees_of_freedom,
            gcv,
        };
        kernel::evaluate_rbf(&model, coordinate_offset, 1).map_err(js_error)?;
        Ok(RbfModel(model))
    }

    #[wasm_bindgen(getter, js_name = inputDimensions)]
    pub fn input_dimensions(&self) -> usize {
        self.0.input_dimensions
    }

    #[wasm_bindgen(getter, js_name = outputDimensions)]
    pub fn output_dimensions(&self) -> usize {
        self.0.output_dimensions
    }

    #[wasm_bindgen(getter, js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.0.node_count
    }

    #[wasm_bindgen(getter)]
    pub fn lambda(&self) -> f64 {
        self.0.lambda
    }

    #[wasm_bindgen(getter, js_name = effectiveDegreesOfFreedom)]
    pub fn effective_degrees_of_freedom(&self) -> f64 {
        self.0.effective_degrees_of_freedom
    }

    #[wasm_bindgen(getter)]
    pub fn gcv(&self) -> f64 {
        self.0.gcv
    }

    pub fn nodes(&self) -> Vec<f64> {
        self.0.nodes.clone()
    }

    #[wasm_bindgen(js_name = coordinateOffset)]
    pub fn coordinate_offset(&self) -> Vec<f64> {
        self.0.coordinate_offset.clone()
    }

    #[wasm_bindgen(js_name = coordinateScale)]
    pub fn coordinate_scale(&self) -> Vec<f64> {
        self.0.coordinate_scale.clone()
    }

    pub fn weights(&self) -> Vec<f64> {
        self.0.weights.clone()
    }

    pub fn polynomial(&self) -> Vec<f64> {
        self.0.polynomial.clone()
    }

    pub fn evaluate(&self, queries: &[f64], rows: usize) -> Result<Vec<f64>, JsValue> {
        kernel::evaluate_rbf(&self.0, queries, rows).map_err(js_error)
    }
}

#[wasm_bindgen]
pub struct SigmaFieldFit(NativeSigmaFieldFit);

#[wasm_bindgen]
impl SigmaFieldFit {
    #[wasm_bindgen(getter, js_name = sigmaMean)]
    pub fn sigma_mean(&self) -> f64 {
        self.0.sigma_mean
    }

    #[wasm_bindgen(getter, js_name = sigmaMin)]
    pub fn sigma_min(&self) -> f64 {
        self.0.sigma_min
    }

    #[wasm_bindgen(getter, js_name = sigmaMax)]
    pub fn sigma_max(&self) -> f64 {
        self.0.sigma_max
    }

    pub fn model(&self) -> RbfModel {
        RbfModel(self.0.model.clone())
    }
}

#[wasm_bindgen]
pub struct RbfOriginFit(NativeRbfOriginFit);

#[wasm_bindgen]
impl RbfOriginFit {
    pub fn coordinates(&self) -> Vec<f64> {
        self.0.coordinates.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn distance(&self) -> f64 {
        self.0.distance
    }
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = invertRbfOrigin)]
pub fn invert_rbf_origin(
    surface: &RbfModel,
    coordinates: &[f64],
    embedded_coordinates: &[f64],
    intrinsic_dimensions: usize,
    periodic_dimensions: usize,
    max_iterations: usize,
    restart_count: usize,
    damping: f64,
) -> Result<RbfOriginFit, JsValue> {
    kernel::invert_rbf_origin(
        &surface.0,
        coordinates,
        embedded_coordinates,
        intrinsic_dimensions,
        periodic_dimensions,
        max_iterations,
        restart_count,
        damping,
    )
    .map(RbfOriginFit)
    .map_err(js_error)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = fitSigmaFieldSmoothed)]
pub fn fit_sigma_field_smoothed(
    surface: &RbfModel,
    covariances: &[f64],
    coordinates: &[f64],
    embedded_coordinates: &[f64],
    intrinsic_dimensions: usize,
    periodic_dimensions: usize,
    smoothing: f64,
    floor_fraction: f64,
    plan: Option<RbfFitPlan>,
) -> Result<SigmaFieldFit, JsValue> {
    kernel::fit_sigma_field(
        &surface.0,
        covariances,
        coordinates,
        embedded_coordinates,
        intrinsic_dimensions,
        periodic_dimensions,
        Some(smoothing),
        floor_fraction,
        plan.as_ref().map(|value| &value.0),
    )
    .map(SigmaFieldFit)
    .map_err(js_error)
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = fitSigmaFieldAutoSmoothed)]
pub fn fit_sigma_field_auto_smoothed(
    surface: &RbfModel,
    covariances: &[f64],
    coordinates: &[f64],
    embedded_coordinates: &[f64],
    intrinsic_dimensions: usize,
    periodic_dimensions: usize,
    floor_fraction: f64,
    plan: Option<RbfFitPlan>,
) -> Result<SigmaFieldFit, JsValue> {
    kernel::fit_sigma_field(
        &surface.0,
        covariances,
        coordinates,
        embedded_coordinates,
        intrinsic_dimensions,
        periodic_dimensions,
        None,
        floor_fraction,
        plan.as_ref().map(|value| &value.0),
    )
    .map(SigmaFieldFit)
    .map_err(js_error)
}

#[wasm_bindgen(js_name = fitRbfSmoothed)]
pub fn fit_rbf_smoothed(
    nodes: &[f64],
    node_count: usize,
    input_dimensions: usize,
    values: &[f64],
    output_dimensions: usize,
    smoothing: f64,
) -> Result<RbfModel, JsValue> {
    kernel::fit_rbf_smoothed(
        nodes,
        node_count,
        input_dimensions,
        values,
        output_dimensions,
        smoothing,
    )
    .map(RbfModel)
    .map_err(js_error)
}

#[wasm_bindgen(js_name = fitRbfAutoSmoothed)]
pub fn fit_rbf_auto_smoothed(
    nodes: &[f64],
    node_count: usize,
    input_dimensions: usize,
    values: &[f64],
    output_dimensions: usize,
) -> Result<RbfModel, JsValue> {
    kernel::fit_rbf_auto_smoothed(
        nodes,
        node_count,
        input_dimensions,
        values,
        output_dimensions,
    )
    .map(RbfModel)
    .map_err(js_error)
}

#[wasm_bindgen]
pub struct TemplateScoreProbabilities(NativeTemplateScoreProbabilities);

#[wasm_bindgen]
impl TemplateScoreProbabilities {
    #[wasm_bindgen(js_name = sumProbabilities)]
    pub fn sum_probabilities(&self) -> Vec<f64> {
        self.0.sum_probabilities.clone()
    }

    #[wasm_bindgen(js_name = meanLogProbabilities)]
    pub fn mean_log_probabilities(&self) -> Vec<f64> {
        self.0.mean_log_probabilities.clone()
    }

    #[wasm_bindgen(js_name = meanProbabilities)]
    pub fn mean_probabilities(&self) -> Vec<f64> {
        self.0.mean_probabilities.clone()
    }
}

#[wasm_bindgen(js_name = normalizeTemplateScores)]
pub fn normalize_template_scores(
    sum_log_probabilities: &[f64],
    token_counts: &[u32],
) -> Result<TemplateScoreProbabilities, JsValue> {
    kernel::normalize_template_scores(sum_log_probabilities, token_counts)
        .map(TemplateScoreProbabilities)
        .map_err(js_error)
}
