use drowse_fitting_wasm::kernel::{
    ablate_rows, apply_mahalanobis_whitener, build_knn_graph, build_normalized_laplacian,
    count_persistent_loops, derive_pca_layout, derive_spectral_embedding, detect_faint_cycle,
    detect_periodic_topology, evaluate_rbf, fit_affine_fisher, fit_mahalanobis_whitener,
    fit_neutral_centering, fit_pca, fit_rbf_auto_smoothed, fit_rbf_smoothed, fit_sigma_field,
    group_row_means, invert_rbf_origin, normalize_template_scores, pairwise_distances,
    pearson_correlations, position, prepare_rbf_fit_plan, project_rows, restore_rbf_fit_plan,
    select_topology_from_targets, GroupedReducedCovarianceAccumulator, MahalanobisWhitener,
    TopologyDiagnostics,
};

fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: {actual} != {expected} within {tolerance}"
        );
    }
}

fn fixture_spectral_gram() -> Vec<f64> {
    let points = [
        [0.0_f32, 0.0_f32],
        [0.8, 0.1],
        [1.7, 0.45],
        [2.4, 1.2],
        [2.7, 2.2],
        [2.1, 3.1],
        [1.0, 3.45],
        [-0.1, 2.8],
        [-0.55, 1.6],
    ];
    let mut gram = vec![0.0; points.len() * points.len()];
    for row in 0..points.len() {
        for column in 0..points.len() {
            gram[row * points.len() + column] =
                f64::from(points[row][0] * points[column][0] + points[row][1] * points[column][1]);
        }
    }
    gram
}

fn row_projection(values: &[f64], rows: usize, columns: usize) -> Vec<f64> {
    let mut projection = vec![0.0; rows * rows];
    for left in 0..rows {
        for right in 0..rows {
            projection[left * rows + right] = (0..columns)
                .map(|column| values[left * columns + column] * values[right * columns + column])
                .sum();
        }
    }
    projection
}

fn centered_gram(points: &[f64], rows: usize, columns: usize) -> Vec<f64> {
    let mut means = vec![0.0; columns];
    for row in points.chunks_exact(columns) {
        for (column, value) in row.iter().enumerate() {
            means[column] += value / rows as f64;
        }
    }
    let centered: Vec<f64> = points
        .chunks_exact(columns)
        .flat_map(|row| row.iter().zip(&means).map(|(value, mean)| value - mean))
        .collect();
    row_projection(&centered, rows, columns)
}

fn circle_points(count: usize, major: f64, minor: f64) -> Vec<f64> {
    (0..count)
        .flat_map(|index| {
            let angle = 2.0 * std::f64::consts::PI * index as f64 / count as f64;
            [major * angle.cos(), minor * angle.sin()]
        })
        .collect()
}

fn torus_points(side: usize) -> Vec<f64> {
    (0..side)
        .flat_map(|left| {
            (0..side).flat_map(move |right| {
                let a = 2.0 * std::f64::consts::PI * left as f64 / side as f64;
                let b = 2.0 * std::f64::consts::PI * right as f64 / side as f64;
                [a.cos(), a.sin(), b.cos(), b.sin()]
            })
        })
        .collect()
}

#[test]
fn centering_and_mahalanobis_whitening_match_python_metric() {
    let values = [1.0, 10.0, 3.0, 10.0, 5.0, 10.0];
    let centered = fit_neutral_centering(&values, 3, 2).unwrap();
    close(&centered.mean, &[3.0, 10.0], 1e-12);
    close(&centered.centered, &[-2.0, 0.0, 0.0, 0.0, 2.0, 0.0], 1e-12);

    let whitener = fit_mahalanobis_whitener(&values, 3, 2, 0.1).unwrap();
    assert_eq!(whitener.rank, 1);
    close(&whitener.basis, &[1.0, 0.0], 1e-12);
    close(&[whitener.ridge], &[0.13333333333333333], 1e-12);

    let probes = [4.0, 10.0, 3.0, 11.0];
    let transformed = apply_mahalanobis_whitener(&whitener, &probes, 2).unwrap();
    // Python: Sigma = X^T X / N + lambda I = diag(2.8, 2/15).
    // Squared transformed norms therefore equal v^T Sigma^-1 v.
    close(
        &[
            transformed[0].powi(2) + transformed[1].powi(2),
            transformed[2].powi(2) + transformed[3].powi(2),
        ],
        &[0.35714285714285715, 7.5],
        1e-10,
    );
}

#[test]
fn grouped_activation_means_preserve_node_order() {
    let values = [1.0, 10.0, 3.0, 14.0, 6.0, 20.0, 9.0, 23.0, 12.0, 26.0];
    close(
        &group_row_means(&values, 5, 2, &[0, 2, 5]).unwrap(),
        &[2.0, 12.0, 9.0, 23.0],
        1e-12,
    );
    assert!(group_row_means(&values, 5, 2, &[0, 2, 2, 5]).is_err());
    assert!(group_row_means(&values, 5, 2, &[1, 5]).is_err());
    assert!(group_row_means(&values, 5, 2, &[0, 4]).is_err());
}

#[test]
fn zero_variance_neutral_layer_matches_python_identity_fallback() {
    let values = [2.0, -3.0, 2.0, -3.0, 2.0, -3.0];
    let whitener = fit_mahalanobis_whitener(&values, 3, 2, 0.1).unwrap();
    assert_eq!(whitener.rank, 0);
    close(&[whitener.ridge], &[1.0], 0.0);
    close(
        &apply_mahalanobis_whitener(&whitener, &[3.0, -5.0], 1).unwrap(),
        &[1.0, -2.0],
        1e-12,
    );
}

#[test]
fn dense_mahalanobis_transform_matches_python_symmetric_inverse_sqrt() {
    let values = [1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 4.0, 1.0, 3.0, 3.0, 5.0, 2.0];
    let whitener = fit_mahalanobis_whitener(&values, 4, 3, 0.2).unwrap();
    let transformed =
        apply_mahalanobis_whitener(&whitener, &[2.0, 2.0, 2.0, 5.0, -1.0, 4.0], 2).unwrap();
    close(
        &transformed,
        &[
            -0.7905693054199219,
            -6.705522537231445e-8,
            0.7905690670013428,
            1.5355360507965088,
            -1.6268976926803589,
            1.5355350971221924,
        ],
        2e-6,
    );
}

#[test]
fn affine_fisher_fit_matches_python_golden() {
    let neutral = [
        1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 4.0, 1.0, 3.0, 3.0, 3.0, 2.0, 5.0, 2.0, 4.0,
    ];
    let centroids = [1.0, 0.0, 2.0, 3.0, 1.0, 1.0, 5.0, 4.0, 3.0, 2.0, 5.0, 6.0];
    let whitener = fit_mahalanobis_whitener(&neutral, 5, 3, 1.0).unwrap();
    let fit = fit_affine_fisher(&centroids, 4, 3, &whitener, 2, 0).unwrap();
    assert_eq!(fit.components, 2);
    close(&fit.centroid_mean, &[2.75, 2.5, 3.0], 2e-6);
    close(
        &fit.mean,
        &[1.2346981763839722, 2.5237584114074707, 0.07898344099521637],
        3e-6,
    );
    close(
        &fit.basis,
        &[
            0.43905341625213623,
            -0.5828602313995361,
            -0.6837441325187683,
            -0.6327729821205139,
            -0.7408590316772461,
            0.22522473335266113,
        ],
        3e-6,
    );
    close(
        &fit.node_coordinates,
        &[
            0.054469551891088486,
            2.450920343399048,
            1.033460259437561,
            0.21929070353507996,
            -1.2045018672943115,
            -2.818382978439331,
            -5.155755043029785,
            -0.9852488040924072,
        ],
        5e-6,
    );
    close(
        &[fit.explained_variance, fit.mahalanobis_share],
        &[0.9989494040944763, 3.9487561597350385],
        5e-6,
    );
    close(
        &fit.neutral_cross_gram,
        &[
            0.6783915758132935,
            1.4169645309448242,
            0.22981590032577515,
            -2.325171947479248,
        ],
        5e-6,
    );
}

#[test]
fn pca_recovers_axis_and_threshold() {
    let values = [-3.0, 0.1, -1.0, -0.1, 1.0, -0.1, 3.0, 0.1];
    let pca = fit_pca(&values, 4, 2, 2, 0.95).unwrap();
    assert_eq!(pca.components, 1);
    assert!(pca.basis[0] > 0.999);
    assert!(pca.basis[1].abs() < 0.01);
    close(&pca.mean, &[0.0, 0.0], 1e-12);
    close(&pca.scores, &[-3.0, -1.0, 1.0, 3.0], 0.01);
}

#[test]
fn pca_rank_threshold_matches_python_one_e_minus_six_singular_policy() {
    let epsilon = 5e-6;
    let values = [-1.0, 0.0, 1.0, 0.0, 0.0, -epsilon, 0.0, epsilon];
    let pca = fit_pca(&values, 4, 2, 2, 1.0).unwrap();
    assert_eq!(pca.components, 2);
}

#[test]
fn positioning_projection_and_ablation_share_one_frame() {
    let mean = [1.0, 2.0, 3.0];
    let basis = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    close(
        &position(&mean, &basis, 2, &[2.0, -1.0]).unwrap(),
        &[3.0, 1.0, 3.0],
        1e-12,
    );
    let rows = [3.0, 5.0, 9.0];
    close(
        &project_rows(&rows, 1, &mean, &basis, 2).unwrap(),
        &[3.0, 5.0, 3.0],
        1e-12,
    );
    close(
        &ablate_rows(&rows, 1, &mean, &basis, 2, &[1.0, 0.5]).unwrap(),
        &[1.0, 3.5, 9.0],
        1e-12,
    );
}

#[test]
fn correlations_and_pairwise_distances_match_closed_forms() {
    let values = [1.0, 6.0, 2.0, 4.0, 3.0, 2.0];
    close(
        &pearson_correlations(&values, 3, 2).unwrap(),
        &[1.0, -1.0, -1.0, 1.0],
        1e-12,
    );
    let points = [0.0, 0.0, 3.0, 4.0, 6.0, 8.0];
    close(
        &pairwise_distances(&points, 3, 2).unwrap(),
        &[0.0, 5.0, 10.0, 5.0, 0.0, 5.0, 10.0, 5.0, 0.0],
        1e-12,
    );
}

#[test]
fn knn_union_graph_matches_python_fixture_without_tie_assumptions() {
    let gram = fixture_spectral_gram();
    let node_count = 9;
    let mut distances = vec![0.0; node_count * node_count];
    for row in 0..node_count {
        for column in (row + 1)..node_count {
            let squared = (gram[row * node_count + row] + gram[column * node_count + column]
                - 2.0 * gram[row * node_count + column])
                .max(0.0);
            let distance = squared.sqrt();
            distances[row * node_count + column] = distance;
            distances[column * node_count + row] = distance;
        }
    }
    let graph = build_knn_graph(&distances, node_count, 2).unwrap();
    let python_mask = [
        0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0,
    ];
    assert_eq!(graph.mask, python_mask);
    assert_eq!(graph.component_count, 1);
    assert_eq!(graph.neighbor_distances.len(), 18);
    for row in 0..node_count {
        assert_eq!(graph.mask[row * node_count + row], 0);
        for column in 0..node_count {
            assert_eq!(
                graph.mask[row * node_count + column],
                graph.mask[column * node_count + row]
            );
        }
    }
}

#[test]
fn knn_equal_distance_ties_preserve_contract_invariants() {
    let distances = [
        0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 1.0, 0.0,
    ];
    let graph = build_knn_graph(&distances, 4, 1).unwrap();
    for row in 0..4 {
        assert_eq!(graph.mask[row * 4 + row], 0);
        assert!((0..4).any(|column| graph.mask[row * 4 + column] != 0));
        for column in 0..4 {
            assert_eq!(graph.mask[row * 4 + column], graph.mask[column * 4 + row]);
        }
    }
}

#[test]
fn normalized_laplacian_matches_python_weights_and_identities() {
    let node_count = 9;
    let result =
        build_normalized_laplacian(&fixture_spectral_gram(), node_count, Some(2), None).unwrap();
    close(&[result.bandwidth], &[1.0816653966903687], 2e-6);
    close(
        &result.nontrivial_eigenvalues,
        &[
            0.19641262292861938,
            0.25976502895355225,
            0.7411192655563354,
            0.9030561447143555,
            1.4500545263290405,
            1.5667873620986938,
            1.9285200834274292,
            1.9542850255966187,
        ],
        2e-5,
    );
    close(
        &result.degrees,
        &[
            1.0517253875732422,
            1.428788423538208,
            1.3090877532958984,
            1.2653898000717163,
            1.2341558933258057,
            1.1723707914352417,
            1.063594102859497,
            0.9933852553367615,
            0.7898916006088257,
        ],
        2e-6,
    );
    for row in 0..node_count {
        assert!((result.laplacian[row * node_count + row] - 1.0).abs() < 1e-12);
        let mut null_residual = 0.0;
        for column in 0..node_count {
            assert!(
                (result.laplacian[row * node_count + column]
                    - result.laplacian[column * node_count + row])
                    .abs()
                    < 1e-12
            );
            assert!(
                (result.weights[row * node_count + column]
                    - result.weights[column * node_count + row])
                    .abs()
                    < 1e-12
            );
            null_residual +=
                result.laplacian[row * node_count + column] * result.degrees[column].sqrt();
        }
        assert!(null_residual.abs() < 2e-6);
    }
}

#[test]
fn spectral_embedding_matches_python_in_sign_and_rotation_invariants() {
    let node_count = 9;
    let result =
        derive_spectral_embedding(&fixture_spectral_gram(), node_count, 4, None, Some(2), None)
            .unwrap();
    assert_eq!(result.dimensions, 2);
    assert_eq!(result.heuristic_dimensions, 2);
    assert!(!result.pinned);
    close(&[result.gap_magnitude], &[0.4813542366027832], 2e-5);

    let python_coordinates = [
        -0.32113906741142273,
        -0.3423033654689789,
        -0.4783514142036438,
        -0.17431148886680603,
        -0.37884247303009033,
        0.1680339276790619,
        -0.14051076769828796,
        0.4236901104450226,
        0.14896051585674286,
        0.45868465304374695,
        0.37734490633010864,
        0.25135794281959534,
        0.45018061995506287,
        -0.08924386650323868,
        0.35219335556030273,
        -0.3994462490081787,
        0.11620479822158813,
        -0.45122167468070984,
    ];
    close(
        &row_projection(&result.coordinates, node_count, result.dimensions),
        &row_projection(&python_coordinates, node_count, 2),
        2e-5,
    );
    close(
        &pairwise_distances(&result.coordinates, node_count, result.dimensions).unwrap(),
        &pairwise_distances(&python_coordinates, node_count, 2).unwrap(),
        2e-5,
    );

    let pinned = derive_spectral_embedding(
        &fixture_spectral_gram(),
        node_count,
        4,
        Some(3),
        Some(2),
        None,
    )
    .unwrap();
    assert_eq!(pinned.heuristic_dimensions, 2);
    assert_eq!(pinned.dimensions, 3);
    assert!(pinned.pinned);
}

#[test]
fn spectral_graph_validation_and_bounds_fail_closed() {
    assert!(build_knn_graph(&[0.0, 1.0, 2.0, 0.0], 2, 1).is_err());
    assert!(build_knn_graph(&[0.0, -1.0, -1.0, 0.0], 2, 1).is_err());
    assert!(build_normalized_laplacian(&[0.0; 9], 3, None, None).is_err());
    assert!(build_normalized_laplacian(&fixture_spectral_gram(), 9, Some(2), Some(0.0)).is_err());
    assert!(
        derive_spectral_embedding(&fixture_spectral_gram(), 9, 0, None, Some(2), None).is_err()
    );

    let disconnected_points = [0.0, 1.0, 10.0, 11.0];
    let mut gram = vec![0.0; 16];
    for row in 0..4 {
        for column in 0..4 {
            gram[row * 4 + column] = disconnected_points[row] * disconnected_points[column];
        }
    }
    let error = build_normalized_laplacian(&gram, 4, Some(1), None).unwrap_err();
    assert!(error.to_string().contains("2 connected components"));
}

#[test]
fn pca_layout_matches_python_gram_threshold() {
    let layout = derive_pca_layout(&fixture_spectral_gram(), 9, 4, 0.70).unwrap();
    assert_eq!(layout.dimensions, 1);
    close(
        &layout.explained_variance,
        &[
            0.8318197131156921,
            0.1681802123785019,
            3.736408871191088e-8,
            2.4974907475439068e-8,
        ],
        2e-6,
    );
    close(
        &layout.cumulative_variance,
        &[
            0.8318197131156921,
            0.9999999403953552,
            0.9999999403953552,
            1.0,
        ],
        2e-6,
    );
}

#[test]
fn persistent_h1_counts_python_circle_ellipse_and_torus_goldens() {
    for (points, rows, columns, expected) in [
        (circle_points(40, 1.0, 1.0), 40, 2, 1),
        (circle_points(40, 3.0, 0.7), 40, 2, 1),
        (torus_points(7), 49, 4, 2),
    ] {
        let distances = pairwise_distances(&points, rows, columns).unwrap();
        assert_eq!(
            count_persistent_loops(&distances, rows, 0.5, 6).unwrap(),
            expected
        );
    }

    let line: Vec<f64> = (0..30)
        .flat_map(|index| {
            let value = index as f64 / 29.0;
            [value, 2.0 * value]
        })
        .collect();
    let distances = pairwise_distances(&line, 30, 2).unwrap();
    assert_eq!(count_persistent_loops(&distances, 30, 0.5, 6).unwrap(), 0);
}

#[test]
fn faint_cycle_fallback_matches_python_acceptance_guards() {
    let ring = circle_points(7, 1.0, 1.0);
    let ring_distances = pairwise_distances(&ring, 7, 2).unwrap();
    assert!(detect_faint_cycle(&ring_distances, 7).unwrap().is_some());

    let line: Vec<f64> = (0..7).flat_map(|index| [index as f64 / 6.0, 0.0]).collect();
    let line_distances = pairwise_distances(&line, 7, 2).unwrap();
    assert!(detect_faint_cycle(&line_distances, 7).unwrap().is_none());

    let grid: Vec<f64> = (0..5)
        .flat_map(|row| (0..5).flat_map(move |column| [row as f64, column as f64]))
        .collect();
    let grid_distances = pairwise_distances(&grid, 25, 2).unwrap();
    assert!(detect_faint_cycle(&grid_distances, 25).unwrap().is_none());

    let clustered: Vec<f64> = (0..4)
        .flat_map(|cluster| {
            let center = 2.0 * std::f64::consts::PI * cluster as f64 / 4.0;
            (0..3).flat_map(move |sample| {
                let angle = center - 0.18 + 0.18 * sample as f64;
                [angle.cos(), angle.sin()]
            })
        })
        .collect();
    let clustered_distances = pairwise_distances(&clustered, 12, 2).unwrap();
    let angles = detect_faint_cycle(&clustered_distances, 12)
        .unwrap()
        .unwrap();
    let mut order: Vec<usize> = (0..12).collect();
    order.sort_by(|left, right| angles[*left].total_cmp(&angles[*right]));
    let transitions = (0..12)
        .filter(|index| order[*index] / 3 != order[(*index + 1) % 12] / 3)
        .count();
    assert_eq!(transitions, 4);
}

#[test]
fn dense_complete_rips_complex_cannot_manufacture_a_loop() {
    let mut points = Vec::new();
    for index in 0..109 {
        let phase = index as f64;
        points.extend([
            0.1 * phase.sin(),
            0.1 * (phase * 1.7).cos(),
            0.1 * (phase * 2.3).sin(),
        ]);
    }
    points.extend([50.0, 0.0, 0.0]);
    let distances = pairwise_distances(&points, 110, 3).unwrap();
    assert_eq!(count_persistent_loops(&distances, 110, 0.5, 8).unwrap(), 0);
}

#[test]
fn periodic_detector_reuses_spectral_eigenpairs_and_recovers_axes() {
    let circle = circle_points(40, 3.0, 0.7);
    let detected = detect_periodic_topology(&centered_gram(&circle, 40, 2), 40, 6, None, None, 0.5)
        .unwrap()
        .unwrap();
    assert_eq!(detected.dimensions, 1);
    assert_eq!(detected.persistent_loops, 1);
    assert!(!detected.used_faint_cycle);

    let torus = torus_points(7);
    let detected = detect_periodic_topology(&centered_gram(&torus, 49, 4), 49, 6, None, None, 0.5)
        .unwrap()
        .unwrap();
    assert_eq!(detected.dimensions, 2);
    assert_eq!(detected.persistent_loops, 2);
}

#[test]
fn automatic_topology_selection_matches_python_flat_curved_and_periodic_goldens() {
    let tiny = select_topology_from_targets(
        &[1.0, -1.0, -1.0, 1.0],
        2,
        &[-1.0, 1.0],
        &[0, 2],
        2,
        0.70,
        "auto",
        None,
        None,
        None,
        0.5,
        None,
    )
    .unwrap();
    let tiny_flat = tiny
        .candidates
        .iter()
        .find(|candidate| candidate.name == "flat-pca")
        .unwrap();
    let tiny_spectral = tiny
        .candidates
        .iter()
        .find(|candidate| candidate.name == "spectral")
        .unwrap();
    assert!(!tiny_spectral.viable);
    assert_eq!(
        tiny_spectral.intrinsic_dimensions,
        tiny_flat.intrinsic_dimensions
    );
    assert!(tiny_spectral.score.is_infinite());

    let grid: Vec<f64> = (0..6)
        .flat_map(|row| {
            (0..6).flat_map(move |column| {
                let x = row as f64 / 5.0;
                let y = column as f64 / 5.0;
                [x, y]
            })
        })
        .collect();
    let flat = select_topology_from_targets(
        &centered_gram(&grid, 36, 2),
        36,
        &grid,
        &[0, 72],
        6,
        0.70,
        "auto",
        None,
        None,
        None,
        0.5,
        None,
    )
    .unwrap();
    assert_eq!(flat.winner_name, "flat-pca");
    assert_eq!(flat.fit_mode, "pca");
    match &flat.diagnostics {
        TopologyDiagnostics::Pca {
            per_component_variance,
            cumulative_variance,
            picked_dimensions,
            threshold,
        } => {
            assert_eq!(*picked_dimensions, 2);
            close(
                per_component_variance,
                &[0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                1e-6,
            );
            close(cumulative_variance, &[0.5, 1.0, 1.0, 1.0, 1.0, 1.0], 1e-6);
            close(&[*threshold], &[0.70], 0.0);
        }
        TopologyDiagnostics::Spectral { .. } => {
            panic!("flat winner returned spectral diagnostics")
        }
    }
    let flat_spectral = flat
        .candidates
        .iter()
        .find(|candidate| candidate.name == "spectral")
        .unwrap();
    close(&[flat_spectral.score], &[0.017712760716676712], 5e-6);

    let curve: Vec<f64> = (0..40)
        .flat_map(|index| {
            let t = index as f64 / 39.0;
            [(5.0 * t).sin(), (7.0 * t).cos(), (11.0 * t).sin()]
        })
        .collect();
    let curved = select_topology_from_targets(
        &centered_gram(&curve, 40, 3),
        40,
        &curve,
        &[0, 120],
        6,
        0.70,
        "auto",
        None,
        None,
        None,
        0.5,
        None,
    )
    .unwrap();
    assert_eq!(curved.winner_name, "spectral");
    assert_eq!(curved.fit_mode, "spectral");
    match &curved.diagnostics {
        TopologyDiagnostics::Spectral {
            eigenvalues,
            picked_dimensions,
            gap_magnitude,
            bandwidth,
            k_nn,
            heuristic_dimensions,
            min_dimensions,
            pinned,
        } => {
            assert_eq!(*picked_dimensions, 3);
            assert_eq!(*heuristic_dimensions, 3);
            assert_eq!(*min_dimensions, Some(2));
            assert!(!pinned);
            assert_eq!(*k_nn, 5);
            close(&[*bandwidth], &[0.42068371176719666], 5e-6);
            close(&[*gap_magnitude], &[0.153346985578537], 5e-6);
            close(
                &eigenvalues[..3],
                &[0.01579860, 0.02887184, 0.06586390],
                5e-6,
            );
        }
        TopologyDiagnostics::Pca { .. } => panic!("curved winner returned PCA diagnostics"),
    }
    let fixed_curved = select_topology_from_targets(
        &centered_gram(&curve, 40, 3),
        40,
        &curve,
        &[0, 120],
        6,
        0.70,
        "auto",
        None,
        None,
        None,
        0.5,
        Some(0.25),
    )
    .unwrap();
    assert_eq!(fixed_curved.winner_name, "spectral");
    let fixed_spectral = fixed_curved
        .candidates
        .iter()
        .find(|candidate| candidate.name == "spectral")
        .unwrap();
    close(&[fixed_spectral.score], &[0.05604735452343903], 5e-5);
    let exact_curved = select_topology_from_targets(
        &centered_gram(&curve, 40, 3),
        40,
        &curve,
        &[0, 120],
        6,
        0.70,
        "auto",
        None,
        None,
        None,
        0.5,
        Some(0.0),
    )
    .unwrap();
    assert_eq!(exact_curved.winner_name, "flat-pca");
    assert!(exact_curved
        .candidates
        .iter()
        .find(|candidate| candidate.name == "spectral")
        .unwrap()
        .score
        .is_infinite());
    let forced_flat = select_topology_from_targets(
        &centered_gram(&curve, 40, 3),
        40,
        &curve,
        &[0, 120],
        6,
        0.70,
        "pca",
        None,
        None,
        None,
        0.5,
        None,
    )
    .unwrap();
    assert_eq!(forced_flat.winner_name, "flat-pca");
    assert_eq!(forced_flat.candidates.len(), 1);
    let forced_spectral = select_topology_from_targets(
        &centered_gram(&grid, 36, 2),
        36,
        &grid,
        &[0, 72],
        6,
        0.70,
        "spectral",
        Some(2),
        None,
        None,
        0.5,
        None,
    )
    .unwrap();
    assert_eq!(forced_spectral.winner_name, "spectral");
    assert!(forced_spectral.intrinsic_dimensions >= 2);
    let flat_candidate = curved
        .candidates
        .iter()
        .find(|candidate| candidate.name == "flat-pca")
        .unwrap();
    let spectral_candidate = curved
        .candidates
        .iter()
        .find(|candidate| candidate.name == "spectral")
        .unwrap();
    assert!(spectral_candidate.intrinsic_dimensions >= flat_candidate.intrinsic_dimensions);
    assert!(spectral_candidate.score < flat_candidate.score);
    close(&[flat_candidate.score], &[0.13052600945950518], 2e-6);
    close(&[spectral_candidate.score], &[0.00863344594836235], 5e-5);

    let mut two_layers = curve.clone();
    two_layers.extend_from_slice(&curve);
    let repeated = select_topology_from_targets(
        &centered_gram(&curve, 40, 3),
        40,
        &two_layers,
        &[0, 120, 240],
        6,
        0.70,
        "auto",
        None,
        None,
        None,
        0.5,
        None,
    )
    .unwrap();
    assert_eq!(repeated.winner_name, "spectral");
    for candidate in &curved.candidates {
        let doubled = repeated
            .candidates
            .iter()
            .find(|other| other.name == candidate.name)
            .unwrap();
        close(&[doubled.score], &[2.0 * candidate.score], 1e-10);
    }

    let circle = circle_points(40, 1.0, 1.0);
    let periodic = select_topology_from_targets(
        &centered_gram(&circle, 40, 2),
        40,
        &circle,
        &[0, 80],
        6,
        0.70,
        "auto",
        None,
        None,
        None,
        0.5,
        None,
    )
    .unwrap();
    assert_eq!(periodic.winner_name, "torus-T1");
    assert_eq!(periodic.periodic_dimensions, 1);
    assert_eq!(periodic.persistent_loops, 1);
    assert!(periodic.winner_plan.is_some());

    let scalar_curve: Vec<f64> = curve.chunks_exact(3).map(|row| row[0]).collect();
    let scalar = select_topology_from_targets(
        &centered_gram(&curve, 40, 3),
        40,
        &scalar_curve,
        &[0, 40],
        6,
        0.70,
        "auto",
        None,
        None,
        None,
        0.5,
        None,
    )
    .unwrap();
    let mut variable_targets = curve.clone();
    variable_targets.extend_from_slice(&scalar_curve);
    let variable = select_topology_from_targets(
        &centered_gram(&curve, 40, 3),
        40,
        &variable_targets,
        &[0, 120, 160],
        6,
        0.70,
        "auto",
        None,
        None,
        None,
        0.5,
        None,
    )
    .unwrap();
    for candidate in &variable.candidates {
        let full = curved
            .candidates
            .iter()
            .find(|other| other.name == candidate.name)
            .unwrap();
        let reduced = scalar
            .candidates
            .iter()
            .find(|other| other.name == candidate.name)
            .unwrap();
        close(&[candidate.score], &[full.score + reduced.score], 1e-9);
    }
}

#[test]
fn exact_rbf_interpolates_and_smoothing_reduces_effective_dof() {
    let nodes = [0.0, 1.0, 2.0, 3.0];
    let values = [0.0, 1.0, 0.0, 1.0];
    let exact = fit_rbf_smoothed(&nodes, 4, 1, &values, 1, 0.0).unwrap();
    close(&evaluate_rbf(&exact, &nodes, 4).unwrap(), &values, 1e-9);
    close(&[exact.effective_degrees_of_freedom], &[4.0], 1e-12);

    let smooth = fit_rbf_smoothed(&nodes, 4, 1, &values, 1, 1.0).unwrap();
    assert!(smooth.lambda > 0.0);
    assert!(smooth.effective_degrees_of_freedom < 4.0);
    assert!(smooth.effective_degrees_of_freedom >= 2.0 - 1e-9);
    let fitted = evaluate_rbf(&smooth, &nodes, 4).unwrap();
    assert!(fitted.iter().all(|value| value.is_finite()));
}

#[test]
fn automatic_rbf_smoothing_matches_python_gcv_grid() {
    let nodes = [
        0.0, 0.07692308, 0.15384616, 0.23076923, 0.30769232, 0.38461539, 0.46153846, 0.53846157,
        0.61538464, 0.69230771, 0.76923078, 0.84615386, 0.92307693, 1.0,
    ];
    let values = [
        -0.18829083,
        0.10635998,
        0.55808729,
        -0.00367613,
        0.81995118,
        -0.01007093,
        1.03011501,
        0.02681591,
        0.98796320,
        -0.17364544,
        0.86664152,
        0.17017899,
        0.46017402,
        -0.01779196,
        -0.10526392,
        0.11699985,
        -0.62534642,
        0.29348135,
        -0.80815363,
        0.19689432,
        -1.17229819,
        0.06007263,
        -0.80606663,
        0.22719169,
        -0.55798244,
        0.18251377,
        -0.21625561,
        0.18497016,
    ];
    let model = fit_rbf_auto_smoothed(&nodes, 14, 1, &values, 2).unwrap();
    let planned = prepare_rbf_fit_plan(&nodes, 14, 1)
        .unwrap()
        .fit_auto_smoothed(&values, 2)
        .unwrap();
    close(&planned.weights, &model.weights, 1e-10);
    close(&planned.polynomial, &model.polynomial, 1e-10);
    close(
        &[
            planned.lambda,
            planned.effective_degrees_of_freedom,
            planned.gcv,
        ],
        &[model.lambda, model.effective_degrees_of_freedom, model.gcv],
        1e-10,
    );
    close(&[model.lambda], &[0.003233351744711399], 5e-7);
    close(
        &[model.effective_degrees_of_freedom],
        &[6.501681327819824],
        5e-4,
    );
    close(&[model.gcv], &[0.037849556654691696], 5e-4);
    close(
        &model.weights,
        &[
            -33.68947983,
            5.92658138,
            33.25513840,
            -8.87477493,
            -2.09877753,
            0.86536556,
            4.78752232,
            18.71720123,
            -5.77924442,
            -43.62068939,
            18.90623474,
            45.04475784,
            14.15004826,
            -27.46852112,
            -5.66823435,
            -6.95730257,
            -21.66469765,
            29.77136612,
            20.63844109,
            2.67829061,
            -54.77775574,
            -31.24170685,
            16.71325302,
            16.84019661,
            6.51181412,
            -0.51300710,
            8.71574116,
            -1.16776085,
        ],
        5e-3,
    );
    close(
        &model.polynomial,
        &[-2.38837957, 0.11510619, 5.87030411, -0.41637132],
        5e-4,
    );
    assert!(model.lambda > 0.0);
    assert!(model.effective_degrees_of_freedom < 14.0);
}

#[test]
fn reusable_rbf_plan_matches_standalone_across_python_multi_output_goldens() {
    let nodes = [
        0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 0.5, 1.0, 1.0, 1.0,
    ];
    let values_a = [
        0.1, -0.2, 0.3, 0.7, 0.4, -0.1, 0.2, 1.1, 0.5, -0.3, 0.8, 0.9, 1.4, -0.7, 0.2, 0.6, 0.2,
        -1.0, 0.0, -0.4, 0.8, 1.2, 0.9, -0.6, -0.5, 0.3, 1.1,
    ];
    let values_b = [
        0.2, 0.9, -0.4, 0.1, 1.3, -0.2, 0.7, -0.8, 0.0, 1.4, -1.1, 0.5, 0.4, -0.3, 1.0, 0.2, -0.6,
        1.1,
    ];
    let plan = prepare_rbf_fit_plan(&nodes, 9, 2).unwrap();
    assert_eq!(plan.node_count(), 9);
    assert_eq!(plan.input_dimensions(), 2);
    let restored = restore_rbf_fit_plan(
        plan.input_dimensions(),
        plan.node_count(),
        plan.nodes(),
        plan.coordinate_offset(),
        plan.coordinate_scale(),
        plan.kernel(),
        plan.kernel_scale(),
        plan.lambdas(),
        plan.spectral_basis(),
        plan.residual_ratios(),
        plan.residual_traces(),
    )
    .unwrap();

    for (values, outputs, python_info) in [
        (
            values_a.as_slice(),
            3,
            [754.6005859375, 3.0023765563964844, 2.366482973098755],
        ),
        (
            values_b.as_slice(),
            2,
            [3.715564489364624, 3.421415328979492, 1.983034372329712],
        ),
    ] {
        let planned = plan.fit_auto_smoothed(values, outputs).unwrap();
        let restored_fit = restored.fit_auto_smoothed(values, outputs).unwrap();
        let standalone = fit_rbf_auto_smoothed(&nodes, 9, 2, values, outputs).unwrap();
        close(&planned.weights, &standalone.weights, 1e-10);
        close(&restored_fit.weights, &standalone.weights, 1e-10);
        close(&planned.polynomial, &standalone.polynomial, 1e-10);
        close(&restored_fit.polynomial, &standalone.polynomial, 1e-10);
        close(
            &[
                planned.lambda,
                planned.effective_degrees_of_freedom,
                planned.gcv,
            ],
            &python_info,
            5e-3,
        );
    }
}

#[test]
fn reusable_rbf_plan_keeps_smallest_lambda_on_gcv_ties() {
    let nodes = [0.0, 1.0, 2.0, 3.0];
    let values = [0.0; 8];
    let planned = prepare_rbf_fit_plan(&nodes, 4, 1)
        .unwrap()
        .fit_auto_smoothed(&values, 2)
        .unwrap();
    let smallest = fit_rbf_smoothed(&nodes, 4, 1, &values, 2, 1e-6).unwrap();
    close(&[planned.lambda], &[smallest.lambda], 1e-15);
    close(&[planned.gcv], &[0.0], 0.0);
}

#[test]
fn automatic_rbf_full_polynomial_space_matches_python_degenerate_gcv() {
    let model = fit_rbf_auto_smoothed(&[0.0, 1.0], 2, 1, &[2.0, 4.0], 1).unwrap();
    let planned = prepare_rbf_fit_plan(&[0.0, 1.0], 2, 1)
        .unwrap()
        .fit_auto_smoothed(&[2.0, 4.0], 1)
        .unwrap();
    assert!(model.lambda > 0.0);
    close(&[planned.lambda], &[model.lambda], 0.0);
    close(&planned.weights, &model.weights, 1e-12);
    close(&planned.polynomial, &model.polynomial, 1e-12);
    close(&[model.effective_degrees_of_freedom], &[2.0], 0.0);
    close(&[planned.effective_degrees_of_freedom], &[2.0], 0.0);
    assert!(model.gcv.is_infinite());
    assert!(planned.gcv.is_infinite());
    close(
        &evaluate_rbf(&model, &[0.0, 1.0], 2).unwrap(),
        &[2.0, 4.0],
        1e-9,
    );
}

#[test]
fn reusable_rbf_plan_rejects_invalid_geometry_values_and_bounds() {
    assert!(prepare_rbf_fit_plan(&[0.0, 1.0], 3, 1).is_err());
    assert!(prepare_rbf_fit_plan(&[0.0, 0.0, 1.0, 1.0], 2, 2).is_err());

    let collinear = [0.0, 0.0, 0.5, 0.5, 1.0, 1.0];
    assert!(prepare_rbf_fit_plan(&collinear, 3, 2).is_err());

    let oversized_nodes: Vec<f64> = (0..511).map(|index| index as f64).collect();
    let error = prepare_rbf_fit_plan(&oversized_nodes, 511, 1).unwrap_err();
    assert!(error.to_string().contains("compute ceiling"));

    let plan = prepare_rbf_fit_plan(&[0.0, 1.0, 2.0], 3, 1).unwrap();
    assert!(plan.fit_auto_smoothed(&[1.0, 2.0], 1).is_err());
    assert!(plan.fit_auto_smoothed(&[f64::MAX, 0.0, 1.0], 1).is_err());
    let error = plan.fit_auto_smoothed(&[], usize::MAX).unwrap_err();
    assert!(error.to_string().contains("shape overflows"));
}

#[test]
fn rbf_poisedness_uses_python_fp32_matrix_rank_policy() {
    let nearly_collinear = [0.0, 0.0, 0.25, 0.25, 0.5, 0.5000001, 0.75, 0.75, 1.0, 1.0];
    assert!(fit_rbf_smoothed(&nearly_collinear, 5, 2, &[0.0; 5], 1, 0.0).is_err());

    let poised = [0.0, 0.0, 0.25, 0.25, 0.5, 0.5001, 0.75, 0.75, 1.0, 1.0];
    assert!(fit_rbf_smoothed(&poised, 5, 2, &[0.0; 5], 1, 0.0).is_ok());
}

#[test]
fn template_score_normalization_keeps_sum_and_mean_views_separate() {
    let scores = normalize_template_scores(&[-2.0, -4.0], &[1, 4]).unwrap();
    assert!(scores.sum_probabilities[0] > scores.sum_probabilities[1]);
    assert!(scores.mean_probabilities[1] > scores.mean_probabilities[0]);
    close(&scores.mean_log_probabilities, &[-2.0, -1.0], 1e-12);
    close(&[scores.sum_probabilities.iter().sum()], &[1.0], 1e-12);
    close(&[scores.mean_probabilities.iter().sum()], &[1.0], 1e-12);
}

#[test]
fn zero_token_template_choices_match_python_degenerate_masking() {
    let scores = normalize_template_scores(&[0.0, -2.0, -4.0], &[0, 1, 4]).unwrap();
    close(&scores.mean_log_probabilities, &[0.0, -2.0, -1.0], 1e-12);
    close(
        &scores.sum_probabilities,
        &[0.0, 0.8807970779778823, 0.11920292202211755],
        1e-12,
    );
    close(
        &scores.mean_probabilities,
        &[0.0, 0.2689414213699951, 0.7310585786300049],
        1e-12,
    );

    let all_empty = normalize_template_scores(&[0.0, 0.0], &[0, 0]).unwrap();
    close(&all_empty.sum_probabilities, &[0.0, 0.0], 0.0);
    close(&all_empty.mean_probabilities, &[0.0, 0.0], 0.0);
}

#[test]
fn output_and_work_ceilings_fail_before_large_allocations() {
    let values = vec![0.0; 4097];
    let error = pairwise_distances(&values, 4097, 1).unwrap_err();
    assert!(error.to_string().contains("output ceiling"));
}

#[test]
fn grouped_reduced_covariance_streams_contiguous_activation_chunks() {
    let mut accumulator = GroupedReducedCovarianceAccumulator::new(
        2,
        &[10.0, -5.0],
        &[1.0, 0.0, 0.0, 1.0],
        2,
        &[0, 2, 5],
    )
    .unwrap();
    accumulator.append(&[10.0, -5.0, 12.0, -5.0], 0, 2).unwrap();
    accumulator
        .append(&[10.0, -4.0, 10.0, -2.0, 10.0, 0.0], 2, 3)
        .unwrap();
    close(
        &accumulator.finalize().unwrap(),
        &[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
        1e-12,
    );

    let mut incomplete =
        GroupedReducedCovarianceAccumulator::new(1, &[0.0], &[1.0], 1, &[0, 2]).unwrap();
    incomplete.append(&[1.0], 0, 1).unwrap();
    assert!(incomplete.finalize().is_err());
    assert!(incomplete.append(&[2.0], 0, 1).is_err());
}

#[test]
fn sigma_field_uses_only_off_surface_covariance() {
    let nodes = [-1.0, 0.0, 1.0];
    let surface = fit_rbf_smoothed(&nodes, 3, 1, &[-1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 0.0).unwrap();
    let covariances = [
        100.0, 0.0, 0.0, 1.0, 100.0, 0.0, 0.0, 4.0, 100.0, 0.0, 0.0, 9.0,
    ];
    let sigma = fit_sigma_field(
        &surface,
        &covariances,
        &nodes,
        &nodes,
        1,
        0,
        Some(0.0),
        1e-3,
        None,
    )
    .unwrap();
    close(
        &[sigma.sigma_mean, sigma.sigma_min, sigma.sigma_max],
        &[2.0, 1.0, 3.0],
        1e-6,
    );
    let log_sigma = evaluate_rbf(&sigma.model, &nodes, 3).unwrap();
    close(
        &log_sigma.into_iter().map(f64::exp).collect::<Vec<_>>(),
        &[1.0, 2.0, 3.0],
        1e-5,
    );
}

#[test]
fn curved_origin_inversion_finds_the_neutral_surface_foot() {
    let nodes = [-1.0, 0.0, 1.0];
    let surface = fit_rbf_smoothed(&nodes, 3, 1, &[-1.0, 1.0, 0.0, 1.0, 1.0, 1.0], 2, 0.0).unwrap();
    let origin = invert_rbf_origin(&surface, &nodes, &nodes, 1, 0, 12, 3, 1e-3).unwrap();
    close(&origin.coordinates, &[0.0], 1e-6);
    close(&[origin.distance], &[1.0], 1e-6);
}

#[test]
fn invalid_non_finite_and_shape_inputs_fail_closed() {
    assert!(fit_pca(&[0.0, f64::NAN], 1, 2, 1, 0.7).is_err());
    assert!(fit_neutral_centering(&[1.0, 2.0], 2, 2).is_err());
    assert!(pearson_correlations(&[1.0, 2.0, 1.0, 3.0], 2, 2).is_err());
    assert!(fit_rbf_smoothed(&[0.0, 0.0, 1.0, 0.0], 2, 2, &[1.0, 2.0], 1, 0.0).is_err());
    assert!(pairwise_distances(&[f64::MAX, -f64::MAX], 2, 1).is_err());

    let malformed = MahalanobisWhitener {
        columns: 2,
        rank: 1,
        mean: vec![0.0, 0.0],
        basis: vec![1.0],
        eigenvalues: vec![1.0],
        inverse_scales: vec![1.0],
        ridge: 1.0,
    };
    assert!(apply_mahalanobis_whitener(&malformed, &[1.0, 2.0], 1).is_err());
}
