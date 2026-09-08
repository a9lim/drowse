use crate::error::{invalid, KernelResult};

pub const MAX_BUFFER_ELEMENTS: usize = 16 * 1024 * 1024;
pub const MAX_WORK_ELEMENTS: usize = 32 * 1024 * 1024;
const MAX_CUBIC_DIMENSION: usize = 512;
const EIGEN_TOLERANCE: f64 = 1e-14;
const PIVOT_TOLERANCE: f64 = 1e-12;

fn checked_product(name: &str, dimensions: &[usize]) -> KernelResult<usize> {
    dimensions.iter().try_fold(1usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| invalid(format!("{name} shape overflows usize")))
    })
}

pub fn checked_output(name: &str, dimensions: &[usize]) -> KernelResult<usize> {
    let elements = checked_product(name, dimensions)?;
    if elements > MAX_BUFFER_ELEMENTS {
        return Err(invalid(format!(
            "{name} needs {elements} elements, above the {MAX_BUFFER_ELEMENTS}-element output ceiling"
        )));
    }
    Ok(elements)
}

pub fn checked_work(name: &str, allocations: &[usize]) -> KernelResult<()> {
    let elements = allocations.iter().try_fold(0usize, |total, allocation| {
        total
            .checked_add(*allocation)
            .ok_or_else(|| invalid(format!("{name} work size overflows usize")))
    })?;
    if elements > MAX_WORK_ELEMENTS {
        return Err(invalid(format!(
            "{name} needs {elements} elements of work, above the {MAX_WORK_ELEMENTS}-element work ceiling"
        )));
    }
    Ok(())
}

pub fn checked_matrix<'a>(
    name: &str,
    values: &'a [f64],
    rows: usize,
    columns: usize,
) -> KernelResult<&'a [f64]> {
    if rows == 0 || columns == 0 {
        return Err(invalid(format!(
            "{name} must have non-zero rows and columns"
        )));
    }
    let expected = checked_output(name, &[rows, columns])?;
    if values.len() != expected {
        return Err(invalid(format!(
            "{name} has {} values, expected {expected} for {rows}x{columns}",
            values.len()
        )));
    }
    checked_finite(name, values)?;
    Ok(values)
}

pub fn checked_vector<'a>(name: &str, values: &'a [f64], length: usize) -> KernelResult<&'a [f64]> {
    if length == 0 {
        return Err(invalid(format!("{name} must not be empty")));
    }
    if values.len() != length {
        return Err(invalid(format!(
            "{name} has {} values, expected {length}",
            values.len()
        )));
    }
    checked_output(name, &[length])?;
    checked_finite(name, values)?;
    Ok(values)
}

pub fn checked_finite(name: &str, values: &[f64]) -> KernelResult<()> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(invalid(format!(
            "{name} contains a non-finite value at index {index}"
        )));
    }
    Ok(())
}

pub fn mean_rows(values: &[f64], rows: usize, columns: usize) -> Vec<f64> {
    let mut mean = vec![0.0; columns];
    for row in values.chunks_exact(columns) {
        for (column, value) in row.iter().enumerate() {
            mean[column] += value;
        }
    }
    let divisor = rows as f64;
    for value in &mut mean {
        *value /= divisor;
    }
    mean
}

pub fn center_rows(values: &[f64], columns: usize, mean: &[f64]) -> Vec<f64> {
    values
        .chunks_exact(columns)
        .flat_map(|row| row.iter().zip(mean).map(|(value, center)| value - center))
        .collect()
}

pub fn row_gram(
    values: &[f64],
    rows: usize,
    columns: usize,
    divisor: f64,
) -> KernelResult<Vec<f64>> {
    let output_elements = checked_output("row Gram", &[rows, rows])?;
    checked_work("row Gram", &[values.len(), output_elements])?;
    let mut gram = vec![0.0; output_elements];
    for left in 0..rows {
        for right in left..rows {
            let mut dot = 0.0;
            for column in 0..columns {
                dot += values[left * columns + column] * values[right * columns + column];
            }
            let value = dot / divisor;
            gram[left * rows + right] = value;
            gram[right * rows + left] = value;
        }
    }
    checked_finite("row Gram", &gram)?;
    Ok(gram)
}

pub fn symmetric_eigen(matrix: &[f64], size: usize) -> KernelResult<(Vec<f64>, Vec<f64>)> {
    checked_matrix("symmetric matrix", matrix, size, size)?;
    if size > MAX_CUBIC_DIMENSION {
        return Err(invalid(format!(
            "symmetric eigendecomposition dimension {size} exceeds the {MAX_CUBIC_DIMENSION}-dimension compute ceiling"
        )));
    }
    checked_work(
        "symmetric eigendecomposition",
        &[matrix.len(), matrix.len(), matrix.len()],
    )?;
    let mut values = matrix.to_vec();
    let mut vectors = vec![0.0; size * size];
    for index in 0..size {
        vectors[index * size + index] = 1.0;
    }

    for _ in 0..100 {
        let mut largest = 0.0_f64;
        for row in 0..size {
            for column in (row + 1)..size {
                let candidate = values[row * size + column].abs();
                if candidate > largest {
                    largest = candidate;
                }
            }
        }
        let diagonal_scale = (0..size)
            .map(|index| values[index * size + index].abs())
            .fold(1.0_f64, f64::max);
        if largest <= EIGEN_TOLERANCE * diagonal_scale {
            let mut order: Vec<usize> = (0..size).collect();
            order.sort_by(|left, right| {
                values[*right * size + *right]
                    .total_cmp(&values[*left * size + *left])
                    .then_with(|| left.cmp(right))
            });
            let eigenvalues: Vec<f64> = order
                .iter()
                .map(|index| values[*index * size + *index])
                .collect();
            let mut eigenvectors = vec![0.0; size * size];
            for (new_column, old_column) in order.into_iter().enumerate() {
                for row in 0..size {
                    eigenvectors[row * size + new_column] = vectors[row * size + old_column];
                }
            }
            checked_finite("eigenvalues", &eigenvalues)?;
            checked_finite("eigenvectors", &eigenvectors)?;
            return Ok((eigenvalues, eigenvectors));
        }

        for p in 0..size {
            for q in (p + 1)..size {
                if values[p * size + q].abs() <= EIGEN_TOLERANCE * diagonal_scale {
                    continue;
                }
                let app = values[p * size + p];
                let aqq = values[q * size + q];
                let apq = values[p * size + q];
                let tau = (aqq - app) / (2.0 * apq);
                let tangent = if tau >= 0.0 {
                    1.0 / (tau + tau.hypot(1.0))
                } else {
                    -1.0 / (-tau + tau.hypot(1.0))
                };
                let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
                let sine = tangent * cosine;

                values[p * size + p] = app - tangent * apq;
                values[q * size + q] = aqq + tangent * apq;
                values[p * size + q] = 0.0;
                values[q * size + p] = 0.0;
                for index in 0..size {
                    if index != p && index != q {
                        let aip = values[index * size + p];
                        let aiq = values[index * size + q];
                        let next_ip = cosine * aip - sine * aiq;
                        let next_iq = sine * aip + cosine * aiq;
                        values[index * size + p] = next_ip;
                        values[p * size + index] = next_ip;
                        values[index * size + q] = next_iq;
                        values[q * size + index] = next_iq;
                    }
                    let vip = vectors[index * size + p];
                    let viq = vectors[index * size + q];
                    vectors[index * size + p] = cosine * vip - sine * viq;
                    vectors[index * size + q] = sine * vip + cosine * viq;
                }
            }
        }
    }

    Err(invalid("symmetric eigendecomposition did not converge"))
}

pub fn thin_principal_components(
    centered: &[f64],
    rows: usize,
    columns: usize,
) -> KernelResult<(Vec<f64>, Vec<f64>)> {
    if rows < 2 {
        return Err(invalid("principal components require at least two rows"));
    }
    thin_components(centered, rows, columns, (rows - 1) as f64, 1e-6)
}

pub fn thin_covariance_components(
    centered: &[f64],
    rows: usize,
    columns: usize,
) -> KernelResult<(Vec<f64>, Vec<f64>)> {
    if rows < 2 {
        return Err(invalid("covariance components require at least two rows"));
    }
    let fp32_rank_tolerance = rows.max(columns) as f64 * f64::from(f32::EPSILON);
    thin_components(centered, rows, columns, rows as f64, fp32_rank_tolerance)
}

fn thin_components(
    centered: &[f64],
    rows: usize,
    columns: usize,
    divisor: f64,
    relative_singular_tolerance: f64,
) -> KernelResult<(Vec<f64>, Vec<f64>)> {
    checked_matrix("centered values", centered, rows, columns)?;
    let gram = row_gram(centered, rows, columns, divisor)?;
    checked_work(
        "principal-component decomposition",
        &[centered.len(), gram.len(), gram.len(), gram.len()],
    )?;
    let (raw_eigenvalues, sample_vectors) = symmetric_eigen(&gram, rows)?;
    let leading = raw_eigenvalues.first().copied().unwrap_or(0.0).max(0.0);
    if leading <= f64::EPSILON {
        return Err(invalid("input has zero centered variance"));
    }
    let rank_floor = (leading * relative_singular_tolerance * relative_singular_tolerance)
        .max(f64::EPSILON * leading);
    let rank = raw_eigenvalues
        .iter()
        .take(rows.min(columns))
        .take_while(|value| **value > rank_floor)
        .count();
    if rank == 0 {
        return Err(invalid("input has no numerically stable components"));
    }

    let eigenvalues: Vec<f64> = raw_eigenvalues.into_iter().take(rank).collect();
    let mut basis = vec![0.0; rank * columns];
    for component in 0..rank {
        let denominator = (divisor * eigenvalues[component]).sqrt();
        for column in 0..columns {
            let mut value = 0.0;
            for row in 0..rows {
                value += centered[row * columns + column] * sample_vectors[row * rows + component];
            }
            basis[component * columns + column] = value / denominator;
        }
        normalize_and_orient(&mut basis[component * columns..(component + 1) * columns])?;
    }
    checked_finite("principal-component eigenvalues", &eigenvalues)?;
    checked_finite("principal-component basis", &basis)?;
    Ok((eigenvalues, basis))
}

pub fn normalize_and_orient(vector: &mut [f64]) -> KernelResult<()> {
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm <= f64::EPSILON {
        return Err(invalid("cannot normalize a zero vector"));
    }
    for value in vector.iter_mut() {
        *value /= norm;
    }
    let anchor = vector
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.abs()
                .total_cmp(&right.abs())
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .unwrap_or(0);
    if vector[anchor].is_sign_negative() {
        for value in vector {
            *value = -*value;
        }
    }
    Ok(())
}

pub fn validate_orthonormal(basis: &[f64], components: usize, columns: usize) -> KernelResult<()> {
    checked_matrix("basis", basis, components, columns)?;
    for left in 0..components {
        for right in left..components {
            let dot = (0..columns)
                .map(|column| basis[left * columns + column] * basis[right * columns + column])
                .sum::<f64>();
            let expected = if left == right { 1.0 } else { 0.0 };
            if (dot - expected).abs() > 1e-4 {
                return Err(invalid(format!(
                    "basis is not orthonormal at components {left},{right}"
                )));
            }
        }
    }
    Ok(())
}

pub fn orthonormalize_rows(values: &[f64], rows: usize, columns: usize) -> KernelResult<Vec<f64>> {
    checked_matrix("basis candidates", values, rows, columns)?;
    let mut basis = vec![0.0; values.len()];
    for row in 0..rows {
        let start = row * columns;
        basis[start..start + columns].copy_from_slice(&values[start..start + columns]);
        for previous in 0..row {
            let dot = (0..columns)
                .map(|column| basis[start + column] * basis[previous * columns + column])
                .sum::<f64>();
            for column in 0..columns {
                basis[start + column] -= dot * basis[previous * columns + column];
            }
        }
        let norm = basis[start..start + columns]
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if norm <= f64::EPSILON {
            return Err(invalid(format!(
                "basis candidate {row} is numerically dependent"
            )));
        }
        for value in &mut basis[start..start + columns] {
            *value /= norm;
        }
    }
    checked_finite("orthonormal basis", &basis)?;
    validate_orthonormal(&basis, rows, columns)?;
    Ok(basis)
}

pub fn solve(
    mut matrix: Vec<f64>,
    size: usize,
    mut rhs: Vec<f64>,
    outputs: usize,
) -> KernelResult<Vec<f64>> {
    checked_matrix("linear system", &matrix, size, size)?;
    checked_matrix("linear-system right side", &rhs, size, outputs)?;
    if size > MAX_CUBIC_DIMENSION {
        return Err(invalid(format!(
            "linear-system dimension {size} exceeds the {MAX_CUBIC_DIMENSION}-dimension compute ceiling"
        )));
    }
    checked_work("linear solve", &[matrix.len(), rhs.len(), rhs.len()])?;
    let scale = matrix
        .iter()
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);

    for pivot in 0..size {
        let mut best_row = pivot;
        let mut best_value = matrix[pivot * size + pivot].abs();
        for row in (pivot + 1)..size {
            let candidate = matrix[row * size + pivot].abs();
            if candidate > best_value {
                best_value = candidate;
                best_row = row;
            }
        }
        if best_value <= PIVOT_TOLERANCE * scale {
            return Err(invalid("linear system is singular or ill-conditioned"));
        }
        if best_row != pivot {
            for column in 0..size {
                matrix.swap(pivot * size + column, best_row * size + column);
            }
            for output in 0..outputs {
                rhs.swap(pivot * outputs + output, best_row * outputs + output);
            }
        }
        let diagonal = matrix[pivot * size + pivot];
        for row in (pivot + 1)..size {
            let factor = matrix[row * size + pivot] / diagonal;
            matrix[row * size + pivot] = 0.0;
            for column in (pivot + 1)..size {
                matrix[row * size + column] -= factor * matrix[pivot * size + column];
            }
            for output in 0..outputs {
                let pivot_value = rhs[pivot * outputs + output];
                rhs[row * outputs + output] -= factor * pivot_value;
            }
        }
    }

    let mut solution = vec![0.0; size * outputs];
    for row in (0..size).rev() {
        for output in 0..outputs {
            let mut value = rhs[row * outputs + output];
            for column in (row + 1)..size {
                value -= matrix[row * size + column] * solution[column * outputs + output];
            }
            solution[row * outputs + output] = value / matrix[row * size + row];
        }
    }
    checked_finite("linear-system solution", &solution)?;
    Ok(solution)
}

pub fn matrix_rank(values: &[f64], rows: usize, columns: usize) -> KernelResult<usize> {
    checked_matrix("rank matrix", values, rows, columns)?;
    let gram_elements = checked_output("rank Gram", &[columns, columns])?;
    checked_work(
        "matrix-rank decomposition",
        &[values.len(), values.len(), gram_elements],
    )?;
    let rounded: Vec<f64> = values
        .iter()
        .map(|value| f64::from(*value as f32))
        .collect();
    let mut gram = vec![0.0; gram_elements];
    for left in 0..columns {
        for right in left..columns {
            let dot = (0..rows)
                .map(|row| rounded[row * columns + left] * rounded[row * columns + right])
                .sum::<f64>();
            gram[left * columns + right] = dot;
            gram[right * columns + left] = dot;
        }
    }
    let (eigenvalues, _) = symmetric_eigen(&gram, columns)?;
    let leading_singular = eigenvalues.first().copied().unwrap_or(0.0).max(0.0).sqrt();
    if leading_singular == 0.0 {
        return Ok(0);
    }
    let tolerance = rows.max(columns) as f64 * f64::from(f32::EPSILON) * leading_singular;
    Ok(eigenvalues
        .iter()
        .take(rows.min(columns))
        .filter(|value| value.max(0.0).sqrt() > tolerance)
        .count())
}

#[cfg(test)]
mod tests {
    use super::{matrix_rank, solve, symmetric_eigen};

    #[test]
    fn symmetric_eigen_reconstructs_distinct_matrix() {
        let matrix = [2.0, 1.0, 1.0, 2.0];
        let (values, vectors) = symmetric_eigen(&matrix, 2).unwrap();
        assert!((values[0] - 3.0).abs() < 1e-12);
        assert!((values[1] - 1.0).abs() < 1e-12);
        for row in 0..2 {
            for column in 0..2 {
                let reconstructed = (0..2)
                    .map(|component| {
                        vectors[row * 2 + component]
                            * values[component]
                            * vectors[column * 2 + component]
                    })
                    .sum::<f64>();
                assert!((reconstructed - matrix[row * 2 + column]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn cyclic_eigen_preserves_residual_and_orthogonality_with_repeated_spectrum() {
        for size in [3, 16, 64] {
            let mut matrix = vec![0.0; size * size];
            for row in 0..size {
                for column in 0..size {
                    let u = ((row + 1) as f64).sin();
                    let v = ((column + 1) as f64).sin();
                    matrix[row * size + column] = u * v + if row == column { 2.0 } else { 0.0 };
                }
            }
            let (values, vectors) = symmetric_eigen(&matrix, size).unwrap();
            for row in 0..size {
                for column in 0..size {
                    let av = (0..size)
                        .map(|k| matrix[row * size + k] * vectors[k * size + column])
                        .sum::<f64>();
                    assert!((av - vectors[row * size + column] * values[column]).abs() < 1e-10);
                    let dot = (0..size)
                        .map(|k| vectors[k * size + row] * vectors[k * size + column])
                        .sum::<f64>();
                    assert!((dot - if row == column { 1.0 } else { 0.0 }).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn pivoted_solve_handles_zero_leading_diagonal() {
        let solution = solve(vec![0.0, 1.0, 2.0, 3.0], 2, vec![1.0, 5.0], 1).unwrap();
        assert!((solution[0] - 1.0).abs() < 1e-12);
        assert!((solution[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rank_rejects_dependent_columns() {
        assert_eq!(
            matrix_rank(&[1.0, 2.0, 2.0, 4.0, 3.0, 6.0], 3, 2).unwrap(),
            1
        );
    }
}
