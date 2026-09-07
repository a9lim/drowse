use drowse_fitting_wasm::kernel::detect_periodic_topology;
use std::f64::consts::PI;

fn samples(fields: &[&str]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let shape = fields[0];
    let n: usize = fields[1].parse().unwrap();
    let m: usize = fields[2].parse().unwrap();
    let width: f64 = fields[3].parse().unwrap();
    let noise: f64 = fields[4].parse().unwrap();
    let mut points = Vec::new();
    let mut truth = Vec::new();
    for i in 0..n {
        let u = 2.0 * PI * i as f64 / n as f64;
        for j in 0..m {
            let v = 2.0 * PI * j as f64 / m as f64;
            let z = if m > 1 { width * (2.0 * j as f64 / (m - 1) as f64 - 1.0) } else { 0.0 };
            let mut point = match shape {
                "circle" | "ellipse" => vec![width * u.cos(), u.sin()],
                "product" => vec![u.cos(), u.sin(), width * v.cos(), width * v.sin()],
                "donut" => vec![(1.0 + width * v.cos()) * u.cos(), (1.0 + width * v.cos()) * u.sin(), width * v.sin()],
                "mobius" => vec![(1.0 + z * (u / 2.0).cos()) * u.cos(), (1.0 + z * (u / 2.0).cos()) * u.sin(), z * (u / 2.0).sin()],
                "cylinder" => vec![u.cos(), u.sin(), z],
                "sphere" => {
                    let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
                    let angle = i as f64 * PI * (3.0 - 5.0_f64.sqrt());
                    let radius = (1.0 - z * z).sqrt();
                    vec![radius * angle.cos(), radius * angle.sin(), z]
                }
                "arc" => {
                    let angle = 1.5 * PI * i as f64 / (n - 1) as f64;
                    vec![angle.cos(), angle.sin()]
                }
                "grid" => vec![i as f64 / (n - 1) as f64, j as f64 / (m - 1) as f64],
                "line" => vec![i as f64 / (n - 1) as f64, 0.0],
                _ => panic!("unknown fixture {shape}"),
            };
            for (axis, x) in point.iter_mut().enumerate() {
                *x = f64::from((*x + noise * ((i * m + j) as f64 * 12.9898 + axis as f64 * 78.233).sin()) as f32);
            }
            points.push(point);
            truth.push(if shape == "product" { vec![u, v] } else { vec![u] });
        }
    }
    (points, truth)
}

fn gram(points: &[Vec<f64>]) -> Vec<f64> {
    let n = points.len();
    let means: Vec<f64> = (0..points[0].len())
        .map(|axis| points.iter().map(|row| row[axis]).sum::<f64>() / n as f64).collect();
    points.iter().flat_map(|left| points.iter().map(|right| {
        left.iter().zip(right).zip(&means).map(|((a, b), mean)| (a - mean) * (b - mean)).sum()
    })).collect()
}

#[test]
fn shared_adversarial_shapes_and_coordinate_fidelity() {
    for line in include_str!("../../fixtures/topology-adversarial-v1.csv").lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        let expected: usize = fields[5].parse().unwrap();
        for variant in 0..3 {
            let (mut points, mut truth) = samples(&fields);
            if variant == 1 {
                points.reverse();
                truth.reverse();
                points.rotate_left(7);
                truth.rotate_left(7);
            } else if variant == 2 {
                for row in &mut points {
                    for value in row {
                        *value = f64::from((*value * 0.025) as f32);
                    }
                }
            }
            let result = detect_periodic_topology(&gram(&points), points.len(), 6, None, None, 0.5);
            let chart = match result {
                Ok(chart) => chart,
                Err(error) => {
                    assert!(error.to_string().contains("connected components"), "{line}: {error}");
                    None
                }
            };
            assert_eq!(chart.as_ref().map_or(0, |chart| chart.dimensions), expected, "{line}, variant {variant}");
            if let Some(chart) = chart {
                let mut matches = Vec::new();
                for axis in 0..chart.dimensions {
                    let mut best = (0, 0.0_f64);
                    for (target, _) in truth[0].iter().enumerate() {
                        for sign in [-1.0, 1.0] {
                            let phases: Vec<f64> = (0..points.len()).map(|row| {
                                chart.angles[row * chart.dimensions + axis] + sign * truth[row][target]
                            }).collect();
                            let re = phases.iter().map(|x| x.cos()).sum::<f64>() / points.len() as f64;
                            let im = phases.iter().map(|x| x.sin()).sum::<f64>() / points.len() as f64;
                            let coherence = re.hypot(im);
                            if coherence > best.1 {
                                best = (target, coherence);
                            }
                        }
                    }
                    assert!(best.1 > 0.95, "folded/mixed axis: {line}, variant {variant}, coherence {}", best.1);
                    assert!(!matches.contains(&best.0), "duplicate/harmonic axis: {line}");
                    matches.push(best.0);
                }
            }
        }
    }
}
