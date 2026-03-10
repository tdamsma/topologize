//! Inflate polylines using clipper2-rust, then union the results.
//! Returns a list of (outer_ring, holes) pairs.

use clipper2_rust::{
    area, inflate_paths_d, union_subjects_d, ClipperOffset, DeltaCallback64, EndType, FillRule,
    JoinType, Path64, PathD, Paths64, Point, Point64,
};

type Pt = (f64, f64);

// arc_tolerance = factor × buffer_distance controls end-cap smoothness.
// Formula: steps = π / arccos(1 - arc_tol / delta)
// at 0.50: steps ≈ 3.0  (2 midpoints — visibly polygonal)
// at 0.20: steps ≈ 4.9  (~5 segments — smooth)
// at 0.10: steps ≈ 7.0  (7 segments — very smooth, more points)
const ARC_TOL_FACTOR: f64 = 0.2;

/// Inflate a single polyline with per-vertex radii using ClipperOffset + callback.
/// Returns PathsD-compatible result (Vec<Vec<Point<f64>>>).
fn inflate_curve_variable(
    pts: &[Pt],
    widths: &[f64],
    arc_tol: f64,
    decimal_places: i32,
) -> Vec<Vec<Point<f64>>> {
    let scale = 10_f64.powi(decimal_places);
    let path64: Path64 = pts
        .iter()
        .map(|&(x, y)| Point64::new((x * scale).round() as i64, (y * scale).round() as i64))
        .collect();
    // arc_tol is derived from the global buffer_distance, not per-vertex widths.
    // End-cap smoothness is therefore approximate when widths vary widely.
    let mut co = ClipperOffset::new(2.0, arc_tol * scale, false, false);
    co.add_path(&path64, JoinType::Square, EndType::Round);
    // Enforce that the widths slice matches the path length. A mismatch means
    // the caller passed wrong data; clamping would silently produce incorrect
    // geometry. Callers (Python wrapper and internal Rust code) are responsible
    // for ensuring lengths match before calling this function.
    assert_eq!(
        widths.len(),
        pts.len(),
        "inflate_curve_variable: widths.len() ({}) must equal pts.len() ({})",
        widths.len(),
        pts.len(),
    );
    let widths_owned: Vec<f64> = widths.to_vec();
    let cb: DeltaCallback64 = Box::new(move |_path: &Path64, _norms: &PathD, curr_idx: usize, _prev_idx: usize| {
        // SAFETY: lengths are asserted equal above; indexing is always in bounds.
        widths_owned[curr_idx] * scale
    });
    let mut solution64 = Paths64::new();
    co.execute_with_callback(cb, &mut solution64);
    solution64
        .iter()
        .map(|p64| {
            p64.iter()
                .map(|p| Point::new(p.x as f64 / scale, p.y as f64 / scale))
                .collect()
        })
        .collect()
}

/// Inflate a list of polylines by `buffer_distance`, union all results,
/// and return as (outer, holes) polygon pairs.
pub fn inflate(
    curves: &[Vec<Pt>],
    buffer_distance: f64,
    per_curve_widths: Option<&[Vec<f64>]>,
    feature_size: f64,
) -> Vec<(Vec<Pt>, Vec<Vec<Pt>>)> {
    let arc_tol = feature_size * ARC_TOL_FACTOR;

    let mut all_inflated: Vec<Vec<Point<f64>>> = Vec::new();

    for (i, pl) in curves.iter().enumerate() {
        if pl.len() < 2 {
            continue;
        }
        let inflated: Vec<Vec<Point<f64>>> =
            if let Some(widths_all) = per_curve_widths {
                if i < widths_all.len() && !widths_all[i].is_empty() {
                    inflate_curve_variable(pl, &widths_all[i], arc_tol, 6)
                } else {
                    let path: Vec<Point<f64>> = pl.iter().map(|&(x, y)| Point::new(x, y)).collect();
                    inflate_paths_d(&vec![path], buffer_distance, JoinType::Square, EndType::Round, 0.0, 6, arc_tol)
                }
            } else {
                let path: Vec<Point<f64>> = pl.iter().map(|&(x, y)| Point::new(x, y)).collect();
                inflate_paths_d(&vec![path], buffer_distance, JoinType::Square, EndType::Round, 0.0, 6, arc_tol)
            };
        all_inflated.extend(inflated);
    }

    if all_inflated.is_empty() {
        return vec![];
    }

    let union_result = union_subjects_d(&all_inflated, FillRule::NonZero, 6);

    // Separate outers (positive shoelace area) from holes (negative).
    let mut outers: Vec<(Vec<Pt>, f64, f64)> = Vec::new(); // (ring, perimeter, area)
    let mut all_holes: Vec<(Vec<Pt>, f64)> = Vec::new(); // (ring, area)

    // Holes smaller than this are degenerate artifacts (e.g. from coarse
    // input sampling) and would cause CDT to fail.
    let min_hole_area = feature_size * feature_size * 0.1;

    for path in &union_result {
        let pts = dedup_ring(path.iter().map(|p| (p.x, p.y)).collect());
        if pts.len() < 3 {
            continue;
        }
        let a = area(path);
        if a > 0.0 {
            let perim: f64 = pts
                .windows(2)
                .map(|w| {
                    let dx = w[1].0 - w[0].0;
                    let dy = w[1].1 - w[0].1;
                    (dx * dx + dy * dy).sqrt()
                })
                .sum();
            outers.push((pts, perim, a));
        } else if -a >= min_hole_area {
            all_holes.push((pts, -a)); // store positive area
        }
    }

    // Sort outers by perimeter ascending so the smallest enclosing outer
    // claims each hole first. This correctly handles nested outer rings and
    // ensures each hole is assigned to exactly one (the tightest) outer.
    outers.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let outer_areas: Vec<f64> = outers.iter().map(|(_, _, a)| *a).collect();
    let mut result: Vec<(Vec<Pt>, Vec<Vec<Pt>>)> =
        outers.into_iter().map(|(o, _, _)| (o, Vec::new())).collect();

    // Each hole is claimed by the smallest enclosing outer and removed from
    // the pool, so it is never double-assigned across disjoint regions.
    // A hole whose area exceeds the outer's area cannot be inside it.
    let mut remaining: Vec<Option<(Vec<Pt>, f64)>> =
        all_holes.into_iter().map(Some).collect();

    for (idx, (outer, holes_out)) in result.iter_mut().enumerate() {
        let outer_area = outer_areas[idx];
        for i in 0..remaining.len() {
            let centroid = match &remaining[i] {
                None => continue,
                Some((hole, _)) if hole.is_empty() => {
                    remaining[i] = None;
                    continue;
                }
                Some((hole, hole_area)) => {
                    if *hole_area >= outer_area {
                        continue; // hole larger than outer — impossible containment
                    }
                    hole[0] // contours from union never overlap: one point suffices
                }
            };
            if point_inside_ring(centroid, outer) {
                let (hole, _) = remaining[i].take().unwrap();
                holes_out.push(hole);
            }
        }
    }

    result
}

/// Remove consecutive duplicate (or near-duplicate) points from a ring.
/// Zero-length edges cause CDT to fail.
fn dedup_ring(pts: Vec<Pt>) -> Vec<Pt> {
    if pts.len() < 2 {
        return pts;
    }
    let mut out: Vec<Pt> = Vec::with_capacity(pts.len());
    for p in pts {
        if out.last().map_or(true, |&last| {
            let dx = p.0 - last.0;
            let dy = p.1 - last.1;
            dx * dx + dy * dy > 1e-10
        }) {
            out.push(p);
        }
    }
    out
}

fn point_inside_ring((px, py): Pt, poly: &[Pt]) -> bool {
    let n = poly.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}
