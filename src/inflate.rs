//! Inflate polylines using clipper2-rust, then union the results.
//! Returns a list of (outer_ring, holes) pairs.

use clipper2_rust::{area, inflate_paths_d, union_subjects_d, EndType, FillRule, JoinType, Point};

type Pt = (f64, f64);

// arc_tolerance = factor × buffer_distance controls end-cap smoothness.
// Formula: steps = π / arccos(1 - arc_tol / delta)
// at 0.50: steps ≈ 3.0  (2 midpoints — visibly polygonal)
// at 0.20: steps ≈ 4.9  (~5 segments — smooth)
// at 0.10: steps ≈ 7.0  (7 segments — very smooth, more points)
const ARC_TOL_FACTOR: f64 = 0.2;

/// Inflate a list of polylines by `buffer_distance`, union all results,
/// and return as (outer, holes) polygon pairs.
pub fn inflate(curves: &[Vec<Pt>], buffer_distance: f64) -> Vec<(Vec<Pt>, Vec<Vec<Pt>>)> {
    let arc_tol = buffer_distance * ARC_TOL_FACTOR;

    let mut all_inflated: Vec<Vec<Point<f64>>> = Vec::new();

    for pl in curves {
        if pl.len() < 2 {
            continue;
        }
        let path: Vec<Point<f64>> = pl.iter().map(|&(x, y)| Point::new(x, y)).collect();
        let inflated = inflate_paths_d(
            &vec![path],
            buffer_distance,
            JoinType::Square,
            EndType::Round,
            0.0, // miter_limit (unused with Square/Round)
            6,   // decimal precision for internal integer scaling
            arc_tol,
        );
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
    let min_hole_area = buffer_distance * buffer_distance * 0.1;

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
