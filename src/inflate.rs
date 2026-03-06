//! Inflate polylines using clipper2-rust, then union the results.
//! Returns a list of (outer_ring, holes) pairs.

use clipper2_rust::{area, inflate_paths_d, simplify_paths, union_subjects_d, EndType, FillRule, JoinType, Point};

type Pt = (f64, f64);

// arc_tolerance = 0.4 × buffer_distance → ~2 arc segments per semicircular cap.
// Formula: steps = π / arccos(1 - arc_tol / delta)
// at 0.4: steps ≈ π / arccos(0.6) ≈ π / 0.927 ≈ 3.4  (rounds to 3)
// at 0.5: steps ≈ π / arccos(0.5) ≈ π / 1.047 ≈ 3.0  (2 midpoints → 2 segs)
const ARC_TOL_FACTOR: f64 = 0.5;

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
    let mut outers: Vec<Vec<Pt>> = Vec::new();
    let mut all_holes: Vec<Vec<Pt>> = Vec::new();

    for path in &union_result {
        let pts: Vec<Pt> = path.iter().map(|p| (p.x, p.y)).collect();
        if area(path) > 0.0 {
            outers.push(pts);
        } else {
            all_holes.push(pts);
        }
    }

    // Assign each hole to the outer ring that contains it (centroid test).
    let mut result: Vec<(Vec<Pt>, Vec<Vec<Pt>>)> =
        outers.iter().map(|o| (o.clone(), Vec::new())).collect();

    for hole in all_holes {
        if hole.is_empty() {
            continue;
        }
        let cx = hole.iter().map(|p| p.0).sum::<f64>() / hole.len() as f64;
        let cy = hole.iter().map(|p| p.1).sum::<f64>() / hole.len() as f64;

        if result.len() == 1 {
            result[0].1.push(hole);
        } else {
            let owner = result
                .iter()
                .position(|(outer, _)| point_inside_ring((cx, cy), outer));
            if let Some(idx) = owner {
                result[idx].1.push(hole);
            }
        }
    }

    result
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
