//! Midpoint-graph skeleton using the `cdt` crate (Formlabs sweep-line CDT).
//!
//! Same midpoint-graph algorithm as `skeleton.rs` but replaces the spade
//! incremental CDT with `cdt::triangulate_contours`, which:
//!   - uses a sweep-line O(n log n) algorithm instead of incremental O(n²)
//!   - only returns triangles inside the polygon (no centroid filter needed)
//!   - uses direct point indices (no handle→index mapping)

use std::collections::HashMap;

type Pt = (f64, f64);
type Segment = (Pt, Pt);

/// Build the midpoint-graph skeleton using the `cdt` crate.
///
/// Uses a topological (boundary-hop) criterion to distinguish cross-edges
/// (spanning the polygon width) from same-side edges (connecting nearby
/// boundary vertices).  This adapts automatically to variable local width.
///
/// Every cross-edge is kept, regardless of length: short cross-edges are *not*
/// filtered here. A length filter at this stage is not topology-aware and
/// severs legitimate interior structure (collapsing T/X junctions and breaking
/// the skeleton across tight bends). Short spurs are removed downstream by the
/// graph-level `prune_short_tips`, which only culls chains attached to a
/// degree-1 node.
pub fn skeletonize(outer: &[Pt], holes: &[Vec<Pt>]) -> Result<Vec<Segment>, String> {
    midpoint_segments(outer, holes)
}

/// Return the raw CDT triangles as vertex triples, for debugging/visualisation.
pub fn get_triangles(outer: &[Pt], holes: &[Vec<Pt>]) -> Result<Vec<(Pt, Pt, Pt)>, String> {
    let mut all_pts: Vec<Pt> = Vec::new();
    let mut contours: Vec<Vec<usize>> = Vec::new();
    for ring in std::iter::once(outer).chain(holes.iter().map(|h| h.as_slice())) {
        if ring.len() < 3 {
            continue;
        }
        let start = all_pts.len();
        all_pts.extend_from_slice(ring);
        let end = all_pts.len();
        let mut contour: Vec<usize> = (start..end).collect();
        contour.push(start);
        contours.push(contour);
    }
    if all_pts.len() < 3 || contours.is_empty() {
        return Ok(vec![]);
    }
    let triangles = triangulate_points(&mut all_pts, &contours)?;
    Ok(triangles
        .iter()
        .map(|&(a, b, c)| (all_pts[a], all_pts[b], all_pts[c]))
        .collect())
}

fn midpoint_segments(outer: &[Pt], holes: &[Vec<Pt>]) -> Result<Vec<Segment>, String> {
    // Build flat point list and closed contours (last index == first).
    let mut all_pts: Vec<Pt> = Vec::new();
    let mut contours: Vec<Vec<usize>> = Vec::new();

    for ring in std::iter::once(outer).chain(holes.iter().map(|h| h.as_slice())) {
        if ring.len() < 3 {
            continue;
        }
        let start = all_pts.len();
        all_pts.extend_from_slice(ring);
        let end = all_pts.len();
        let mut contour: Vec<usize> = (start..end).collect();
        contour.push(start); // close the contour
        contours.push(contour);
    }

    if all_pts.len() < 3 || contours.is_empty() {
        return Ok(vec![]);
    }

    // Build point → (ring_id, position_in_ring) mapping for topological
    // same-side detection.
    let n_pts = all_pts.len();
    let mut pt_ring: Vec<usize> = vec![0; n_pts];
    let mut pt_pos: Vec<usize> = vec![0; n_pts];
    let mut ring_lens: Vec<usize> = Vec::new();
    for (ring_id, contour) in contours.iter().enumerate() {
        let rlen = contour.len() - 1; // unique vertices (last == first)
        ring_lens.push(rlen);
        for (pos, &idx) in contour[..rlen].iter().enumerate() {
            pt_ring[idx] = ring_id;
            pt_pos[idx] = pos;
        }
    }

    let triangles = triangulate_points(&mut all_pts, &contours)?;

    // Build edge → adjacent triangle count.
    let mut edge_to_count: HashMap<(usize, usize), u8> = HashMap::new();
    for &(a, b, c) in &triangles {
        for e in [
            (a.min(b), a.max(b)),
            (b.min(c), b.max(c)),
            (a.min(c), a.max(c)),
        ] {
            *edge_to_count.entry(e).or_insert(0) += 1;
        }
    }

    // Topological same-side filter: an internal edge whose endpoints are
    // close along the boundary ring (≤ HOP_THRESHOLD hops) connects nearby
    // vertices on the same side of the polygon — not a cross-edge.
    const HOP_THRESHOLD: usize = 2;

    let is_ignored = |e: (usize, usize)| -> bool {
        // Boundary edge (adjacent to only 1 triangle).
        if edge_to_count.get(&e).copied().unwrap_or(0) <= 1 {
            return true;
        }
        // Same-side: both endpoints on the same ring and close in hops.
        let (ru, pu) = (pt_ring[e.0], pt_pos[e.0]);
        let (rv, pv) = (pt_ring[e.1], pt_pos[e.1]);
        if ru == rv {
            let rlen = ring_lens[ru];
            let diff = if pu > pv { pu - pv } else { pv - pu };
            let arc = diff.min(rlen - diff);
            if arc <= HOP_THRESHOLD {
                return true;
            }
        }
        false
    };

    let edge_midpoint = |e: (usize, usize)| -> Pt {
        let (ax, ay) = all_pts[e.0];
        let (bx, by) = all_pts[e.1];
        ((ax + bx) / 2.0, (ay + by) / 2.0)
    };

    let mut out: Vec<Segment> = Vec::new();
    for &(a, b, c) in &triangles {
        let edges = [
            (a.min(b), a.max(b)),
            (b.min(c), b.max(c)),
            (a.min(c), a.max(c)),
        ];
        let internal: Vec<(usize, usize)> =
            edges.iter().copied().filter(|&e| !is_ignored(e)).collect();

        match internal.len() {
            3 => {
                let (ax, ay) = all_pts[a];
                let (bx, by) = all_pts[b];
                let (cx, cy) = all_pts[c];
                let centroid = ((ax + bx + cx) / 3.0, (ay + by + cy) / 3.0);
                for &e in &internal {
                    out.push((edge_midpoint(e), centroid));
                }
            }
            2 => {
                out.push((edge_midpoint(internal[0]), edge_midpoint(internal[1])));
            }
            _ => {}
        }
    }

    Ok(out)
}

/// Keep the historical perturbation on the first attempt. Retry only exact
/// vertex/constraint coincidences, `cdt`'s numerical wedge-search failure and
/// its internal assertion panics, always starting from the original vertices
/// so perturbations do not accumulate. Invalid/crossing constraints are errors.
fn triangulate_points(
    points: &mut [Pt],
    contours: &[Vec<usize>],
) -> Result<Vec<(usize, usize, usize)>, String> {
    silence_cdt_panics();
    triangulate_with(points, contours, |pts, rings| {
        cdt::triangulate_contours(pts, rings)
    })
}

/// `cdt` panics are caught and reported as errors, so keep the default hook
/// from printing them; panics from anywhere else still reach it.
fn silence_cdt_panics() {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let from_cdt = info.location().is_some_and(|loc| {
                loc.file().split(['/', '\\']).any(|part| part.starts_with("cdt-"))
            });
            if !from_cdt {
                previous(info);
            }
        }));
    });
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
}

fn triangulate_with(
    points: &mut [Pt],
    contours: &[Vec<usize>],
    mut triangulate: impl FnMut(&[Pt], &[Vec<usize>]) -> Result<Vec<(usize, usize, usize)>, cdt::Error>,
) -> Result<Vec<(usize, usize, usize)>, String> {
    let original = points.to_vec();
    let mut min = (f64::INFINITY, f64::INFINITY);
    let mut max = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut magnitude = 0.0_f64;
    for &(x, y) in points.iter() {
        min.0 = min.0.min(x);
        min.1 = min.1.min(y);
        max.0 = max.0.max(x);
        max.1 = max.1.max(y);
        magnitude = magnitude.max(x.abs()).max(y.abs());
    }
    let extent = (max.0 - min.0).max(max.1 - min.1);
    let retry_step = 1e-8_f64
        .max(extent * 1e-10)
        .max(magnitude * f64::EPSILON * 32.0);
    for attempt in 0..4 {
        let epsilon = if attempt == 0 {
            1e-9
        } else {
            retry_step * 10_f64.powi(attempt - 1)
        };
        for (i, (point, &base)) in points.iter_mut().zip(&original).enumerate() {
            let seed = i as f64 + attempt as f64 * 17.0;
            point.0 = base.0 + epsilon * (seed * 1.1).sin();
            point.1 = base.1 + epsilon * (seed * 1.3).cos();
        }
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            triangulate(points, contours)
        }));
        match outcome {
            Ok(Ok(triangles)) => return Ok(triangles),
            Ok(Err(cdt::Error::PointOnFixedEdge(_) | cdt::Error::WedgeEscape)) | Err(_)
                if attempt < 3 =>
            {
                continue
            }
            Ok(Err(error)) => {
                return Err(format!(
                    "constrained triangulation failed after {} attempt(s): {error:?}",
                    attempt + 1
                ))
            }
            Err(payload) => {
                return Err(format!(
                    "constrained triangulation panicked after {} attempt(s): {}",
                    attempt + 1,
                    panic_message(payload.as_ref())
                ))
            }
        }
    }
    unreachable!("last attempt always returns")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coincidence_retry_is_deterministic_and_scale_aware() {
        let original = vec![(1e9, 1e9), (1e9 + 10., 1e9), (1e9, 1e9 + 10.)];
        let contours = vec![vec![0, 1, 2, 0]];
        let run = || {
            let mut pts = original.clone();
            let mut calls = 0;
            let triangles = triangulate_with(&mut pts, &contours, |pts, rings| {
                calls += 1;
                if calls == 1 {
                    Err(cdt::Error::PointOnFixedEdge(0))
                } else {
                    cdt::triangulate_contours(pts, rings)
                }
            })
            .unwrap();
            assert_eq!(calls, 2);
            assert!(!triangles.is_empty());
            assert_ne!(pts, original);
            pts
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn persistent_coincidence_is_reported() {
        let mut pts = vec![(0., 0.), (10., 0.), (0., 10.)];
        let mut calls = 0;
        let error = triangulate_with(&mut pts, &[vec![0, 1, 2, 0]], |_, _| {
            calls += 1;
            Err(cdt::Error::PointOnFixedEdge(1))
        })
        .unwrap_err();
        assert_eq!(calls, 4);
        assert!(error.contains("PointOnFixedEdge(1)"));
    }

    #[test]
    fn wedge_escape_is_retried() {
        let mut pts = vec![(0., 0.), (10., 0.), (0., 10.)];
        let mut calls = 0;
        let triangles = triangulate_with(&mut pts, &[vec![0, 1, 2, 0]], |pts, rings| {
            calls += 1;
            if calls == 1 {
                Err(cdt::Error::WedgeEscape)
            } else {
                cdt::triangulate_contours(pts, rings)
            }
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert!(!triangles.is_empty());
    }

    #[test]
    fn cdt_panic_is_retried_then_reported() {
        let mut pts = vec![(0., 0.), (10., 0.), (0., 10.)];
        let mut calls = 0;
        let triangles = triangulate_with(&mut pts, &[vec![0, 1, 2, 0]], |pts, rings| {
            calls += 1;
            if calls == 1 {
                panic!("assertion failed: edge_ca.buddy != EMPTY_EDGE");
            }
            cdt::triangulate_contours(pts, rings)
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert!(!triangles.is_empty());

        let error = triangulate_with(&mut pts, &[vec![0, 1, 2, 0]], |_, _| -> Result<_, cdt::Error> {
            panic!("assertion failed: edge_ca.buddy != EMPTY_EDGE")
        })
        .unwrap_err();
        assert!(error.contains("panicked after 4 attempt(s): assertion failed"));
    }

    #[test]
    fn invalid_input_is_reported_without_retry() {
        let outer = [(0., 0.), (f64::NAN, 0.), (0., 10.)];
        assert!(get_triangles(&outer, &[])
            .unwrap_err()
            .contains("InvalidInput"));
        assert!(skeletonize(&outer, &[])
            .unwrap_err()
            .contains("InvalidInput"));
    }

    #[test]
    fn collinear_boundary_vertices_produce_triangles() {
        let outer = [
            (0., 0.),
            (5., 0.),
            (10., 0.),
            (10., 2.5),
            (10., 5.),
            (5., 5.),
            (0., 5.),
            (0., 2.5),
        ];
        assert!(!get_triangles(&outer, &[]).unwrap().is_empty());
        assert!(skeletonize(&outer, &[]).is_ok());
    }

    #[test]
    fn perturbation_recovers_vertex_on_fixed_edge() {
        let mut points = vec![(0., 0.), (10., 0.), (10., 5.), (0., 5.), (5., 0.)];
        let contours = vec![vec![0, 1, 2, 3, 0]];
        assert!(matches!(
            cdt::triangulate_contours(&points, &contours),
            Err(cdt::Error::PointOnFixedEdge(_))
        ));
        assert!(!triangulate_points(&mut points, &contours)
            .unwrap()
            .is_empty());
    }
}
