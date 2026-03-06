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
pub fn skeletonize(outer: &[Pt], holes: &[Vec<Pt>], buffer_distance: f64) -> Vec<Segment> {
    midpoint_segments(outer, holes, buffer_distance, 1.9)
}

fn midpoint_segments(
    outer: &[Pt],
    holes: &[Vec<Pt>],
    buffer_distance: f64,
    threshold: f64,
) -> Vec<Segment> {
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
        return vec![];
    }

    // Triangulate. `triangulate_contours` returns only interior triangles.
    let triangles = match cdt::triangulate_contours(&all_pts, &contours) {
        Ok(t) => t,
        Err(_) => return vec![],
    };

    // Build edge → adjacent triangle count.
    let mut edge_to_count: HashMap<(usize, usize), u8> = HashMap::new();
    for &(a, b, c) in &triangles {
        for e in [(a.min(b), a.max(b)), (b.min(c), b.max(c)), (a.min(c), a.max(c))] {
            *edge_to_count.entry(e).or_insert(0) += 1;
        }
    }

    let min_len = threshold * buffer_distance;

    let is_ignored = |e: (usize, usize)| -> bool {
        if edge_to_count.get(&e).copied().unwrap_or(0) <= 1 {
            return true;
        }
        let (ax, ay) = all_pts[e.0];
        let (bx, by) = all_pts[e.1];
        ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt() < min_len
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

    out
}
