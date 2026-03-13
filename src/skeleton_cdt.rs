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
/// `min_edge_lengths`: per-boundary-point minimum edge length. Internal CDT
/// edges shorter than the local minimum (of both endpoints) are treated as
/// same-side (ignored). Typically set to `2.0 * local_buffer_width` minus
/// any boundary simplification tolerance, to suppress short cross-edges that
/// square endcaps create at polygon corners without filtering legitimate
/// edges in a simplified polygon. Pass an empty slice to disable length
/// filtering entirely.
pub fn skeletonize(outer: &[Pt], holes: &[Vec<Pt>], min_edge_lengths: &[f64]) -> Vec<Segment> {
    midpoint_segments(outer, holes, min_edge_lengths)
}

/// Return the raw CDT triangles as vertex triples, for debugging/visualisation.
pub fn get_triangles(outer: &[Pt], holes: &[Vec<Pt>]) -> Vec<(Pt, Pt, Pt)> {
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
        return vec![];
    }
    match cdt::triangulate_contours(&all_pts, &contours) {
        Ok(tris) => tris
            .iter()
            .map(|&(a, b, c)| (all_pts[a], all_pts[b], all_pts[c]))
            .collect(),
        Err(_) => vec![],
    }
}

fn midpoint_segments(
    outer: &[Pt],
    holes: &[Vec<Pt>],
    min_edge_lengths: &[f64],
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

    // Break any accidental vertex-on-constraint-edge coincidences that arise
    // from the subdivision step. A 1e-9 perturbation is invisible in the output.
    for (i, pt) in all_pts.iter_mut().enumerate() {
        let s = i as f64;
        pt.0 += 1e-9 * (s * 1.1_f64).sin();
        pt.1 += 1e-9 * (s * 1.3_f64).cos();
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

    // Topological same-side filter: an internal edge whose endpoints are
    // close along the boundary ring (≤ HOP_THRESHOLD hops) connects nearby
    // vertices on the same side of the polygon — not a cross-edge.
    const HOP_THRESHOLD: usize = 2;

    let has_len_filter = !min_edge_lengths.is_empty();

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
        // Too-short edge filter: square endcap corners produce narrow
        // triangles whose cross-edges are shorter than the per-point
        // threshold provided by the caller, causing "snake tongue"
        // artifacts. Use the minimum of both endpoints (conservative).
        if has_len_filter {
            let local_min = min_edge_lengths[e.0].min(min_edge_lengths[e.1]);
            let local_min_sq = local_min * local_min;
            let (ax, ay) = all_pts[e.0];
            let (bx, by) = all_pts[e.1];
            let dx = bx - ax;
            let dy = by - ay;
            if dx * dx + dy * dy < local_min_sq {
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

    out
}
