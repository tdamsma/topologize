//! Python-facing bindings. No pure algorithm logic here.

use pyo3::prelude::*;
use std::time::Instant;

use crate::{graph, inflate, skeleton, skeleton_voronoi};

type Pt = (f64, f64);

/// Decimate a polyline: drop intermediate points closer than `min_step` to
/// the previous kept point. First and last points are always preserved.
/// Only removes points in dense regions; well-spaced points are untouched.
fn decimate_curve(pts: &[Pt], min_step: f64) -> Vec<Pt> {
    if pts.len() < 2 {
        return pts.to_vec();
    }
    let mut kept: Vec<Pt> = Vec::with_capacity(pts.len());
    kept.push(pts[0]);
    let mut last = pts[0];
    for &p in &pts[1..pts.len() - 1] {
        let dx = p.0 - last.0;
        let dy = p.1 - last.1;
        if dx * dx + dy * dy >= min_step * min_step {
            kept.push(p);
            last = p;
        }
    }
    kept.push(*pts.last().unwrap());
    kept
}

/// Straighten a terminal endpoint by projecting it onto the line extrapolated
/// from the penultimate segment. Moves the endpoint halfway toward that line
/// (same weighting as the interior smoothing). Only call for terminal endpoints
/// (degree-1 nodes); junction endpoints must not be moved.
fn straighten_terminal_end(pts: &mut Vec<Pt>) {
    let n = pts.len();
    if n < 3 {
        return;
    }
    let (px, py) = pts[n - 3];
    let (mx, my) = pts[n - 2];
    let (ex, ey) = pts[n - 1];
    let dx = mx - px;
    let dy = my - py;
    let len_sq = dx * dx + dy * dy;
    if len_sq == 0.0 {
        return;
    }
    // Project endpoint onto the line through pts[n-2] in direction (dx, dy).
    let t = ((ex - mx) * dx + (ey - my) * dy) / len_sq;
    let proj_x = mx + t * dx;
    let proj_y = my + t * dy;
    pts[n - 1] = ((ex + proj_x) / 2.0, (ey + proj_y) / 2.0);
}

fn straighten_terminal_start(pts: &mut Vec<Pt>) {
    let n = pts.len();
    if n < 3 {
        return;
    }
    let (px, py) = pts[2];
    let (mx, my) = pts[1];
    let (ex, ey) = pts[0];
    let dx = mx - px;
    let dy = my - py;
    let len_sq = dx * dx + dy * dy;
    if len_sq == 0.0 {
        return;
    }
    let t = ((ex - mx) * dx + (ey - my) * dy) / len_sq;
    let proj_x = mx + t * dx;
    let proj_y = my + t * dy;
    pts[0] = ((ex + proj_x) / 2.0, (ey + proj_y) / 2.0);
}

/// One pass of projection smoothing on an open polyline.
/// Each interior point moves halfway toward its projection on the line
/// through its two neighbours, reducing the CDT zigzag without moving
/// endpoints or introducing new points.
fn smooth_chain_pass(pts: &[Pt]) -> Vec<Pt> {
    let n = pts.len();
    if n < 3 {
        return pts.to_vec();
    }
    let mut out = pts.to_vec();
    for i in 1..n - 1 {
        let (px, py) = pts[i - 1];
        let (cx, cy) = pts[i];
        let (nx, ny) = pts[i + 1];
        let dx = nx - px;
        let dy = ny - py;
        let len_sq = dx * dx + dy * dy;
        if len_sq == 0.0 {
            continue;
        }
        let t = ((cx - px) * dx + (cy - py) * dy) / len_sq;
        let proj_x = px + t * dx;
        let proj_y = py + t * dy;
        out[i] = ((cx + proj_x) / 2.0, (cy + proj_y) / 2.0);
    }
    out
}

/// Ramer-Douglas-Peucker simplification of an open polyline.
fn rdp(pts: &[Pt], epsilon: f64) -> Vec<Pt> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let (ax, ay) = pts[0];
    let (bx, by) = *pts.last().unwrap();
    let dx = bx - ax;
    let dy = by - ay;
    let len_sq = dx * dx + dy * dy;

    let mut max_dist = 0.0f64;
    let mut max_idx = 0usize;
    for (i, &(px, py)) in pts[1..pts.len() - 1].iter().enumerate() {
        let dist = if len_sq == 0.0 {
            let ex = px - ax;
            let ey = py - ay;
            (ex * ex + ey * ey).sqrt()
        } else {
            let t = ((px - ax) * dx + (py - ay) * dy) / len_sq;
            let qx = ax + t * dx - px;
            let qy = ay + t * dy - py;
            (qx * qx + qy * qy).sqrt()
        };
        if dist > max_dist {
            max_dist = dist;
            max_idx = i + 1; // offset by 1 because we sliced [1..]
        }
    }

    if max_dist <= epsilon {
        return vec![pts[0], *pts.last().unwrap()];
    }

    let mut left = rdp(&pts[..=max_idx], epsilon);
    let right = rdp(&pts[max_idx..], epsilon);
    left.extend_from_slice(&right[1..]);
    left
}

/// Subdivide a closed polygon ring so that no edge exceeds `max_len`.
/// Inserts equally-spaced intermediate points on long edges; existing
/// vertices are never moved or removed.
fn subdivide_ring(pts: &[Pt], max_len: f64) -> Vec<Pt> {
    if pts.len() < 2 {
        return pts.to_vec();
    }
    let n = pts.len();
    // Treat the ring as closed: edges are pts[i] → pts[(i+1) % n].
    let mut result: Vec<Pt> = Vec::with_capacity(n * 2);
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        result.push(a);
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        let seg_len = (dx * dx + dy * dy).sqrt();
        if seg_len > max_len {
            let steps = (seg_len / max_len).ceil() as usize;
            for k in 1..steps {
                let t = k as f64 / steps as f64;
                result.push((a.0 + t * dx, a.1 + t * dy));
            }
        }
    }
    result
}

/// Topologize a list of polylines into clean centerline chains.
///
/// 1. Inflate all curves by `buffer_distance` and union them into polygons.
/// 2. Skeletonize each polygon (midpoint CDT or Voronoi, depending on `method`).
/// 3. Snap nearby endpoints and extract maximal non-branching chains.
///
/// Parameters
/// ----------
/// curves : list of lists of (x, y) tuples
/// buffer_distance : float
/// method : "midpoint" (default) | "voronoi"
///     "midpoint" — constrained Delaunay triangulation, midpoint graph.
///     "voronoi"  — Boost Voronoi diagram via the `centerline` crate.
/// cos_angle : float, default 0.0
///     Voronoi only. Cosine of the minimum acceptable angle between a Voronoi
///     edge and the nearest input segment. 0.0 keeps all edges; values toward
///     1.0 prune shallow-angle branches progressively.
/// simplification : float, default None (= buffer_distance / 10)
///     RDP (Ramer-Douglas-Peucker) tolerance applied to output polylines
///     (in input units). For midpoint: applied after projection smoothing.
///     For voronoi: applied internally by the skeletonizer.
///     Larger values produce fewer output points; 0.0 disables.
///
/// Returns
/// -------
/// list of lists of (x, y) tuples
#[pyfunction]
#[pyo3(signature = (curves, buffer_distance, method=None, cos_angle=0.0, simplification=None))]
pub fn topologize(
    _py: Python<'_>,
    curves: Vec<Vec<Pt>>,
    buffer_distance: f64,
    method: Option<&str>,
    cos_angle: f64,
    simplification: Option<f64>,
) -> PyResult<Vec<Vec<Pt>>> {
    let t0 = Instant::now();

    // Decimate dense input before inflate so clipper2 isn't fed millions of
    // nearly-duplicate points. min_step = 0.15 × buffer only removes points
    // in truly dense regions.
    let min_step = buffer_distance * 0.15;
    let decimated: Vec<Vec<Pt>> = curves
        .iter()
        .map(|c| decimate_curve(c, min_step))
        .collect();
    let n_in: usize = decimated.iter().map(|c| c.len()).sum();
    eprintln!("[timing] decimate:  {:>6.1} ms  ({} pts after)", t0.elapsed().as_secs_f64()*1e3, n_in);

    let t1 = Instant::now();
    let polygons = inflate::inflate(&decimated, buffer_distance);
    let n_poly_pts: usize = polygons.iter().map(|(o, hs)| o.len() + hs.iter().map(|h| h.len()).sum::<usize>()).sum();
    eprintln!("[timing] inflate:   {:>6.1} ms  ({} polys, {} boundary pts)", t1.elapsed().as_secs_f64()*1e3, polygons.len(), n_poly_pts);

    // Subdivide polygon ring edges after inflate so the CDT sees no edge
    // longer than max_seg. Long boundary edges produce elongated triangles
    // that break the midpoint skeleton. max_seg < 2 × buffer keeps triangles
    // compact. Existing vertices are kept; only intermediate points are added.
    // Simplify the polygon boundary: the CDT midpoint skeleton can't resolve
    // features below buffer_distance, so near-collinear boundary points are
    // just CDT overhead. RDP with epsilon = 0.15 × buffer reduces the dense
    // parallel-offset boundary dramatically without affecting skeleton quality.
    let t2 = Instant::now();
    let rdp_boundary = buffer_distance * 0.15;
    let polygons: Vec<(Vec<Pt>, Vec<Vec<Pt>>)> = polygons
        .into_iter()
        .map(|(outer, holes)| {
            let outer = rdp(&outer, rdp_boundary);
            let holes = holes.iter().map(|h| rdp(h, rdp_boundary)).collect();
            (outer, holes)
        })
        .collect();
    let n_rdp_pts: usize = polygons.iter().map(|(o, hs)| o.len() + hs.iter().map(|h| h.len()).sum::<usize>()).sum();
    eprintln!("[timing] rdp rings: {:>6.1} ms  ({} boundary pts after)", t2.elapsed().as_secs_f64()*1e3, n_rdp_pts);

    // Subdivide ring edges after simplification so the CDT sees no edge
    // longer than max_seg. Existing vertices are kept; only inserts new pts.
    let t2b = Instant::now();
    let max_seg = buffer_distance * 1.5;
    let polygons: Vec<(Vec<Pt>, Vec<Vec<Pt>>)> = polygons
        .into_iter()
        .map(|(outer, holes)| {
            let outer = subdivide_ring(&outer, max_seg);
            let holes = holes
                .iter()
                .map(|h| subdivide_ring(h, max_seg))
                .collect();
            (outer, holes)
        })
        .collect();
    let n_sub_pts: usize = polygons.iter().map(|(o, hs)| o.len() + hs.iter().map(|h| h.len()).sum::<usize>()).sum();
    eprintln!("[timing] subdivide: {:>6.1} ms  ({} boundary pts after)", t2b.elapsed().as_secs_f64()*1e3, n_sub_pts);

    let snap_tol = buffer_distance / 20.0;
    let use_voronoi = matches!(method, Some("voronoi"));
    // For voronoi, rdp_tol is used internally by the skeletonizer.
    // For midpoint, it is applied as a post-processing step below.
    let rdp_tol = simplification.unwrap_or(buffer_distance / 10.0);

    let t3 = Instant::now();
    let mut all_segments: Vec<(Pt, Pt)> = Vec::new();
    for (outer, holes) in &polygons {
        if outer.len() < 3 {
            continue;
        }
        let segs = if use_voronoi {
            skeleton_voronoi::voronoi_skeletonize(outer, holes, cos_angle, rdp_tol)
        } else {
            skeleton::skeletonize(outer, holes, buffer_distance)
        };
        all_segments.extend(segs);
    }
    eprintln!("[timing] skeleton:  {:>6.1} ms  ({} segments)", t3.elapsed().as_secs_f64()*1e3, all_segments.len());

    if all_segments.is_empty() {
        return Ok(vec![]);
    }

    let t4 = Instant::now();
    let graph = graph::segments_to_graph(&all_segments, snap_tol);
    let chains = graph::extract_chains(&graph);
    eprintln!("[timing] graph:     {:>6.1} ms  ({} chains)", t4.elapsed().as_secs_f64()*1e3, chains.len());

    // Post-process midpoint chains: smooth out CDT zigzag, then RDP.
    // Voronoi already applies RDP internally; skip post-processing for it.
    if use_voronoi {
        return Ok(chains.into_iter().map(|c| c.pts).collect());
    }

    let t5 = Instant::now();
    let out: Vec<Vec<Pt>> = chains
        .into_iter()
        .map(|chain| {
            let graph::Chain { mut pts, start_terminal, end_terminal } = chain;
            // Three passes of projection smoothing to reduce the staircase
            // artifact from alternating CDT triangle orientations.
            pts = smooth_chain_pass(&pts);
            pts = smooth_chain_pass(&pts);
            pts = smooth_chain_pass(&pts);
            // Straighten terminal endpoints: project the endpoint onto the
            // extrapolated chain direction. Junction endpoints are left fixed
            // so chains remain connected.
            if end_terminal {
                straighten_terminal_end(&mut pts);
            }
            if start_terminal {
                straighten_terminal_start(&mut pts);
            }
            // RDP to remove redundant points on near-straight runs.
            if rdp_tol > 0.0 {
                rdp(&pts, rdp_tol)
            } else {
                pts
            }
        })
        .collect();

    let n_out_pts: usize = out.iter().map(|c| c.len()).sum();
    eprintln!("[timing] postproc:  {:>6.1} ms  ({} chains, {} pts)", t5.elapsed().as_secs_f64()*1e3, out.len(), n_out_pts);
    eprintln!("[timing] total:     {:>6.1} ms", t0.elapsed().as_secs_f64()*1e3);

    Ok(out)
}
