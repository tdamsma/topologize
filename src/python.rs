//! Python-facing bindings. No pure algorithm logic here.

use pyo3::prelude::*;

use crate::{graph, inflate, skeleton_cdt};

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

/// Straighten a terminal endpoint by projecting it halfway toward the line
/// extrapolated from the penultimate segment direction.
/// Only call for terminal (degree-1) endpoints; junction endpoints must stay fixed.
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
            max_idx = i + 1;
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

/// Return the CDT triangles for each inflated polygon, using the same
/// boundary preprocessing (RDP simplification + subdivision) as `topologize`.
/// Useful for visualising and debugging the triangulation step.
///
/// Returns a flat list of triangles across all polygons, each as
/// ((x0,y0),(x1,y1),(x2,y2)).
#[pyfunction]
#[pyo3(signature = (curves, buffer_distance, per_curve_widths=None))]
pub fn triangulate_curves(
    curves: Vec<Vec<Pt>>,
    buffer_distance: f64,
    per_curve_widths: Option<Vec<Vec<f64>>>,
) -> Vec<(Pt, Pt, Pt)> {
    let min_step = buffer_distance * 0.15;
    let decimated: Vec<Vec<Pt>> = curves.iter().map(|c| decimate_curve(c, min_step)).collect();
    let polygons = inflate::inflate(&decimated, buffer_distance, per_curve_widths.as_deref());
    let rdp_boundary = buffer_distance * 0.15;
    let max_seg = buffer_distance * 1.5;
    let mut out = Vec::new();
    for (outer, holes) in polygons {
        let outer = subdivide_ring(&rdp(&outer, rdp_boundary), max_seg);
        let holes: Vec<Vec<Pt>> = holes
            .iter()
            .map(|h| subdivide_ring(h, max_seg))
            .collect();
        if outer.len() >= 3 {
            out.extend(skeleton_cdt::get_triangles(&outer, &holes));
        }
    }
    out
}

/// Inflate a list of polylines by `buffer_distance`, union all results,
/// and return the buffer polygons as (outer_ring, holes) pairs.
///
/// The same input decimation applied inside `topologize` is used here so
/// the polygons match exactly what the skeleton sees.
///
/// Parameters
/// ----------
/// curves : list of lists of (x, y) tuples
/// buffer_distance : float
/// per_curve_widths : list of lists of float, default None
///     Per-vertex radii for each curve (one list per curve). If provided,
///     each sub-list must have the same length as the corresponding curve.
///     Curves with an empty sub-list (or missing entry) use `buffer_distance`.
///
/// Returns
/// -------
/// list of (outer, holes) where outer and each hole is a list of (x, y) tuples
#[pyfunction]
#[pyo3(signature = (curves, buffer_distance, per_curve_widths=None))]
pub fn inflate_curves(
    curves: Vec<Vec<Pt>>,
    buffer_distance: f64,
    per_curve_widths: Option<Vec<Vec<f64>>>,
) -> Vec<(Vec<Pt>, Vec<Vec<Pt>>)> {
    let min_step = buffer_distance * 0.15;
    let decimated: Vec<Vec<Pt>> = curves
        .iter()
        .map(|c| decimate_curve(c, min_step))
        .collect();
    inflate::inflate(&decimated, buffer_distance, per_curve_widths.as_deref())
}

/// Core topologize logic, callable from both single and batch entry points.
fn topologize_inner(
    curves: &[Vec<Pt>],
    buffer_distance: f64,
    simplification: Option<f64>,
    min_tip_length: Option<f64>,
    junction_merge_fraction: Option<f64>,
    per_curve_widths: Option<&[Vec<f64>]>,
    compute_widths: bool,
) -> (Vec<Vec<Pt>>, Vec<Pt>, Vec<(usize, usize)>, Vec<Vec<f64>>) {
    let min_step = buffer_distance * 0.15;
    let decimated: Vec<Vec<Pt>> = curves
        .iter()
        .map(|c| decimate_curve(c, min_step))
        .collect();

    let polygons = inflate::inflate(&decimated, buffer_distance, per_curve_widths);

    let rdp_boundary = buffer_distance * 0.15;
    let max_seg = buffer_distance * 1.5;
    let polygons: Vec<(Vec<Pt>, Vec<Vec<Pt>>)> = polygons
        .into_iter()
        .map(|(outer, holes)| {
            let outer = subdivide_ring(&rdp(&outer, rdp_boundary), max_seg);
            let holes = holes
                .iter()
                .map(|h| subdivide_ring(h, max_seg))
                .collect();
            (outer, holes)
        })
        .collect();

    let snap_tol = buffer_distance / 20.0;
    let rdp_tol = simplification.unwrap_or(buffer_distance / 10.0);

    let mut all_segments: Vec<(Pt, Pt)> = Vec::new();
    for (outer, holes) in &polygons {
        if outer.len() < 3 {
            continue;
        }
        all_segments.extend(skeleton_cdt::skeletonize(outer, holes));
    }

    if all_segments.is_empty() {
        return (vec![], vec![], vec![], vec![]);
    }

    // Collect boundary vertices only if width computation was requested.
    let boundary_pts: Vec<Pt> = if compute_widths {
        let mut pts = Vec::new();
        for (outer, holes) in &polygons {
            pts.extend_from_slice(outer);
            for h in holes {
                pts.extend_from_slice(h);
            }
        }
        pts
    } else {
        Vec::new()
    };

    let raw_graph = graph::segments_to_graph(&all_segments, snap_tol);
    let tip_len = min_tip_length.unwrap_or(buffer_distance * 2.0);
    let graph = if tip_len > 0.0 {
        graph::prune_short_tips(&raw_graph, tip_len)
    } else {
        raw_graph
    };
    let merge_frac = junction_merge_fraction.unwrap_or(1.5);
    let graph = if merge_frac > 0.0 {
        graph::merge_close_junctions(&graph, merge_frac * buffer_distance)
    } else {
        graph
    };
    let chains = graph::extract_chains(&graph);

    let processed: Vec<(Vec<Pt>, usize, usize)> = chains
        .into_iter()
        .map(|chain| {
            let graph::Chain { mut pts, start_terminal, end_terminal, start_node, end_node } =
                chain;
            pts = smooth_chain_pass(&pts);
            pts = smooth_chain_pass(&pts);
            pts = smooth_chain_pass(&pts);
            if end_terminal {
                straighten_terminal_end(&mut pts);
            }
            if start_terminal {
                straighten_terminal_start(&mut pts);
            }
            if rdp_tol > 0.0 {
                pts = rdp(&pts, rdp_tol);
            }
            (pts, start_node, end_node)
        })
        .collect();

    let mut node_remap: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    let mut nodes: Vec<Pt> = Vec::new();
    let mut chain_node_ids: Vec<(usize, usize)> = Vec::new();

    for (pts, sn, en) in &processed {
        let compact_s = if let Some(&id) = node_remap.get(sn) {
            id
        } else {
            let id = nodes.len();
            nodes.push(pts[0]);
            node_remap.insert(*sn, id);
            id
        };
        let compact_e = if let Some(&id) = node_remap.get(en) {
            id
        } else {
            let id = nodes.len();
            nodes.push(*pts.last().unwrap());
            node_remap.insert(*en, id);
            id
        };
        chain_node_ids.push((compact_s, compact_e));
    }

    let out: Vec<Vec<Pt>> = processed.into_iter().map(|(pts, _, _)| pts).collect();

    let chain_widths: Vec<Vec<f64>> = if compute_widths {
        out.iter().map(|pts| {
            pts.iter().map(|&(px, py)| {
                let min_d2 = boundary_pts.iter().fold(f64::INFINITY, |acc, &(bx, by)| {
                    let d2 = (px - bx) * (px - bx) + (py - by) * (py - by);
                    acc.min(d2)
                });
                2.0 * min_d2.sqrt()
            }).collect()
        }).collect()
    } else {
        vec![]
    };

    (out, nodes, chain_node_ids, chain_widths)
}

/// Topologize a list of polylines into clean centerline chains.
#[pyfunction]
#[pyo3(signature = (curves, buffer_distance, simplification=None, min_tip_length=None, junction_merge_fraction=None, per_curve_widths=None, compute_widths=false))]
pub fn topologize(
    py: Python<'_>,
    curves: Vec<Vec<Pt>>,
    buffer_distance: f64,
    simplification: Option<f64>,
    min_tip_length: Option<f64>,
    junction_merge_fraction: Option<f64>,
    per_curve_widths: Option<Vec<Vec<f64>>>,
    compute_widths: bool,
) -> PyResult<(Vec<Vec<Pt>>, Vec<Pt>, Vec<(usize, usize)>, Vec<Vec<f64>>)> {
    Ok(py.detach(|| topologize_inner(&curves, buffer_distance, simplification, min_tip_length, junction_merge_fraction, per_curve_widths.as_deref(), compute_widths)))
}

/// Process multiple independent curve-sets in parallel using Rayon.
///
/// Each element of `jobs` is a tuple of (curves, buffer_distance,
/// simplification, min_tip_length, junction_merge_fraction) — one
/// independent topologize invocation with its own parameters.
/// The GIL is released for the duration of the parallel work.
///
/// Returns a list of (chains, nodes, chain_node_ids) tuples, one per job.
#[pyfunction]
pub fn topologize_batch(
    py: Python<'_>,
    jobs: Vec<(Vec<Vec<Pt>>, f64, Option<f64>, Option<f64>, Option<f64>)>,
) -> PyResult<Vec<(Vec<Vec<Pt>>, Vec<Pt>, Vec<(usize, usize)>)>> {
    use rayon::prelude::*;
    Ok(py.detach(|| {
        jobs.par_iter()
            .map(|(curves, bd, simp, tip, jmf)| {
                let (chains, nodes, ids, _) = topologize_inner(curves, *bd, *simp, *tip, *jmf, None, false);
                (chains, nodes, ids)
            })
            .collect()
    }))
}
