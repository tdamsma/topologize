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

/// Taubin smoothing (λ|μ alternating filter) for an open polyline.
/// Each iteration applies a Laplacian shrink step (factor λ) followed by
/// an expand step (factor μ, negative) to remove high-frequency zigzag
/// without net shrinkage. Endpoints are preserved.
fn smooth_chain_taubin(pts: &[Pt], iterations: usize, lambda: f64, mu: f64) -> Vec<Pt> {
    let n = pts.len();
    if n < 3 {
        return pts.to_vec();
    }
    let mut current = pts.to_vec();
    let mut scratch = current.clone();
    for _ in 0..iterations {
        for &factor in &[lambda, mu] {
            scratch.copy_from_slice(&current);
            for i in 1..n - 1 {
                let (px, py) = scratch[i - 1];
                let (cx, cy) = scratch[i];
                let (nx, ny) = scratch[i + 1];
                let lx = (px + nx) / 2.0 - cx;
                let ly = (py + ny) / 2.0 - cy;
                current[i] = (cx + factor * lx, cy + factor * ly);
            }
        }
    }
    current
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

/// Spatial index of original curve segments with interpolated buffer widths.
///
/// Used to look up the local buffer width at any point on the inflated polygon
/// boundary, for per-point minimum-edge-length filtering in the skeleton step.
struct WidthIndex {
    cell_size: f64,
    grid: std::collections::HashMap<(i64, i64), Vec<usize>>,
    /// (x0, y0, w0, x1, y1, w1) per segment
    segments: Vec<(f64, f64, f64, f64, f64, f64)>,
}

impl WidthIndex {
    /// Build from decimated curves and their per-vertex widths.
    fn new(
        curves: &[Vec<Pt>],
        buffer_distance: f64,
        per_curve_widths: Option<&[Vec<f64>]>,
    ) -> Self {
        let cell_size = 4.0 * buffer_distance;
        let inv = 1.0 / cell_size;
        let mut segments = Vec::new();
        let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> =
            std::collections::HashMap::new();

        for (ci, curve) in curves.iter().enumerate() {
            if curve.len() < 2 {
                continue;
            }
            let widths: Option<&Vec<f64>> = per_curve_widths
                .and_then(|pcw| pcw.get(ci))
                .filter(|w| !w.is_empty());

            for i in 0..curve.len() - 1 {
                let (x0, y0) = curve[i];
                let (x1, y1) = curve[i + 1];
                let w0 = widths.map_or(buffer_distance, |w| w[i]);
                let w1 = widths.map_or(buffer_distance, |w| w[i + 1]);
                let seg_idx = segments.len();
                segments.push((x0, y0, w0, x1, y1, w1));

                // Rasterize segment into grid cells
                let dx = x1 - x0;
                let dy = y1 - y0;
                let seg_len = (dx * dx + dy * dy).sqrt();
                let steps = ((seg_len / cell_size).ceil() as usize).max(1);
                let mut inserted: std::collections::HashSet<(i64, i64)> =
                    std::collections::HashSet::new();
                for s in 0..=steps {
                    let t = s as f64 / steps as f64;
                    let px = x0 + t * dx;
                    let py = y0 + t * dy;
                    let cx = (px * inv).floor() as i64;
                    let cy = (py * inv).floor() as i64;
                    if inserted.insert((cx, cy)) {
                        grid.entry((cx, cy)).or_default().push(seg_idx);
                    }
                }
            }
        }

        Self {
            cell_size,
            grid,
            segments,
        }
    }

    /// Look up the local buffer width at point (px, py) by finding the
    /// closest original curve segment and interpolating its width.
    fn local_width_at(&self, px: f64, py: f64) -> f64 {
        let inv = 1.0 / self.cell_size;
        let cx = (px * inv).floor() as i64;
        let cy = (py * inv).floor() as i64;
        let mut best_dist_sq = f64::INFINITY;
        let mut best_width = f64::INFINITY;

        for dcx in -1..=1 {
            for dcy in -1..=1 {
                if let Some(bucket) = self.grid.get(&(cx + dcx, cy + dcy)) {
                    for &si in bucket {
                        let (x0, y0, w0, x1, y1, w1) = self.segments[si];
                        let sdx = x1 - x0;
                        let sdy = y1 - y0;
                        let len_sq = sdx * sdx + sdy * sdy;
                        let t = if len_sq < 1e-24 {
                            0.0
                        } else {
                            (((px - x0) * sdx + (py - y0) * sdy) / len_sq).clamp(0.0, 1.0)
                        };
                        let qx = x0 + t * sdx;
                        let qy = y0 + t * sdy;
                        let d2 = (px - qx) * (px - qx) + (py - qy) * (py - qy);
                        if d2 < best_dist_sq {
                            best_dist_sq = d2;
                            best_width = w0 + t * (w1 - w0);
                        }
                    }
                }
            }
        }

        best_width
    }
}

/// Spatial index of high-curvature boundary vertices.
///
/// Collects turning angles from all polygon rings, keeping only vertices
/// above `min_angle`. Stored in a flat grid for fast radius queries.
struct CurvatureIndex {
    /// (x, y, turning_angle) for each high-curvature vertex.
    entries: Vec<(f64, f64, f64)>,
    /// Grid cell size (= search radius for fast lookup).
    cell_size: f64,
    /// Grid buckets: cell coord → indices into `entries`.
    grid: std::collections::HashMap<(i64, i64), Vec<usize>>,
}

impl CurvatureIndex {
    fn new(search_radius: f64) -> Self {
        let cell_size = if search_radius > 0.0 && search_radius.is_finite() {
            search_radius
        } else {
            1.0 // safe fallback; queries will simply find nothing
        };
        Self {
            entries: Vec::new(),
            cell_size,
            grid: std::collections::HashMap::new(),
        }
    }

    /// Add all high-curvature vertices from a closed ring.
    fn add_ring(&mut self, pts: &[Pt], min_angle: f64) {
        let n = pts.len();
        if n < 3 {
            return;
        }
        for i in 0..n {
            let prev = pts[(i + n - 1) % n];
            let curr = pts[i];
            let next = pts[(i + 1) % n];
            let d1 = (curr.0 - prev.0, curr.1 - prev.1);
            let d2 = (next.0 - curr.0, next.1 - curr.1);
            let len1 = (d1.0 * d1.0 + d1.1 * d1.1).sqrt();
            let len2 = (d2.0 * d2.0 + d2.1 * d2.1).sqrt();
            if len1 < 1e-12 || len2 < 1e-12 {
                continue;
            }
            let cos_a = ((d1.0 * d2.0 + d1.1 * d2.1) / (len1 * len2)).clamp(-1.0, 1.0);
            let angle = cos_a.acos();
            if angle >= min_angle {
                let idx = self.entries.len();
                self.entries.push((curr.0, curr.1, angle));
                let inv = 1.0 / self.cell_size;
                let cx = (curr.0 * inv).floor() as i64;
                let cy = (curr.1 * inv).floor() as i64;
                self.grid.entry((cx, cy)).or_default().push(idx);
            }
        }
    }

    /// Query: highest (distance-weighted) turning angle near `(px, py)`.
    ///
    /// Full strength within `inner_radius`, linear decay to zero at
    /// `outer_radius`. Returns the max decayed angle across all entries.
    fn max_angle_near(&self, px: f64, py: f64, inner_radius: f64, outer_radius: f64) -> f64 {
        let inv = 1.0 / self.cell_size;
        let r_sq = outer_radius * outer_radius;
        let cx0 = ((px - outer_radius) * inv).floor() as i64;
        let cx1 = ((px + outer_radius) * inv).floor() as i64;
        let cy0 = ((py - outer_radius) * inv).floor() as i64;
        let cy1 = ((py + outer_radius) * inv).floor() as i64;
        let mut best = 0.0_f64;
        for cx in cx0..=cx1 {
            for cy in cy0..=cy1 {
                if let Some(bucket) = self.grid.get(&(cx, cy)) {
                    for &idx in bucket {
                        let (ex, ey, angle) = self.entries[idx];
                        let dx = ex - px;
                        let dy = ey - py;
                        let dist_sq = dx * dx + dy * dy;
                        if dist_sq <= r_sq {
                            let dist = dist_sq.sqrt();
                            let frac = if dist <= inner_radius {
                                1.0
                            } else {
                                1.0 - ((dist - inner_radius) / (outer_radius - inner_radius))
                            };
                            best = best.max(angle * frac);
                        }
                    }
                }
            }
        }
        best
    }
}

/// Curvature-adaptive refinement of a closed polygon ring using a spatial
/// index of high-curvature vertices (potentially from multiple rings).
///
/// For each edge, queries the index to find the highest nearby turning angle
/// (full strength within `3 * buffer_distance`, linear decay to zero at
/// `4 * buffer_distance`), then subdivides accordingly.
/// Only adds points — never removes or moves existing vertices.
fn refine_ring_from_index(
    pts: &[Pt],
    index: &CurvatureIndex,
    buffer_distance: f64,
    ratio_90: f64,
) -> Vec<Pt> {
    let n = pts.len();
    if n < 3 || ratio_90 <= 0.0 {
        return pts.to_vec();
    }
    let half_pi = std::f64::consts::FRAC_PI_2;
    let min_angle = 15.0_f64.to_radians();
    let min_len_90 = ratio_90 * buffer_distance;

    let mut result: Vec<Pt> = Vec::with_capacity(n * 2);
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        result.push(a);

        // Query at edge midpoint for the highest nearby curvature.
        let mx = (a.0 + b.0) * 0.5;
        let my = (a.1 + b.1) * 0.5;
        let inner_r = buffer_distance * 3.0;
        let outer_r = buffer_distance * 4.0;
        let max_angle = index.max_angle_near(mx, my, inner_r, outer_r);

        let effective_max = if max_angle < min_angle {
            f64::INFINITY
        } else {
            // Cap at 90° — angles above don't need more triangles.
            let clamped = max_angle.min(half_pi);
            let t = ((clamped - min_angle) / (half_pi - min_angle)).clamp(0.0, 1.0);
            // Quadratic: most refinement concentrated near 90°,
            // gentle at lower angles.
            let t2 = t * t;
            // At 15° → buffer_distance (barely any refinement).
            // At 90° → min_len_90 (= ratio × buffer_distance).
            let max_at_threshold = buffer_distance;
            max_at_threshold + t2 * (min_len_90 - max_at_threshold)
        };

        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        let seg_len = (dx * dx + dy * dy).sqrt();
        if seg_len > effective_max {
            let steps = (seg_len / effective_max).ceil() as usize;
            for k in 1..steps {
                let frac = k as f64 / steps as f64;
                result.push((a.0 + frac * dx, a.1 + frac * dy));
            }
        }
    }
    result
}

/// Shared boundary preprocessing: RDP simplify, baseline subdivision,
/// then optional curvature-adaptive refinement.
fn preprocess_boundaries(
    polygons: Vec<(Vec<Pt>, Vec<Vec<Pt>>)>,
    feature_size: f64,
    buffer_distance: f64,
    subdivision_ratio: Option<f64>,
) -> Vec<(Vec<Pt>, Vec<Vec<Pt>>)> {
    let rdp_boundary = feature_size * 0.15;
    let max_seg = feature_size * 1.5;
    let ratio = match subdivision_ratio {
        Some(r) if r > 0.0 => r.max(0.01),
        Some(r) => r, // 0 or negative: disables refinement
        None => 0.5,
    };

    // Step 1: baseline subdivision of all rings.
    let base_polygons: Vec<(Vec<Pt>, Vec<Vec<Pt>>)> = polygons
        .into_iter()
        .map(|(outer, holes)| {
            let outer = subdivide_ring(&rdp(&outer, rdp_boundary), max_seg);
            let holes = holes.iter().map(|h| subdivide_ring(h, max_seg)).collect();
            (outer, holes)
        })
        .collect();

    // Step 2: curvature-adaptive refinement (skip when ratio<=0 or buffer_distance <= 0).
    if buffer_distance > 0.0 && ratio > 0.0 {
        // Index threshold: only store vertices with ≥45° turn (skip endcap arcs ~36°).
        let index_min_angle = 45.0_f64.to_radians();
        let mut curv_index = CurvatureIndex::new(buffer_distance * 4.0);
        for (outer, holes) in &base_polygons {
            curv_index.add_ring(outer, index_min_angle);
            for h in holes {
                curv_index.add_ring(h, index_min_angle);
            }
        }
        base_polygons
            .into_iter()
            .map(|(outer, holes)| {
                let outer = refine_ring_from_index(&outer, &curv_index, buffer_distance, ratio);
                let holes = holes
                    .iter()
                    .map(|h| refine_ring_from_index(h, &curv_index, buffer_distance, ratio))
                    .collect();
                (outer, holes)
            })
            .collect()
    } else {
        base_polygons
    }
}

/// Return the CDT triangles for each inflated polygon, using the same
/// boundary preprocessing (RDP simplification + subdivision) as `topologize`.
/// Useful for visualising and debugging the triangulation step.
///
/// Returns a flat list of triangles across all polygons, each as
/// ((x0,y0),(x1,y1),(x2,y2)).
#[pyfunction]
#[pyo3(signature = (curves, buffer_distance, feature_size, per_curve_widths=None, subdivision_ratio=None))]
pub fn triangulate_curves(
    curves: Vec<Vec<Pt>>,
    buffer_distance: f64,
    feature_size: f64,
    per_curve_widths: Option<Vec<Vec<f64>>>,
    subdivision_ratio: Option<f64>,
) -> Vec<(Pt, Pt, Pt)> {
    let min_step = feature_size * 0.15;
    let decimated: Vec<Vec<Pt>> = curves.iter().map(|c| decimate_curve(c, min_step)).collect();
    let polygons = inflate::inflate(&decimated, buffer_distance, per_curve_widths.as_deref(), feature_size);
    let polygons = preprocess_boundaries(polygons, feature_size, buffer_distance, subdivision_ratio);

    let mut out = Vec::new();
    for (outer, holes) in polygons {
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
///     Uniform inflation radius (used for curves without per-vertex widths).
/// feature_size : float
///     Scale parameter for input decimation and derived thresholds.
/// per_curve_widths : list of lists of float, default None
///     Per-vertex radii for each curve (one list per curve). If provided,
///     each sub-list must have the same length as the corresponding curve.
///     Curves with an empty sub-list (or missing entry) use `buffer_distance`.
///
/// Returns
/// -------
/// list of (outer, holes) where outer and each hole is a list of (x, y) tuples
#[pyfunction]
#[pyo3(signature = (curves, buffer_distance, feature_size, per_curve_widths=None))]
pub fn inflate_curves(
    curves: Vec<Vec<Pt>>,
    buffer_distance: f64,
    feature_size: f64,
    per_curve_widths: Option<Vec<Vec<f64>>>,
) -> Vec<(Vec<Pt>, Vec<Vec<Pt>>)> {
    let min_step = feature_size * 0.15;
    let decimated: Vec<Vec<Pt>> = curves
        .iter()
        .map(|c| decimate_curve(c, min_step))
        .collect();
    inflate::inflate(&decimated, buffer_distance, per_curve_widths.as_deref(), feature_size)
}

/// Core topologize logic, callable from both single and batch entry points.
fn topologize_inner(
    curves: &[Vec<Pt>],
    buffer_distance: f64,
    feature_size: f64,
    simplification: Option<f64>,
    min_tip_length: Option<f64>,
    junction_merge_fraction: Option<f64>,
    per_curve_widths: Option<&[Vec<f64>]>,
    compute_widths: bool,
    subdivision_ratio: Option<f64>,
) -> (Vec<Vec<Pt>>, Vec<Pt>, Vec<(usize, usize)>, Vec<Vec<f64>>) {
    let min_step = feature_size * 0.15;
    let decimated: Vec<Vec<Pt>> = curves
        .iter()
        .map(|c| decimate_curve(c, min_step))
        .collect();

    let polygons = inflate::inflate(&decimated, buffer_distance, per_curve_widths, feature_size);
    let polygons = preprocess_boundaries(polygons, feature_size, buffer_distance, subdivision_ratio);

    let rdp_boundary = feature_size * 0.15;

    let snap_tol = feature_size / 20.0;
    let rdp_tol = simplification.unwrap_or(feature_size / 10.0);

    // Build width index only for variable-width case
    let width_index = per_curve_widths
        .filter(|pcw| !pcw.is_empty())
        .map(|pcw| WidthIndex::new(&decimated, buffer_distance, Some(pcw)));

    // The skeleton threshold is 2×buffer_width (the full polygon width), but
    // RDP simplification can narrow the polygon by up to rdp_boundary per side,
    // so we subtract that to avoid filtering legitimate cross-edges.
    let rdp_shrink = 2.0 * rdp_boundary;

    let mut all_segments: Vec<(Pt, Pt)> = Vec::new();
    for (outer, holes) in &polygons {
        if outer.len() < 3 {
            continue;
        }
        let min_edge_lengths: Vec<f64> = if let Some(ref idx) = width_index {
            // Variable widths: query index per boundary point
            outer
                .iter()
                .chain(holes.iter().flat_map(|h| h.iter()))
                .map(|&(x, y)| (2.0 * idx.local_width_at(x, y) - rdp_shrink).max(0.0))
                .collect()
        } else {
            // Uniform: fill with global threshold
            let threshold = (2.0 * buffer_distance - rdp_shrink).max(0.0);
            let n = outer.len() + holes.iter().map(|h| h.len()).sum::<usize>();
            vec![threshold; n]
        };
        all_segments.extend(skeleton_cdt::skeletonize(outer, holes, &min_edge_lengths));
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
    let tip_len = min_tip_length.unwrap_or(feature_size * 2.0);
    let graph = if tip_len > 0.0 {
        graph::prune_short_tips(&raw_graph, tip_len)
    } else {
        raw_graph
    };
    let merge_frac = junction_merge_fraction.unwrap_or(1.5);
    let graph = if merge_frac > 0.0 {
        graph::merge_close_junctions(&graph, merge_frac * feature_size)
    } else {
        graph
    };
    let chains = graph::extract_chains(&graph);

    let processed: Vec<(Vec<Pt>, usize, usize)> = chains
        .into_iter()
        .map(|chain| {
            let graph::Chain { mut pts, start_terminal, end_terminal, start_node, end_node } =
                chain;
            pts = smooth_chain_taubin(&pts, 3, 0.5, -0.53);
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
#[pyo3(signature = (curves, buffer_distance, feature_size, simplification=None, min_tip_length=None, junction_merge_fraction=None, per_curve_widths=None, compute_widths=false, subdivision_ratio=None))]
pub fn topologize(
    py: Python<'_>,
    curves: Vec<Vec<Pt>>,
    buffer_distance: f64,
    feature_size: f64,
    simplification: Option<f64>,
    min_tip_length: Option<f64>,
    junction_merge_fraction: Option<f64>,
    per_curve_widths: Option<Vec<Vec<f64>>>,
    compute_widths: bool,
    subdivision_ratio: Option<f64>,
) -> PyResult<(Vec<Vec<Pt>>, Vec<Pt>, Vec<(usize, usize)>, Vec<Vec<f64>>)> {
    Ok(py.detach(|| topologize_inner(&curves, buffer_distance, feature_size, simplification, min_tip_length, junction_merge_fraction, per_curve_widths.as_deref(), compute_widths, subdivision_ratio)))
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
    jobs: Vec<(Vec<Vec<Pt>>, f64, f64, Option<f64>, Option<f64>, Option<f64>)>,
) -> PyResult<Vec<(Vec<Vec<Pt>>, Vec<Pt>, Vec<(usize, usize)>)>> {
    use rayon::prelude::*;
    Ok(py.detach(|| {
        jobs.par_iter()
            .map(|(curves, bd, fs, simp, tip, jmf)| {
                let (chains, nodes, ids, _) = topologize_inner(curves, *bd, *fs, *simp, *tip, *jmf, None, false, None);
                (chains, nodes, ids)
            })
            .collect()
    }))
}
