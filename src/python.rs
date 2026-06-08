//! Python-facing bindings. No pure algorithm logic here.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::{graph, inflate, skeleton_cdt};

type Pt = (f64, f64);

fn validate_feature_size(feature_size: f64) -> PyResult<()> {
    if feature_size <= 0.0 || !feature_size.is_finite() {
        return Err(PyValueError::new_err(format!(
            "feature_size must be a positive finite number, got {feature_size}"
        )));
    }
    Ok(())
}

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

/// Quantize a point to an integer cell for endpoint matching.
fn pt_key(p: Pt, inv_tol: f64) -> (i64, i64) {
    ((p.0 * inv_tol).round() as i64, (p.1 * inv_tol).round() as i64)
}

/// Effective per-vertex widths for curve `i`: the supplied vector if present
/// and non-empty, else a uniform fill at `buffer_distance`.
fn curve_widths(widths: Option<&[Vec<f64>]>, i: usize, len: usize, buffer_distance: f64) -> Vec<f64> {
    match widths.and_then(|w| w.get(i)).filter(|v| !v.is_empty()) {
        Some(v) => v.clone(),
        None => vec![buffer_distance; len],
    }
}

/// Join input polylines that share endpoints into maximal chains, so the
/// offsetter does not emit spurious square end-caps at interior junctions of a
/// contour authored as many separate subpaths. Only degree-2 shared endpoints
/// are merged; dead-ends (degree 1) and true junctions (degree >= 3) remain
/// boundaries. Endpoints within `tol` are treated as coincident.
///
/// Per-vertex widths are carried along (and reversed with a flipped curve) so
/// they stay aligned with points. Returns merged curves and, when `widths` is
/// `Some`, merged widths of equal per-curve length (uniform-width curves are
/// filled at `buffer_distance`).
fn merge_connected(
    curves: &[Vec<Pt>],
    widths: Option<&[Vec<f64>]>,
    buffer_distance: f64,
    tol: f64,
) -> (Vec<Vec<Pt>>, Option<Vec<Vec<f64>>>) {
    let n = curves.len();
    let has_w = widths.is_some();
    let inv = if tol > 0.0 { 1.0 / tol } else { 1e9 };

    // endpoint -> [(curve_idx, which_end)]; which_end: 0 = start, 1 = end.
    let mut at: std::collections::HashMap<(i64, i64), Vec<(usize, u8)>> =
        std::collections::HashMap::new();
    for (i, c) in curves.iter().enumerate() {
        if c.len() < 2 {
            continue;
        }
        at.entry(pt_key(c[0], inv)).or_default().push((i, 0));
        at.entry(pt_key(*c.last().unwrap(), inv)).or_default().push((i, 1));
    }

    // The single unused continuation at a degree-2 endpoint, if any.
    let cont_at = |k: (i64, i64), used: &[bool]| -> Option<(usize, u8)> {
        let v = at.get(&k)?;
        if v.len() != 2 {
            return None; // dead-end or real junction: do not merge through
        }
        let mut found = None;
        for &(j, e) in v {
            if !used[j] {
                if found.is_some() {
                    return None; // both ends unused (e.g. start curve): ambiguous
                }
                found = Some((j, e));
            }
        }
        found
    };

    let mut used = vec![false; n];
    let mut out_c: Vec<Vec<Pt>> = Vec::new();
    let mut out_w: Vec<Vec<f64>> = Vec::new();

    for s in 0..n {
        if used[s] || curves[s].len() < 2 {
            continue;
        }
        used[s] = true;
        let mut chain: Vec<Pt> = curves[s].clone();
        let mut wch: Vec<f64> = curve_widths(widths, s, curves[s].len(), buffer_distance);

        // extend forward off the chain's tail
        loop {
            let k = pt_key(*chain.last().unwrap(), inv);
            let (j, end) = match cont_at(k, &used) {
                Some(x) => x,
                None => break,
            };
            used[j] = true;
            let cj = &curves[j];
            let wj = curve_widths(widths, j, cj.len(), buffer_distance);
            if end == 0 {
                chain.extend_from_slice(&cj[1..]);
                if has_w {
                    wch.extend_from_slice(&wj[1..]);
                }
            } else {
                chain.extend(cj[..cj.len() - 1].iter().rev().copied());
                if has_w {
                    wch.extend(wj[..wj.len() - 1].iter().rev().copied());
                }
            }
        }
        // extend backward off the chain's head
        loop {
            let k = pt_key(chain[0], inv);
            let (j, end) = match cont_at(k, &used) {
                Some(x) => x,
                None => break,
            };
            used[j] = true;
            let cj = &curves[j];
            let wj = curve_widths(widths, j, cj.len(), buffer_distance);
            // build a piece that ENDS at k, then prepend it before the chain.
            let (mut piece, mut wpiece): (Vec<Pt>, Vec<f64>) = if end == 1 {
                (cj[..cj.len() - 1].to_vec(), wj[..wj.len() - 1].to_vec())
            } else {
                let mut p: Vec<Pt> = cj.iter().rev().copied().collect();
                p.pop();
                let mut w: Vec<f64> = wj.iter().rev().copied().collect();
                w.pop();
                (p, w)
            };
            piece.extend_from_slice(&chain);
            chain = piece;
            if has_w {
                wpiece.extend_from_slice(&wch);
                wch = wpiece;
            }
        }

        out_c.push(chain);
        if has_w {
            out_w.push(wch);
        }
    }

    // keep degenerate (<2 pt) curves so nothing is silently dropped
    for i in 0..n {
        if !used[i] && !curves[i].is_empty() {
            out_c.push(curves[i].clone());
            if has_w {
                out_w.push(curve_widths(widths, i, curves[i].len(), buffer_distance));
            }
        }
    }

    (out_c, if has_w { Some(out_w) } else { None })
}

/// Resolve the endpoint-merge tolerance: explicit override, else
/// `feature_size * 0.01`. Returns 0.0 (disabled) only when explicitly set <= 0.
fn merge_tol(merge_tolerance: Option<f64>, feature_size: f64) -> f64 {
    merge_tolerance
        .map(|t| t.max(0.0))
        .unwrap_or(feature_size * 0.01)
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

/// Resample an open polyline span at approximately-uniform arc-length spacing,
/// following the actual path (not chords between endpoints). Both endpoints are
/// always kept. Returns at least `[first, last]`.
fn resample_span_open(span: &[Pt], target: f64) -> Vec<Pt> {
    if span.len() < 2 {
        return span.to_vec();
    }
    let mut cum = vec![0.0_f64; span.len()];
    for i in 1..span.len() {
        let dx = span[i].0 - span[i - 1].0;
        let dy = span[i].1 - span[i - 1].1;
        cum[i] = cum[i - 1] + (dx * dx + dy * dy).sqrt();
    }
    let total = *cum.last().unwrap();
    if total <= 1e-12 {
        return vec![span[0], *span.last().unwrap()];
    }
    let n_int = ((total / target).round() as usize).max(1);
    let mut out = Vec::with_capacity(n_int + 1);
    let mut j = 0usize;
    for k in 0..=n_int {
        let s = total * (k as f64) / (n_int as f64);
        while j + 1 < span.len() && cum[j + 1] < s {
            j += 1;
        }
        if j + 1 >= span.len() {
            out.push(*span.last().unwrap());
        } else {
            let seg = cum[j + 1] - cum[j];
            let t = if seg > 1e-12 { (s - cum[j]) / seg } else { 0.0 };
            out.push((
                span[j].0 + t * (span[j + 1].0 - span[j].0),
                span[j].1 + t * (span[j + 1].1 - span[j].1),
            ));
        }
    }
    out
}

/// Indices of vertices whose turning angle exceeds `corner_angle` (radians):
/// genuine corners that must be preserved as anchors when resampling.
fn corner_anchors(pts: &[Pt], corner_angle: f64) -> Vec<usize> {
    let n = pts.len();
    let cos_thresh = corner_angle.cos();
    let mut anchors: Vec<usize> = Vec::new();
    for i in 0..n {
        let prev = pts[(i + n - 1) % n];
        let curr = pts[i];
        let next = pts[(i + 1) % n];
        let d1 = (curr.0 - prev.0, curr.1 - prev.1);
        let d2 = (next.0 - curr.0, next.1 - curr.1);
        let l1 = (d1.0 * d1.0 + d1.1 * d1.1).sqrt();
        let l2 = (d2.0 * d2.0 + d2.1 * d2.1).sqrt();
        if l1 < 1e-12 || l2 < 1e-12 {
            continue;
        }
        let cos_a = ((d1.0 * d2.0 + d1.1 * d2.1) / (l1 * l2)).clamp(-1.0, 1.0);
        // turning angle > corner_angle  <=>  cos(between directions) < cos(corner_angle)
        if cos_a < cos_thresh {
            anchors.push(i);
        }
    }
    anchors
}

/// Resample a closed ring to approximately-uniform vertex spacing while
/// preserving genuine sharp corners. `target` is the desired arc-length
/// spacing; vertices whose turning angle exceeds `corner_angle` (radians) are
/// always kept as anchors, and each span between consecutive anchors is
/// resampled uniformly along its actual path.
///
/// Unlike `subdivide_ring` (which only splits long edges and never thins dense
/// stretches), this decouples CDT vertex density from the offset's accidental
/// vertex distribution — the convex side of a tight bend no longer carries many
/// times more vertices than the concave side.
fn resample_ring(pts: &[Pt], target: f64, corner_angle: f64) -> Vec<Pt> {
    let n = pts.len();
    if n < 3 || target <= 0.0 {
        return pts.to_vec();
    }
    let anchors = corner_anchors(pts, corner_angle);

    // No genuine corners: resample the whole loop uniformly.
    if anchors.is_empty() {
        let mut span = pts.to_vec();
        span.push(pts[0]); // close
        let looped = resample_span_open(&span, target);
        // drop the duplicated closing point
        let r = looped[..looped.len() - 1].to_vec();
        return if r.len() >= 3 { r } else { pts.to_vec() };
    }

    let mut out: Vec<Pt> = Vec::new();
    let m = anchors.len();
    for a in 0..m {
        let start = anchors[a];
        let end = anchors[(a + 1) % m];
        let span = ring_span(pts, start, end);
        let resampled = resample_span_open(&span, target);
        // keep all but last; the end anchor is added as the next span's start
        out.extend_from_slice(&resampled[..resampled.len() - 1]);
    }
    // Never hand a degenerate ring downstream: a CDT contour needs ≥3 vertices.
    if out.len() >= 3 {
        out
    } else {
        pts.to_vec()
    }
}

/// Collect the run of vertices from `start` forward (cyclically) to `end`,
/// inclusive of both. When `start == end` (the single-anchor case) this returns
/// the *entire* ring wrapped back to the anchor, rather than a 1-point span.
fn ring_span(pts: &[Pt], start: usize, end: usize) -> Vec<Pt> {
    let n = pts.len();
    let mut span: Vec<Pt> = vec![pts[start]];
    let mut i = (start + 1) % n;
    loop {
        span.push(pts[i]);
        if i == end {
            break;
        }
        i = (i + 1) % n;
    }
    span
}

/// Radius of curvature (in buffer units) at which the adaptive resample starts
/// tightening the spacing. Above this the boundary is treated as straight and
/// sampled at `base_target`; below it the spacing eases down following a
/// constant chord-error law. Set so gentle bends (radius 5–10× buffer) — not
/// just sharp turns — pick up extra samples.
const CURVATURE_ONSET_RADIUS_FACTOR: f64 = 10.0;

/// Local target arc-length spacing at a point, driven by the curvature index.
///
/// The index is built by sampling the offset at `base_target` spacing, so a
/// per-vertex turning angle θ implies a local radius of curvature
/// `R ≈ base_target / θ`. We phrase the whole response in radius terms: spacing
/// stays at `base_target` until `R` drops below `CURVATURE_ONSET_RADIUS_FACTOR ×
/// buffer`, then eases down as `base_target · sqrt(R / onset)` — the constant
/// chord-error (sagitta) law — bottoming out at the `ratio_90 × buffer` floor.
/// Because spacing is a smooth function of position (no integer subdivision
/// counts), walking the boundary in steps of this size yields a continuously
/// varying vertex density with no jumps between adjacent edges.
fn local_target(
    index: &CurvatureIndex,
    px: f64,
    py: f64,
    base_target: f64,
    buffer_distance: f64,
    ratio_90: f64,
) -> f64 {
    let inner_r = buffer_distance * 3.0;
    let outer_r = buffer_distance * 4.0;
    let angle = index.max_angle_near(px, py, inner_r, outer_r);
    if angle <= 1e-9 {
        return base_target;
    }
    let radius = base_target / angle;
    let onset = CURVATURE_ONSET_RADIUS_FACTOR * buffer_distance;
    if radius >= onset {
        return base_target;
    }
    let floor = ratio_90 * buffer_distance;
    let curv_spacing = base_target * (radius / onset).sqrt();
    curv_spacing.clamp(floor.min(base_target), base_target)
}

/// Resample an open span with spatially-varying spacing: at each step the local
/// target spacing is queried from the curvature index, and the next sample is
/// interpolated *on the original span* that far along its arc. Every emitted
/// vertex therefore lies on the true offset boundary (never on a chord), and
/// the spacing varies continuously rather than in discrete subdivision steps.
/// Endpoints are always kept; a too-short tail is folded into the end anchor.
fn adaptive_resample_span_open(
    span: &[Pt],
    base_target: f64,
    index: &CurvatureIndex,
    buffer_distance: f64,
    ratio_90: f64,
) -> Vec<Pt> {
    if span.len() < 2 {
        return span.to_vec();
    }
    let mut cum = vec![0.0_f64; span.len()];
    for i in 1..span.len() {
        let dx = span[i].0 - span[i - 1].0;
        let dy = span[i].1 - span[i - 1].1;
        cum[i] = cum[i - 1] + (dx * dx + dy * dy).sqrt();
    }
    let total = *cum.last().unwrap();
    if total <= 1e-12 {
        return vec![span[0], *span.last().unwrap()];
    }

    let mut out: Vec<Pt> = vec![span[0]];
    let mut s = 0.0_f64;
    let mut j = 0usize;
    loop {
        let p = *out.last().unwrap();
        let step = local_target(index, p.0, p.1, base_target, buffer_distance, ratio_90).max(1e-6);
        let s_next = s + step;
        // Fold a short remaining tail into the end anchor to avoid a sliver edge.
        if s_next >= total - 0.5 * step {
            break;
        }
        while j + 1 < span.len() && cum[j + 1] < s_next {
            j += 1;
        }
        let q = if j + 1 >= span.len() {
            *span.last().unwrap()
        } else {
            let seg = cum[j + 1] - cum[j];
            let t = if seg > 1e-12 { (s_next - cum[j]) / seg } else { 0.0 };
            (
                span[j].0 + t * (span[j + 1].0 - span[j].0),
                span[j].1 + t * (span[j + 1].1 - span[j].1),
            )
        };
        out.push(q);
        s = s_next;
    }
    out.push(*span.last().unwrap());
    out
}

/// Curvature-adaptive resample of a closed ring: same corner-anchored, per-span
/// structure as [`resample_ring`], but each span is walked with
/// [`adaptive_resample_span_open`] so density follows curvature smoothly while
/// every vertex stays on the original boundary.
fn resample_ring_adaptive(
    pts: &[Pt],
    base_target: f64,
    index: &CurvatureIndex,
    buffer_distance: f64,
    ratio_90: f64,
    corner_angle: f64,
) -> Vec<Pt> {
    let n = pts.len();
    if n < 3 || base_target <= 0.0 {
        return pts.to_vec();
    }
    let anchors = corner_anchors(pts, corner_angle);

    if anchors.is_empty() {
        let mut span = pts.to_vec();
        span.push(pts[0]); // close
        let looped =
            adaptive_resample_span_open(&span, base_target, index, buffer_distance, ratio_90);
        let r = looped[..looped.len() - 1].to_vec();
        return if r.len() >= 3 { r } else { pts.to_vec() };
    }

    let mut out: Vec<Pt> = Vec::new();
    let m = anchors.len();
    for a in 0..m {
        let start = anchors[a];
        let end = anchors[(a + 1) % m];
        let span = ring_span(pts, start, end);
        let resampled =
            adaptive_resample_span_open(&span, base_target, index, buffer_distance, ratio_90);
        out.extend_from_slice(&resampled[..resampled.len() - 1]);
    }
    // Never hand a degenerate ring downstream: a CDT contour needs ≥3 vertices.
    if out.len() >= 3 {
        out
    } else {
        pts.to_vec()
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
/// Default boundary-RDP tolerance as a fraction of `feature_size`. Simplifies
/// the dense offset boundary before CDT just enough to erase square-join
/// micro-jank without visibly pulling the triangulation off the true offset.
const BOUNDARY_RDP_FRACTION: f64 = 0.05;

/// Resolve the boundary-RDP tolerance: explicit override, else
/// `feature_size * BOUNDARY_RDP_FRACTION`. Negative is clamped to 0 (no
/// simplification). Used both for the pre-CDT simplification and the skeleton's
/// cross-edge compensation, so they stay in sync.
fn boundary_rdp_tol(boundary_simplification: Option<f64>, feature_size: f64) -> f64 {
    boundary_simplification
        .map(|t| t.max(0.0))
        .unwrap_or(feature_size * BOUNDARY_RDP_FRACTION)
}

fn preprocess_boundaries(
    polygons: Vec<(Vec<Pt>, Vec<Vec<Pt>>)>,
    feature_size: f64,
    buffer_distance: f64,
    subdivision_ratio: Option<f64>,
    resample: Option<f64>,
    boundary_simplification: Option<f64>,
) -> Vec<(Vec<Pt>, Vec<Vec<Pt>>)> {
    let rdp_boundary = boundary_rdp_tol(boundary_simplification, feature_size);
    let max_seg = feature_size * 1.5;
    let ratio = match subdivision_ratio {
        Some(r) if r > 0.0 => r.max(0.01),
        Some(r) => r, // 0 or negative: disables curvature weighting
        None => 0.5,
    };
    // Uniform-resample target spacing (absolute units). When Some(>0), each ring
    // is redistributed to ~uniform spacing instead of merely split on long edges,
    // so the CDT no longer inherits the offset's lopsided vertex distribution.
    let resample_target = resample.filter(|&r| r > 0.0);
    // Genuine corners (turning angle ≥ this) are preserved during resampling.
    let corner_angle = 60.0_f64.to_radians();
    let curvature_on = buffer_distance > 0.0 && ratio > 0.0;
    // Index threshold: only store vertices with ≥45° turn (skip endcap arcs ~36°).
    let index_min_angle = 45.0_f64.to_radians();

    // --- Adaptive-resample path -------------------------------------------
    // Curvature refinement is folded *into* the resample rather than bolted on
    // afterwards: we choose where to sample the (RDP-simplified) offset — wider
    // apart on straights, closer through curves — and interpolate every sample
    // on the boundary itself. So vertices always lie on the true offset (never
    // on a chord added post-hoc) and density varies smoothly, not in discrete
    // subdivide-or-not jumps.
    if let Some(t) = resample_target {
        // RDP-simplify first; this is the boundary every sample is taken from.
        let post: Vec<(Vec<Pt>, Vec<Vec<Pt>>)> = polygons
            .into_iter()
            .map(|(outer, holes)| {
                (
                    rdp(&outer, rdp_boundary),
                    holes.iter().map(|h| rdp(h, rdp_boundary)).collect(),
                )
            })
            .collect();

        if !curvature_on {
            return post
                .iter()
                .map(|(outer, holes)| {
                    (
                        resample_ring(outer, t, corner_angle),
                        holes
                            .iter()
                            .map(|h| resample_ring(h, t, corner_angle))
                            .collect(),
                    )
                })
                .collect();
        }

        // Step 1+2: uniform pre-sample at the base spacing (so per-vertex turning
        // angles are meaningful), then index curved vertices across *all* rings —
        // cross-ring, so opposing channel walls get matched density. The angle
        // threshold corresponds to the onset radius at this spacing
        // (θ = t / R_onset): anything gentler than ~10× buffer is left at base.
        let onset = CURVATURE_ONSET_RADIUS_FACTOR * buffer_distance;
        let resample_index_min_angle = (t / onset).clamp(0.0, std::f64::consts::FRAC_PI_2);
        let mut curv_index = CurvatureIndex::new(buffer_distance * 4.0);
        for (outer, holes) in &post {
            let ring = resample_ring(outer, t, corner_angle);
            curv_index.add_ring(&ring, resample_index_min_angle);
            for h in holes {
                let ring = resample_ring(h, t, corner_angle);
                curv_index.add_ring(&ring, resample_index_min_angle);
            }
        }

        // Step 3: resample the boundary at curvature-refined arc-length steps.
        let adapt = |ring: &[Pt]| -> Vec<Pt> {
            resample_ring_adaptive(ring, t, &curv_index, buffer_distance, ratio, corner_angle)
        };
        return post
            .iter()
            .map(|(outer, holes)| (adapt(outer), holes.iter().map(|h| adapt(h)).collect()))
            .collect();
    }

    // --- Subdivide path (resample off) ------------------------------------
    // No uniform resample requested: split long edges only, then apply the
    // legacy post-hoc curvature refinement.
    let prep_outer = |ring: &[Pt]| -> Vec<Pt> { subdivide_ring(&rdp(ring, rdp_boundary), max_seg) };
    let prep_hole = |ring: &[Pt]| -> Vec<Pt> { subdivide_ring(ring, max_seg) };
    let base_polygons: Vec<(Vec<Pt>, Vec<Vec<Pt>>)> = polygons
        .into_iter()
        .map(|(outer, holes)| {
            let outer = prep_outer(&outer);
            let holes = holes.iter().map(|h| prep_hole(h)).collect();
            (outer, holes)
        })
        .collect();

    if curvature_on {
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
#[pyo3(signature = (curves, buffer_distance, feature_size, per_curve_widths=None, subdivision_ratio=None, resample=None, boundary_simplification=None, merge_tolerance=None))]
pub fn triangulate_curves(
    curves: Vec<Vec<Pt>>,
    buffer_distance: f64,
    feature_size: f64,
    per_curve_widths: Option<Vec<Vec<f64>>>,
    subdivision_ratio: Option<f64>,
    resample: Option<f64>,
    boundary_simplification: Option<f64>,
    merge_tolerance: Option<f64>,
) -> PyResult<Vec<(Pt, Pt, Pt)>> {
    validate_feature_size(feature_size)?;
    let mtol = merge_tol(merge_tolerance, feature_size);
    let (curves, widths) = if mtol > 0.0 {
        merge_connected(&curves, per_curve_widths.as_deref(), buffer_distance, mtol)
    } else {
        (curves, per_curve_widths)
    };
    let min_step = feature_size * 0.15;
    let decimated: Vec<Vec<Pt>> = curves.iter().map(|c| decimate_curve(c, min_step)).collect();
    let polygons = inflate::inflate(&decimated, buffer_distance, widths.as_deref(), feature_size);
    let polygons = preprocess_boundaries(polygons, feature_size, buffer_distance, subdivision_ratio, resample, boundary_simplification);

    let mut out = Vec::new();
    for (outer, holes) in polygons {
        if outer.len() >= 3 {
            out.extend(skeleton_cdt::get_triangles(&outer, &holes));
        }
    }
    Ok(out)
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
#[pyo3(signature = (curves, buffer_distance, feature_size, per_curve_widths=None, merge_tolerance=None))]
pub fn inflate_curves(
    curves: Vec<Vec<Pt>>,
    buffer_distance: f64,
    feature_size: f64,
    per_curve_widths: Option<Vec<Vec<f64>>>,
    merge_tolerance: Option<f64>,
) -> PyResult<Vec<(Vec<Pt>, Vec<Vec<Pt>>)>> {
    validate_feature_size(feature_size)?;
    let mtol = merge_tol(merge_tolerance, feature_size);
    let (curves, widths) = if mtol > 0.0 {
        merge_connected(&curves, per_curve_widths.as_deref(), buffer_distance, mtol)
    } else {
        (curves, per_curve_widths)
    };
    let min_step = feature_size * 0.15;
    let decimated: Vec<Vec<Pt>> = curves
        .iter()
        .map(|c| decimate_curve(c, min_step))
        .collect();
    Ok(inflate::inflate(&decimated, buffer_distance, widths.as_deref(), feature_size))
}

/// Core topologize logic, callable from both single and batch entry points.
fn topologize_inner(
    curves: &[Vec<Pt>],
    buffer_distance: f64,
    feature_size: f64,
    simplification: Option<f64>,
    min_tip_fraction: Option<f64>,
    junction_merge_fraction: Option<f64>,
    per_curve_widths: Option<&[Vec<f64>]>,
    compute_widths: bool,
    subdivision_ratio: Option<f64>,
    max_nodes: Option<usize>,
    resample: Option<f64>,
    boundary_simplification: Option<f64>,
    merge_tolerance: Option<f64>,
) -> Result<(Vec<Vec<Pt>>, Vec<Pt>, Vec<(usize, usize)>, Vec<Vec<f64>>), String> {
    // Join end-to-end subpaths first so the offset has no interior square caps
    // at junctions of a contour authored as separate pieces.
    let mtol = merge_tol(merge_tolerance, feature_size);
    let (merged_curves, merged_widths) = if mtol > 0.0 {
        merge_connected(curves, per_curve_widths, buffer_distance, mtol)
    } else {
        (curves.to_vec(), per_curve_widths.map(|w| w.to_vec()))
    };
    let curves: &[Vec<Pt>] = &merged_curves;
    let per_curve_widths: Option<&[Vec<f64>]> = merged_widths.as_deref();

    let min_step = feature_size * 0.15;
    let decimated: Vec<Vec<Pt>> = curves
        .iter()
        .map(|c| decimate_curve(c, min_step))
        .collect();

    let polygons = inflate::inflate(&decimated, buffer_distance, per_curve_widths, feature_size);
    let polygons = preprocess_boundaries(polygons, feature_size, buffer_distance, subdivision_ratio, resample, boundary_simplification);

    let snap_tol = feature_size / 20.0;
    let rdp_tol = simplification.unwrap_or(feature_size / 10.0);

    // Build the raw midpoint skeleton for every polygon. We deliberately keep
    // *every* internal cross-edge here, including short ones: culling short
    // edges at triangulation time is not topology-aware and severs legitimate
    // interior structure (it collapses T/X junctions and breaks the skeleton
    // across tight bends, especially with little/no boundary simplification).
    // Short spurs (square-endcap "snake tongues" and the like) are instead
    // removed downstream by `prune_short_tips`, which only culls chains attached
    // to a degree-1 node — after contracting degree-2 nodes so the whole spur is
    // measured, not a single edge.
    let mut all_segments: Vec<(Pt, Pt)> = Vec::new();
    for (outer, holes) in &polygons {
        if outer.len() < 3 {
            continue;
        }
        all_segments.extend(skeleton_cdt::skeletonize(outer, holes));
    }

    if all_segments.is_empty() {
        return Ok((vec![], vec![], vec![], vec![]));
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
    if let Some(limit) = max_nodes {
        if raw_graph.nodes.len() > limit {
            return Err(format!(
                "skeleton has {} nodes, exceeding max_nodes limit of {}",
                raw_graph.nodes.len(),
                limit
            ));
        }
    }
    // Prune first, then merge junctions: remove short noise spurs before
    // simplifying the junction structure, so spurs don't leave behind spurious
    // junctions. `prune_short_tips` is the only edge-culling step — it walks a
    // degree-1 tip through degree-2 nodes (contracting them into one chain),
    // measures the full spur length, and culls only if that whole spur is short.
    let tip_len = min_tip_fraction.unwrap_or(2.0) * feature_size;
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

    Ok((out, nodes, chain_node_ids, chain_widths))
}

/// Topologize a list of polylines into clean centerline chains.
#[pyfunction]
#[pyo3(signature = (curves, buffer_distance, feature_size, simplification=None, min_tip_fraction=None, junction_merge_fraction=None, per_curve_widths=None, compute_widths=false, subdivision_ratio=None, max_nodes=None, resample=None, boundary_simplification=None, merge_tolerance=None))]
pub fn topologize(
    py: Python<'_>,
    curves: Vec<Vec<Pt>>,
    buffer_distance: f64,
    feature_size: f64,
    simplification: Option<f64>,
    min_tip_fraction: Option<f64>,
    junction_merge_fraction: Option<f64>,
    per_curve_widths: Option<Vec<Vec<f64>>>,
    compute_widths: bool,
    subdivision_ratio: Option<f64>,
    max_nodes: Option<usize>,
    resample: Option<f64>,
    boundary_simplification: Option<f64>,
    merge_tolerance: Option<f64>,
) -> PyResult<(Vec<Vec<Pt>>, Vec<Pt>, Vec<(usize, usize)>, Vec<Vec<f64>>)> {
    validate_feature_size(feature_size)?;
    py.detach(|| topologize_inner(&curves, buffer_distance, feature_size, simplification, min_tip_fraction, junction_merge_fraction, per_curve_widths.as_deref(), compute_widths, subdivision_ratio, max_nodes, resample, boundary_simplification, merge_tolerance))
        .map_err(PyValueError::new_err)
}

/// Process multiple independent curve-sets in parallel using Rayon.
///
/// Each element of `jobs` is a tuple of (curves, buffer_distance,
/// feature_size, simplification, min_tip_fraction, junction_merge_fraction,
/// max_nodes) — one independent topologize invocation with its own parameters.
/// The GIL is released for the duration of the parallel work.
///
/// Returns a list of (chains, nodes, chain_node_ids) tuples, one per job.
#[pyfunction]
pub fn topologize_batch(
    py: Python<'_>,
    jobs: Vec<(Vec<Vec<Pt>>, f64, f64, Option<f64>, Option<f64>, Option<f64>, Option<usize>)>,
) -> PyResult<Vec<(Vec<Vec<Pt>>, Vec<Pt>, Vec<(usize, usize)>)>> {
    use rayon::prelude::*;
    for (_, _, fs, _, _, _, _) in &jobs {
        validate_feature_size(*fs)?;
    }
    py.detach(|| {
        jobs.par_iter()
            .enumerate()
            .map(|(job_idx, (curves, bd, fs, simp, tip, jmf, max_n))| {
                let (chains, nodes, ids, _) = topologize_inner(curves, *bd, *fs, *simp, *tip, *jmf, None, false, None, *max_n, None, None, None)
                    .map_err(|err| format!("topologize_batch job {job_idx} failed: {err}"))?;
                Ok((chains, nodes, ids))
            })
            .collect::<Result<Vec<_>, String>>()
    }).map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single corner anchor (`start == end`) must wrap the whole ring, not
    /// collapse to one vertex — otherwise the resampled ring is empty and the
    /// CDT silently produces no triangles (the buffer=50 "0 chains" bug).
    #[test]
    fn ring_span_single_anchor_wraps_full_ring() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0)];
        let span = ring_span(&pts, 1, 1); // start == end
        assert_eq!(span.len(), pts.len() + 1, "should wrap the entire ring");
        assert_eq!(span[0], pts[1]);
        assert_eq!(*span.last().unwrap(), pts[1]);
        // distinct anchors still produce the inclusive sub-run unchanged.
        let sub = ring_span(&pts, 1, 3);
        assert_eq!(sub, vec![pts[1], pts[2], pts[3]]);
    }

    /// Build a teardrop: a smooth half-circle (no corners) closed by a single
    /// sharp tip, so exactly one vertex exceeds the 60° corner threshold.
    fn teardrop() -> Vec<Pt> {
        let r = 10.0;
        let mut ring: Vec<Pt> = Vec::new();
        let steps = 24;
        for k in 0..=steps {
            let theta = std::f64::consts::PI * (k as f64) / (steps as f64);
            ring.push((r * theta.cos(), r * theta.sin())); // (R,0) over the top to (-R,0)
        }
        ring.push((0.0, -r)); // sharp tip
        ring
    }

    #[test]
    fn resample_ring_single_anchor_is_not_degenerate() {
        let ring = teardrop();
        let corner = 60.0_f64.to_radians();
        assert_eq!(
            corner_anchors(&ring, corner).len(),
            1,
            "test setup should yield exactly one corner anchor"
        );
        let out = resample_ring(&ring, 3.0, corner);
        assert!(out.len() >= 3, "single-anchor ring collapsed to {} pts", out.len());
    }

    #[test]
    fn resample_ring_adaptive_single_anchor_is_not_degenerate() {
        let ring = teardrop();
        let corner = 60.0_f64.to_radians();
        let index = CurvatureIndex::new(40.0);
        let out = resample_ring_adaptive(&ring, 3.0, &index, 10.0, 0.5, corner);
        assert!(out.len() >= 3, "single-anchor ring collapsed to {} pts", out.len());
    }
}
