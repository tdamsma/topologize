//! Targeted subdivision of input curves near intersections/junctions and
//! at sharp corners, to improve CDT skeleton stability when input vertices
//! shift slightly between runs.

use std::collections::HashMap;

type Pt = (f64, f64);

/// Subdivide input curves in regions that are sensitive to CDT instability:
/// near intersections/junctions (where curves are close) and at sharp corners.
///
/// `feature_size` controls all thresholds. Only segments within detected zones
/// are densified; straight, well-separated segments are untouched.
///
/// When `per_curve_widths` is provided, interpolated width values are produced
/// for every newly inserted point.
pub fn targeted_subdivide(
    curves: &[Vec<Pt>],
    per_curve_widths: Option<&[Vec<f64>]>,
    feature_size: f64,
    max_seg_in_zone: f64,
) -> (Vec<Vec<Pt>>, Option<Vec<Vec<f64>>>) {
    // A very large max_seg_in_zone effectively disables subdivision.
    if max_seg_in_zone <= 0.0 || !max_seg_in_zone.is_finite() {
        return (
            curves.to_vec(),
            per_curve_widths.map(|w| w.to_vec()),
        );
    }

    let proximity_threshold = 2.0 * feature_size;
    let zone_radius = 2.0 * feature_size;
    let corner_angle_cos = 0.707; // cos(45°)

    // Collect all segments with curve/index info for spatial hashing.
    let mut all_segs: Vec<Segment> = Vec::new();
    for (ci, curve) in curves.iter().enumerate() {
        if curve.len() < 2 {
            continue;
        }
        for si in 0..curve.len() - 1 {
            all_segs.push(Segment {
                curve_idx: ci,
                seg_idx: si,
                a: curve[si],
                b: curve[si + 1],
            });
        }
    }

    // Find proximity zones via spatial hash.
    let mut zones: Vec<Vec<Zone>> = curves.iter().map(|c| Vec::with_capacity(c.len())).collect();
    find_proximity_zones(&all_segs, proximity_threshold, zone_radius, &mut zones);

    // Find sharp corner zones.
    for (ci, curve) in curves.iter().enumerate() {
        find_sharp_corners(curve, corner_angle_cos, zone_radius, &mut zones[ci]);
    }

    // Subdivide each curve in its zones.
    let mut out_curves: Vec<Vec<Pt>> = Vec::with_capacity(curves.len());
    let mut out_widths: Option<Vec<Vec<f64>>> = per_curve_widths.map(|_| Vec::with_capacity(curves.len()));

    for (ci, curve) in curves.iter().enumerate() {
        let widths = per_curve_widths.and_then(|w| {
            if ci < w.len() && !w[ci].is_empty() {
                Some(w[ci].as_slice())
            } else {
                None
            }
        });
        let (sub_pts, sub_widths) =
            subdivide_curve_in_zones(curve, widths, &zones[ci], max_seg_in_zone);
        out_curves.push(sub_pts);
        if let Some(ref mut ow) = out_widths {
            ow.push(sub_widths.unwrap_or_default());
        }
    }

    (out_curves, out_widths)
}

// ---------- internal types ----------

struct Segment {
    curve_idx: usize,
    seg_idx: usize,
    a: Pt,
    b: Pt,
}

/// A zone on a curve where subdivision is needed.
/// Defined by center arc-length position and radius.
struct Zone {
    /// Index of the segment whose start point is closest to the zone center.
    seg_idx: usize,
    /// Fractional position along that segment (0.0 = at start vertex).
    _frac: f64,
    /// Radius in arc-length units around the center.
    radius: f64,
}

// ---------- proximity detection ----------

fn find_proximity_zones(
    segs: &[Segment],
    threshold: f64,
    zone_radius: f64,
    zones: &mut [Vec<Zone>],
) {
    if segs.is_empty() {
        return;
    }
    let cell_size = threshold;
    let inv_cell = 1.0 / cell_size;
    let thresh_sq = threshold * threshold;

    // Build grid: cell → list of segment indices.
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (idx, seg) in segs.iter().enumerate() {
        let cells = segment_cells(seg.a, seg.b, inv_cell);
        for c in cells {
            grid.entry(c).or_default().push(idx);
        }
    }

    // Check pairs within each cell.
    for bucket in grid.values() {
        for i in 0..bucket.len() {
            for j in (i + 1)..bucket.len() {
                let si = bucket[i];
                let sj = bucket[j];
                let sa = &segs[si];
                let sb = &segs[sj];

                // Skip adjacent segments on the same curve.
                if sa.curve_idx == sb.curve_idx {
                    let diff = if sa.seg_idx > sb.seg_idx {
                        sa.seg_idx - sb.seg_idx
                    } else {
                        sb.seg_idx - sa.seg_idx
                    };
                    if diff <= 1 {
                        continue;
                    }
                }

                let (dist_sq, ta, tb) = seg_seg_dist_sq(sa.a, sa.b, sb.a, sb.b);
                if dist_sq < thresh_sq {
                    zones[sa.curve_idx].push(Zone {
                        seg_idx: sa.seg_idx,
                        _frac: ta,
                        radius: zone_radius,
                    });
                    zones[sb.curve_idx].push(Zone {
                        seg_idx: sb.seg_idx,
                        _frac: tb,
                        radius: zone_radius,
                    });
                }
            }
        }
    }
}

/// Cells overlapped by a segment (conservative rasterization).
fn segment_cells(a: Pt, b: Pt, inv_cell: f64) -> Vec<(i64, i64)> {
    let min_x = a.0.min(b.0);
    let max_x = a.0.max(b.0);
    let min_y = a.1.min(b.1);
    let max_y = a.1.max(b.1);
    let cx0 = (min_x * inv_cell).floor() as i64;
    let cx1 = (max_x * inv_cell).floor() as i64;
    let cy0 = (min_y * inv_cell).floor() as i64;
    let cy1 = (max_y * inv_cell).floor() as i64;
    let mut cells = Vec::with_capacity(((cx1 - cx0 + 1) * (cy1 - cy0 + 1)) as usize);
    for cx in cx0..=cx1 {
        for cy in cy0..=cy1 {
            cells.push((cx, cy));
        }
    }
    cells
}

/// Squared distance between two line segments, plus parameters t_a, t_b.
fn seg_seg_dist_sq(a0: Pt, a1: Pt, b0: Pt, b1: Pt) -> (f64, f64, f64) {
    let da = (a1.0 - a0.0, a1.1 - a0.1);
    let db = (b1.0 - b0.0, b1.1 - b0.1);
    let r = (a0.0 - b0.0, a0.1 - b0.1);
    let a = da.0 * da.0 + da.1 * da.1;
    let e = db.0 * db.0 + db.1 * db.1;
    let f = db.0 * r.0 + db.1 * r.1;

    let eps = 1e-12;
    let (mut s, t);
    if a <= eps && e <= eps {
        s = 0.0;
        t = 0.0;
    } else if a <= eps {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = da.0 * r.0 + da.1 * r.1;
        if e <= eps {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b_val = da.0 * db.0 + da.1 * db.1;
            let denom = a * e - b_val * b_val;
            s = if denom.abs() > eps {
                ((b_val * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let tnom = b_val * s + f;
            if tnom < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if tnom > e {
                t = 1.0;
                s = ((b_val - c) / a).clamp(0.0, 1.0);
            } else {
                t = tnom / e;
            }
        }
    }

    let dx = r.0 + da.0 * s - db.0 * t;
    let dy = r.1 + da.1 * s - db.1 * t;
    (dx * dx + dy * dy, s, t)
}

// ---------- sharp corner detection ----------

fn find_sharp_corners(curve: &[Pt], cos_threshold: f64, zone_radius: f64, zones: &mut Vec<Zone>) {
    if curve.len() < 3 {
        return;
    }
    for i in 1..curve.len() - 1 {
        let (px, py) = curve[i - 1];
        let (cx, cy) = curve[i];
        let (nx, ny) = curve[i + 1];
        let d1 = (cx - px, cy - py);
        let d2 = (nx - cx, ny - cy);
        let len1 = (d1.0 * d1.0 + d1.1 * d1.1).sqrt();
        let len2 = (d2.0 * d2.0 + d2.1 * d2.1).sqrt();
        if len1 < 1e-12 || len2 < 1e-12 {
            continue;
        }
        let cos_angle = (d1.0 * d2.0 + d1.1 * d2.1) / (len1 * len2);
        if cos_angle < cos_threshold {
            // Sharp corner — subdivide around this vertex.
            // Add zone centered at the vertex (seg_idx = i-1 end / i start).
            zones.push(Zone {
                seg_idx: i.saturating_sub(1),
                _frac: 1.0,
                radius: zone_radius,
            });
        }
    }
}

// ---------- curve subdivision ----------

fn subdivide_curve_in_zones(
    curve: &[Pt],
    widths: Option<&[f64]>,
    zones: &[Zone],
    max_seg: f64,
) -> (Vec<Pt>, Option<Vec<f64>>) {
    if curve.len() < 2 || zones.is_empty() {
        return (curve.to_vec(), widths.map(|w| w.to_vec()));
    }

    // Compute cumulative arc lengths.
    let mut cum_len: Vec<f64> = Vec::with_capacity(curve.len());
    cum_len.push(0.0);
    for i in 1..curve.len() {
        let dx = curve[i].0 - curve[i - 1].0;
        let dy = curve[i].1 - curve[i - 1].1;
        cum_len.push(cum_len[i - 1] + (dx * dx + dy * dy).sqrt());
    }

    // Compute zone center arc-length positions and merge overlapping zones.
    let mut intervals: Vec<(f64, f64)> = Vec::with_capacity(zones.len());
    for z in zones {
        let seg_start_len = cum_len[z.seg_idx];
        let seg_end_len = if z.seg_idx + 1 < cum_len.len() {
            cum_len[z.seg_idx + 1]
        } else {
            *cum_len.last().unwrap()
        };
        let center = seg_start_len + z._frac * (seg_end_len - seg_start_len);
        let lo = (center - z.radius).max(0.0);
        let hi = (center + z.radius).min(*cum_len.last().unwrap());
        intervals.push((lo, hi));
    }
    intervals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Merge overlapping intervals.
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (lo, hi) in intervals {
        if let Some(last) = merged.last_mut() {
            if lo <= last.1 {
                last.1 = last.1.max(hi);
                continue;
            }
        }
        merged.push((lo, hi));
    }

    // Walk the curve and subdivide segments that fall within merged intervals.
    let mut out_pts: Vec<Pt> = Vec::with_capacity(curve.len() * 2);
    let mut out_w: Option<Vec<f64>> = widths.map(|_| Vec::with_capacity(curve.len() * 2));

    out_pts.push(curve[0]);
    if let (Some(ref mut ow), Some(w)) = (&mut out_w, widths) {
        ow.push(w[0]);
    }

    for i in 0..curve.len() - 1 {
        let seg_start = cum_len[i];
        let seg_end = cum_len[i + 1];
        let seg_len = seg_end - seg_start;

        // Check if this segment overlaps any zone.
        let in_zone = merged.iter().any(|&(lo, hi)| seg_start < hi && seg_end > lo);

        if in_zone && seg_len > max_seg {
            let steps = (seg_len / max_seg).ceil() as usize;
            let a = curve[i];
            let b = curve[i + 1];
            let wa = widths.map(|w| w[i]);
            let wb = widths.map(|w| w[i + 1]);
            for k in 1..steps {
                let t = k as f64 / steps as f64;
                out_pts.push((a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1)));
                if let (Some(ref mut ow), Some(wa_v), Some(wb_v)) = (&mut out_w, wa, wb) {
                    ow.push(wa_v + t * (wb_v - wa_v));
                }
            }
        }

        out_pts.push(curve[i + 1]);
        if let (Some(ref mut ow), Some(w)) = (&mut out_w, widths) {
            ow.push(w[i + 1]);
        }
    }

    (out_pts, out_w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_zones_returns_original() {
        let curves = vec![vec![(0.0, 0.0), (10.0, 0.0)]];
        let (out, _) = targeted_subdivide(&curves, None, 1.0);
        assert_eq!(out[0].len(), curves[0].len());
    }

    #[test]
    fn crossing_curves_get_subdivided() {
        // Two curves crossing near (5, 5).
        let curves = vec![
            vec![(0.0, 0.0), (10.0, 10.0)],
            vec![(0.0, 10.0), (10.0, 0.0)],
        ];
        let (out, _) = targeted_subdivide(&curves, None, 1.0);
        // Both curves should have more points than the originals.
        assert!(out[0].len() > 2, "curve 0 should be subdivided");
        assert!(out[1].len() > 2, "curve 1 should be subdivided");
    }

    #[test]
    fn sharp_corner_subdivided() {
        // L-shaped curve with a 90° corner.
        let curves = vec![vec![(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]];
        let (out, _) = targeted_subdivide(&curves, None, 1.0);
        assert!(out[0].len() > 3, "sharp corner should be subdivided");
    }

    #[test]
    fn widths_are_interpolated() {
        let curves = vec![
            vec![(0.0, 0.0), (10.0, 10.0)],
            vec![(0.0, 10.0), (10.0, 0.0)],
        ];
        let widths = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let (out, out_w) = targeted_subdivide(&curves, Some(&widths), 1.0);
        let ow = out_w.unwrap();
        for (ci, curve) in out.iter().enumerate() {
            assert_eq!(
                curve.len(),
                ow[ci].len(),
                "widths must match points for curve {}",
                ci
            );
        }
    }
}
