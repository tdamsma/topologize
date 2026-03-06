//! Voronoi-based skeleton using the `centerline` crate (Boost Voronoi).
//!
//! The `centerline` crate takes boundary segments as integer coordinates,
//! builds a Boost Voronoi diagram, filters infinite/external edges, and
//! returns connected polylines tracing the medial axis.
//!
//! Key differences from the midpoint CDT approach:
//! - Uses the dual Voronoi diagram rather than a Delaunay midpoint graph.
//! - Curved Voronoi edges (parabolas near endpoints) are discretised.
//! - `cos_angle` controls how aggressively shallow-angle branches are pruned.

use boostvoronoi::geometry::{Line as BVLine, Point as BVPoint};
use centerline::Centerline;
use glam::DVec3;

type Pt = (f64, f64);
type Segment = (Pt, Pt);

/// Scale factor: converts f64 coordinates to i32 for Boost Voronoi.
/// Scale of 1000 gives 0.001-unit precision; handles coordinates up to ±2 000 000.
const SCALE: f64 = 1000.0;

/// Build a Voronoi-based skeleton from a polygon with optional holes.
///
/// `cos_angle`: cosine of the minimum acceptable angle between a Voronoi edge
/// and the nearest input segment. `0.0` keeps all edges; values approaching
/// `1.0` remove progressively more shallow-angle branches.
///
/// `rdp_tol`: RDP (Ramer-Douglas-Peucker) simplification tolerance applied to
/// the output polylines (in input units). Larger = fewer output points.
pub fn voronoi_skeletonize(
    outer: &[Pt],
    holes: &[Vec<Pt>],
    cos_angle: f64,
    rdp_tol: f64,
) -> Vec<Segment> {
    let segments = rings_to_bv_segments(outer, holes);
    if segments.is_empty() {
        return vec![];
    }

    let mut cl = Centerline::<i32, DVec3>::with_segments(segments);

    if cl.build_voronoi().is_err() {
        return vec![];
    }

    // rdp_tol is in input units; scale to match the integer coordinate space.
    if cl
        .calculate_centerline(cos_angle, rdp_tol * SCALE, None)
        .is_err()
    {
        return vec![];
    }

    let line_strings = match cl.line_strings.as_ref() {
        Some(ls) => ls,
        None => return vec![],
    };

    // Output vertices are in integer-scaled coordinates → divide by SCALE.
    // Filter: discard segments whose midpoint lies outside the input polygon.
    line_strings
        .iter()
        .flat_map(|ls| {
            ls.windows(2).map(|w| {
                let p1 = (w[0].x / SCALE, w[0].y / SCALE);
                let p2 = (w[1].x / SCALE, w[1].y / SCALE);
                (p1, p2)
            })
        })
        .filter(|(p1, p2)| {
            let mid = ((p1.0 + p2.0) / 2.0, (p1.1 + p2.1) / 2.0);
            point_in_polygon(mid, outer, holes)
        })
        .collect()
}

/// Ray-casting point-in-polygon test (even-odd rule).
fn point_in_ring(pt: Pt, ring: &[Pt]) -> bool {
    let (x, y) = pt;
    let n = ring.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = ring[i];
        let (xj, yj) = ring[j];
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn point_in_polygon(pt: Pt, outer: &[Pt], holes: &[Vec<Pt>]) -> bool {
    point_in_ring(pt, outer) && holes.iter().all(|hole| !point_in_ring(pt, hole))
}

fn rings_to_bv_segments(outer: &[Pt], holes: &[Vec<Pt>]) -> Vec<BVLine<i32>> {
    std::iter::once(outer)
        .chain(holes.iter().map(|h| h.as_slice()))
        .flat_map(|ring| {
            let n = ring.len();
            (0..n).filter_map(move |i| {
                let (x0, y0) = ring[i];
                let (x1, y1) = ring[(i + 1) % n];
                let p0 = BVPoint::new((x0 * SCALE) as i32, (y0 * SCALE) as i32);
                let p1 = BVPoint::new((x1 * SCALE) as i32, (y1 * SCALE) as i32);
                // Skip zero-length segments: after integer rounding they become
                // point sites, which the centerline crate cannot handle.
                if p0.x == p1.x && p0.y == p1.y {
                    None
                } else {
                    Some(BVLine::new(p0, p1))
                }
            })
        })
        .collect()
}
