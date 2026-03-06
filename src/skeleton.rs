//! Midpoint-graph skeleton: constrained Delaunay triangulation → midpoint edges.

use std::collections::HashMap;

use spade::handles::FixedVertexHandle;
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

type Pt = (f64, f64);
type Segment = (Pt, Pt);

/// Build the midpoint-graph skeleton of a polygon with optional holes.
///
/// 1. Constrained Delaunay triangulation of all boundary edges.
/// 2. Filter triangles outside the polygon (centroid ray-cast test).
/// 3. Discard boundary edges (1 adjacent triangle) and short edges
///    (< `threshold * buffer_distance`).
/// 4. For each triangle with 2 internal edges: connect their midpoints.
///    For each triangle with 3 internal edges: connect each midpoint to the centroid.
pub fn skeletonize(outer: &[Pt], holes: &[Vec<Pt>], buffer_distance: f64) -> Vec<Segment> {
    midpoint_segments(outer, holes, buffer_distance, 1.9)
}

fn midpoint_segments(
    outer: &[Pt],
    holes: &[Vec<Pt>],
    buffer_distance: f64,
    threshold: f64,
) -> Vec<Segment> {
    let rings: Vec<&[Pt]> = std::iter::once(outer)
        .chain(holes.iter().map(|h| h.as_slice()))
        .collect();

    let mut all_pts: Vec<Pt> = Vec::new();
    let mut ring_offsets: Vec<usize> = Vec::new();
    for ring in &rings {
        ring_offsets.push(all_pts.len());
        all_pts.extend_from_slice(ring);
    }

    if all_pts.len() < 3 {
        return vec![];
    }

    let mut cdt = ConstrainedDelaunayTriangulation::<Point2<f64>>::new();
    let mut handles: Vec<FixedVertexHandle> = Vec::with_capacity(all_pts.len());
    for &(x, y) in &all_pts {
        match cdt.insert(Point2::new(x, y)) {
            Ok(h) => handles.push(h),
            Err(_) => handles.push(FixedVertexHandle::from_index(0)),
        }
    }
    for (ring_i, ring) in rings.iter().enumerate() {
        let offset = ring_offsets[ring_i];
        let n = ring.len();
        for i in 0..n {
            let h_a = handles[offset + i];
            let h_b = handles[offset + (i + 1) % n];
            if h_a != h_b {
                let _ = cdt.add_constraint(h_a, h_b);
            }
        }
    }

    let mut cdt_idx_to_pt: HashMap<usize, usize> = HashMap::new();
    for (pt_i, _) in all_pts.iter().enumerate() {
        let h = handles[pt_i];
        cdt_idx_to_pt.entry(h.index()).or_insert(pt_i);
    }

    struct TriIdx {
        a: usize,
        b: usize,
        c: usize,
    }

    let mut tris: Vec<TriIdx> = Vec::new();
    for face in cdt.inner_faces() {
        let verts = face.vertices();
        let a = cdt_idx_to_pt.get(&verts[0].fix().index()).copied();
        let b = cdt_idx_to_pt.get(&verts[1].fix().index()).copied();
        let c = cdt_idx_to_pt.get(&verts[2].fix().index()).copied();
        if let (Some(a), Some(b), Some(c)) = (a, b, c) {
            let (ax, ay) = all_pts[a];
            let (bx, by) = all_pts[b];
            let (cx, cy) = all_pts[c];
            let centroid = ((ax + bx + cx) / 3.0, (ay + by + cy) / 3.0);
            if !ray_cast(centroid, outer) {
                continue;
            }
            if holes.iter().any(|h| ray_cast(centroid, h)) {
                continue;
            }
            tris.push(TriIdx { a, b, c });
        }
    }

    let mut edge_to_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (ti, tri) in tris.iter().enumerate() {
        let edges = [
            (tri.a.min(tri.b), tri.a.max(tri.b)),
            (tri.b.min(tri.c), tri.b.max(tri.c)),
            (tri.a.min(tri.c), tri.a.max(tri.c)),
        ];
        for e in edges {
            edge_to_tris.entry(e).or_default().push(ti);
        }
    }

    let min_len = threshold * buffer_distance;

    let is_ignored = |e: (usize, usize)| -> bool {
        let adj = edge_to_tris.get(&e).map_or(0, |v| v.len());
        if adj <= 1 {
            return true;
        }
        let (ax, ay) = all_pts[e.0];
        let (bx, by) = all_pts[e.1];
        let len = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt();
        len < min_len
    };

    let edge_midpoint = |e: (usize, usize)| -> Pt {
        let (ax, ay) = all_pts[e.0];
        let (bx, by) = all_pts[e.1];
        ((ax + bx) / 2.0, (ay + by) / 2.0)
    };

    let mut out: Vec<Segment> = Vec::new();
    for tri in &tris {
        let edges = [
            (tri.a.min(tri.b), tri.a.max(tri.b)),
            (tri.b.min(tri.c), tri.b.max(tri.c)),
            (tri.a.min(tri.c), tri.a.max(tri.c)),
        ];
        let internal: Vec<(usize, usize)> =
            edges.iter().copied().filter(|&e| !is_ignored(e)).collect();

        match internal.len() {
            3 => {
                let (ax, ay) = all_pts[tri.a];
                let (bx, by) = all_pts[tri.b];
                let (cx, cy) = all_pts[tri.c];
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

fn ray_cast((px, py): Pt, poly: &[Pt]) -> bool {
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
