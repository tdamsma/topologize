//! SVG loading via usvg: parses and normalises all SVG path commands
//! (including S, T, A, arcs, transforms, groups) and samples them into
//! polylines.

use pyo3::prelude::*;
use usvg::tiny_skia_path::PathSegment;

type Pt = (f64, f64);

fn apply(t: usvg::Transform, x: f32, y: f32) -> Pt {
    (
        (t.sx * x + t.kx * y + t.tx) as f64,
        (t.ky * x + t.sy * y + t.ty) as f64,
    )
}

fn cubic_at(p0: Pt, p1: Pt, p2: Pt, p3: Pt, t: f64) -> Pt {
    let u = 1.0 - t;
    (
        u * u * u * p0.0 + 3.0 * u * u * t * p1.0 + 3.0 * u * t * t * p2.0 + t * t * t * p3.0,
        u * u * u * p0.1 + 3.0 * u * u * t * p1.1 + 3.0 * u * t * t * p2.1 + t * t * t * p3.1,
    )
}

fn push_cubic(pts: &mut Vec<Pt>, p0: Pt, p1: Pt, p2: Pt, p3: Pt, step: f64) {
    let mut len = 0.0f64;
    let mut prev = p0;
    for i in 1..=10 {
        let q = cubic_at(p0, p1, p2, p3, i as f64 / 10.0);
        len += ((q.0 - prev.0).powi(2) + (q.1 - prev.1).powi(2)).sqrt();
        prev = q;
    }
    let n = ((len / step).ceil() as usize).max(1);
    for i in 1..=n {
        pts.push(cubic_at(p0, p1, p2, p3, i as f64 / n as f64));
    }
}

fn push_line(pts: &mut Vec<Pt>, p0: Pt, p1: Pt, step: f64) {
    let dx = p1.0 - p0.0;
    let dy = p1.1 - p0.1;
    let len = (dx * dx + dy * dy).sqrt();
    if len < step * 0.01 {
        pts.push(p1);
        return;
    }
    let n = ((len / step).ceil() as usize).max(1);
    for i in 1..=n {
        let t = i as f64 / n as f64;
        pts.push((p0.0 + t * dx, p0.1 + t * dy));
    }
}

fn collect_path(path: &usvg::Path, step: f64, curves: &mut Vec<Vec<Pt>>) {
    let t = path.abs_transform();
    let mut current: Vec<Pt> = Vec::new();
    let mut pos = (0.0f64, 0.0f64);
    let mut subpath_start = (0.0f64, 0.0f64);

    for seg in path.data().segments() {
        match seg {
            PathSegment::MoveTo(p) => {
                if current.len() >= 2 {
                    curves.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
                let end = apply(t, p.x, p.y);
                pos = end;
                subpath_start = end;
                current.push(end);
            }
            PathSegment::LineTo(p) => {
                let end = apply(t, p.x, p.y);
                push_line(&mut current, pos, end, step);
                pos = end;
            }
            PathSegment::QuadTo(cp, p) => {
                let c = apply(t, cp.x, cp.y);
                let end = apply(t, p.x, p.y);
                // Elevate quadratic to cubic
                let c1 = (pos.0 + 2.0 / 3.0 * (c.0 - pos.0), pos.1 + 2.0 / 3.0 * (c.1 - pos.1));
                let c2 = (end.0 + 2.0 / 3.0 * (c.0 - end.0), end.1 + 2.0 / 3.0 * (c.1 - end.1));
                push_cubic(&mut current, pos, c1, c2, end, step);
                pos = end;
            }
            PathSegment::CubicTo(cp1, cp2, p) => {
                let c1 = apply(t, cp1.x, cp1.y);
                let c2 = apply(t, cp2.x, cp2.y);
                let end = apply(t, p.x, p.y);
                push_cubic(&mut current, pos, c1, c2, end, step);
                pos = end;
            }
            PathSegment::Close => {
                let dx = subpath_start.0 - pos.0;
                let dy = subpath_start.1 - pos.1;
                if dx * dx + dy * dy > 1e-20 {
                    push_line(&mut current, pos, subpath_start, step);
                }
                pos = subpath_start;
            }
        }
    }
    if current.len() >= 2 {
        curves.push(current);
    }
}

fn traverse(group: &usvg::Group, step: f64, curves: &mut Vec<Vec<Pt>>) {
    for child in group.children() {
        match child {
            usvg::Node::Path(path) => collect_path(path, step, curves),
            usvg::Node::Group(g) => traverse(g, step, curves),
            _ => {}
        }
    }
}

/// Load an SVG file and return sampled polylines as lists of (x, y) tuples.
///
/// Parameters
/// ----------
/// path : str
///     Path to the SVG file.
/// sample_distance : float, default 5.0
///     Maximum distance between consecutive sample points (in SVG user units).
#[pyfunction]
#[pyo3(signature = (path, sample_distance = 5.0))]
pub fn load_svg(path: &str, sample_distance: f64) -> PyResult<Vec<Vec<Pt>>> {
    let data = std::fs::read(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let opt = usvg::Options::default();
    let tree = usvg::Tree::from_data(&data, &opt)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let mut curves: Vec<Vec<Pt>> = Vec::new();
    traverse(tree.root(), sample_distance, &mut curves);
    Ok(curves)
}
