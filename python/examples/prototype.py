# %%
import numpy as np
import shapely
import triangle as tr
from collections import defaultdict
import plotly.graph_objects as go


# %% --- algorithm ---

def _to_geom(c):
    """(N,2) array → LinearRing if closed, LineString otherwise."""
    if np.allclose(c[0], c[-1], atol=1e-8):
        return shapely.LinearRing(c)
    return shapely.LineString(c)


def _midpoint_segments(polygon, buffer_distance, threshold=1.9):
    """Constrained Delaunay → list of ((x1,y1),(x2,y2)) midpoint-graph edges."""
    all_pts, all_segs, hole_pts = [], [], []
    offset = 0
    for ring in [polygon.exterior, *polygon.interiors]:
        coords = np.array(ring.coords[:-1])
        n = len(coords)
        all_pts.append(coords)
        all_segs += [[offset + i, offset + (i + 1) % n] for i in range(n)]
        offset += n
    for interior in polygon.interiors:
        hole_pts.append(shapely.Polygon(interior).representative_point().coords[0])

    pts = np.concatenate(all_pts)
    segs = np.array(all_segs)
    tri_input = {'vertices': pts, 'segments': segs}
    if hole_pts:
        tri_input['holes'] = np.array(hole_pts)

    result = tr.triangulate(tri_input, 'p')
    verts = result['vertices']
    tris = result['triangles']

    edge_to_tris = defaultdict(list)
    for ti, tri in enumerate(tris):
        for j in range(3):
            edge = tuple(sorted([tri[j], tri[(j + 1) % 3]]))
            edge_to_tris[edge].append(ti)

    def elen(e):
        a, b = verts[e[0]], verts[e[1]]
        return np.hypot(b[0] - a[0], b[1] - a[1])

    def emid(e):
        a, b = verts[e[0]], verts[e[1]]
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

    ignored = {
        e for e, ids in edge_to_tris.items()
        if len(ids) == 1 or elen(e) < threshold * buffer_distance
    }

    out = []
    for tri in tris:
        edges = [tuple(sorted([tri[j], tri[(j + 1) % 3]])) for j in range(3)]
        internal = [e for e in edges if e not in ignored]
        mids = [emid(e) for e in internal]
        if len(internal) == 3:
            centroid = tuple(verts[tri].mean(axis=0))
            for m in mids:
                out.append((m, centroid))
        else:
            for i in range(len(mids)):
                for j in range(i + 1, len(mids)):
                    out.append((mids[i], mids[j]))
    return out


def _extract_chains(segments):
    """Convert (p1,p2) edge list into maximal non-branching chains (numpy arrays)."""
    pt_idx, pt_list = {}, []

    def get_idx(p):
        key = (round(p[0], 8), round(p[1], 8))
        if key not in pt_idx:
            pt_idx[key] = len(pt_list)
            pt_list.append(key)
        return pt_idx[key]

    adj = defaultdict(set)
    for p1, p2 in segments:
        a, b = get_idx(p1), get_idx(p2)
        if a != b:
            adj[a].add(b)
            adj[b].add(a)

    visited = set()
    chains = []

    def traverse(start, nxt):
        key = (min(start, nxt), max(start, nxt))
        if key in visited:
            return None
        path = [start, nxt]
        visited.add(key)
        prev, cur = start, nxt
        while len(adj[cur]) == 2:
            nb = next(iter(adj[cur] - {prev}), None)
            if nb is None:
                break
            k = (min(cur, nb), max(cur, nb))
            if k in visited:
                break
            visited.add(k)
            path.append(nb)
            prev, cur = cur, nb
        return np.array([pt_list[i] for i in path])

    # junctions and endpoints first, then any remaining cycles
    for start in list(adj):
        if len(adj[start]) != 2:
            for nxt in list(adj[start]):
                chain = traverse(start, nxt)
                if chain is not None:
                    chains.append(chain)
    for start in list(adj):
        for nxt in list(adj[start]):
            chain = traverse(start, nxt)
            if chain is not None:
                chains.append(chain)

    return chains


def extract_centerline_python(curves, buffer_distance):
    """
    curves:           list of (N,2) numpy arrays
    buffer_distance:  float
    returns:          list of (M,2) numpy arrays, one per continuous segment
    """
    o = shapely.unary_union([_to_geom(c) for c in curves]).buffer(
        buffer_distance, join_style='round', cap_style='square'
    )
    polygons = list(o.geoms) if o.geom_type == 'MultiPolygon' else [o]
    segments = []
    for poly in polygons:
        segments.extend(_midpoint_segments(poly, buffer_distance))
    return _extract_chains(segments)




# %% --- plot ---
def plot_curves_and_chains(curves, chains, buffer_distance):
    # recompute buffer boundary for display
    _o = shapely.unary_union([_to_geom(c) for c in curves]).buffer(
        buffer_distance, join_style='round', cap_style='square'
    )
    _polygons = list(_o.geoms) if _o.geom_type == 'MultiPolygon' else [_o]

    COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3',
            '#a65628', '#f781bf', '#333333', '#e6ab02', '#66c2a5']

    traces = []
    _nan_row = np.full((1, 2), np.nan)

    _buf_rings = [ring for poly in _polygons for ring in [poly.exterior, *poly.interiors]]
    _buf_pts = np.concatenate([
        np.vstack([np.array(ring.coords), _nan_row]) for ring in _buf_rings
    ])
    traces.append(go.Scatter(x=_buf_pts[:, 0], y=_buf_pts[:, 1], mode='lines',
                            name='buffer', line=dict(color='lightgray', width=1)))

    _input_pts = np.concatenate([np.vstack([c, _nan_row]) for c in curves])
    traces.append(go.Scatter(x=_input_pts[:, 0], y=_input_pts[:, 1], mode='lines',
                            name='input', line=dict(color='lightgray', width=5)))

    for i, chain in enumerate(chains):
        traces.append(go.Scatter(x=chain[:, 0], y=chain[:, 1], mode='lines',
                                name=f'segment {i}',
                                line=dict(color=COLORS[i % len(COLORS)], width=2.5)))

    fig = go.Figure(traces)
    fig.update_layout(yaxis_scaleanchor='x', width=900, height=600)
    fig.show()

# %% input
if __name__ == "__main__":
    # open polyline p
    p_pts = np.array([[0.0, 2.0], [10.0, 0.0], [5.0, 0.1], [5.0, 5.0]])

    # open polyline p2 (resampled to 24 points)
    _p2 = shapely.LineString([[0, 6], [1, 4], [2, 0]])
    p2_pts = np.array(shapely.LineString(_p2.interpolate(np.linspace(0, 1, 24), normalized=True)).coords)

    # closed ring: circle tangent to p at (10, 0)
    _cc, _cr = np.array([13.0, 1.0]), np.linalg.norm(np.array([13.0, 1.0]) - [10.0, 0.0])
    _a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    _circle_open = _cc + _cr * np.column_stack([np.cos(_a), np.sin(_a)])
    circle_pts = np.vstack([_circle_open, _circle_open[:1]])   # close the ring

    # closed ring: 5-point star
    _sc = np.array([5.0, -5.0])
    _sa = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 10, endpoint=False)
    _sr = np.tile([8.0, 1.8], 5)
    _star_open = _sc + np.column_stack([_sr * np.cos(_sa), _sr * np.sin(_sa)])
    star_pts = np.vstack([_star_open, _star_open[:1]])          # close the ring

    curves = [p_pts, p2_pts, circle_pts, star_pts]

    # %%
    buffer_distance = 0.6
    chains = extract_centerline_python(curves, buffer_distance)
    plot_curves_and_chains(curves, chains, buffer_distance)
# %%
