"""Shared benchmark inputs."""
from pathlib import Path

import numpy as np
import shapely

# open polyline
p_pts = np.array([[0.0, 2.0], [10.0, 0.0], [5.0, 0.1], [5.0, 5.0]])

# open polyline (resampled to 24 points)
_p2 = shapely.LineString([[0, 6], [1, 4], [2, 0]])
p2_pts = np.array(shapely.LineString(_p2.interpolate(np.linspace(0, 1, 24), normalized=True)).coords)

# closed ring: circle tangent to p at (10, 0)
_cc = np.array([13.0, 1.0])
_cr = np.linalg.norm(_cc - [10.0, 0.0])
_a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
_circle_open = _cc + _cr * np.column_stack([np.cos(_a), np.sin(_a)])
circle_pts = np.vstack([_circle_open, _circle_open[:1]])

# closed ring: 5-point star
_sc = np.array([5.0, -5.0])
_sa = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 10, endpoint=False)
_sr = np.tile([8.0, 1.8], 5)
_star_open = _sc + np.column_stack([_sr * np.cos(_sa), _sr * np.sin(_sa)])
star_pts = np.vstack([_star_open, _star_open[:1]])

SIMPLE_CURVES = [p_pts, p2_pts, circle_pts, star_pts]
SIMPLE_BUFFERS = [0.3, 0.6, 1.2]

SVG_PATH = Path(__file__).parent.parent.parent / "svg-fixer" / "input.svg"
SVG_BUFFERS = [5, 10, 20, 50]
