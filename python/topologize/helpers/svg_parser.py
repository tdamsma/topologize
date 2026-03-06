"""
SVG Parser - extract and discretize paths from SVG files.

Public API: load_svg(path, sample_distance, scale_factor) -> List[np.ndarray]

Point and Transform are module-level implementation details used by load_svg.
Point stores coordinates with a cached hash (perf). Transform wraps SVG matrix math.
The actual parsing logic lives as closures inside load_svg so that sample_distance
and scale_factor are shared without threading them through every helper signature.
"""

import re
import math
from typing import List, Tuple
from xml.etree import ElementTree as ET
import numpy as np


class Point:
    """Float coordinate point with cached hash for performance."""
    __slots__ = ('x', 'y', '_hash')

    def __init__(self, x: float, y: float):
        # Round to avoid floating point precision issues
        self.x = round(float(x), 10)
        self.y = round(float(y), 10)
        # Cache hash on construction - Point.__hash__ was taking ~57% of runtime
        self._hash = hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def distance_to(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def distance_squared_to(self, other: 'Point') -> float:
        """Squared distance — use for comparisons to avoid sqrt overhead."""
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy


class Transform:
    """Represents an SVG transformation matrix."""

    def __init__(self, a=1, b=0, c=0, d=1, e=0, f=0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def apply(self, x: float, y: float) -> Tuple[float, float]:
        return self.a * x + self.c * y + self.e, self.b * x + self.d * y + self.f

    def compose(self, other: 'Transform') -> 'Transform':
        return Transform(
            a=self.a * other.a + self.c * other.b,
            b=self.b * other.a + self.d * other.b,
            c=self.a * other.c + self.c * other.d,
            d=self.b * other.c + self.d * other.d,
            e=self.a * other.e + self.c * other.f + self.e,
            f=self.b * other.e + self.d * other.f + self.f
        )

    @staticmethod
    def parse(transform_str: str) -> 'Transform':
        if not transform_str:
            return Transform()

        matrix_match = re.search(r'matrix\s*\(\s*([^)]+)\s*\)', transform_str)
        if matrix_match:
            values = [float(x.strip()) for x in matrix_match.group(1).split(',')]
            if len(values) == 6:
                return Transform(*values)

        translate_match = re.search(r'translate\s*\(\s*([^)]+)\s*\)', transform_str)
        if translate_match:
            values = [float(x.strip()) for x in translate_match.group(1).split(',')]
            tx = values[0]
            ty = values[1] if len(values) > 1 else 0
            return Transform(e=tx, f=ty)

        scale_match = re.search(r'scale\s*\(\s*([^)]+)\s*\)', transform_str)
        if scale_match:
            values = [float(x.strip()) for x in scale_match.group(1).split(',')]
            sx = values[0]
            sy = values[1] if len(values) > 1 else sx
            return Transform(a=sx, d=sy)

        return Transform()


def load_svg(path: str, sample_distance: float = 5, scale_factor: float = 1.0) -> List[np.ndarray]:
    """Parse an SVG file and return a list of (N,2) numpy arrays."""

    def sc(coord):
        return coord / scale_factor

    def usc(coord):
        return coord * scale_factor

    def sample_line(p1: Point, p2: Point) -> List[Point]:
        x1, y1 = usc(p1.x), usc(p1.y)
        x2, y2 = usc(p2.x), usc(p2.y)
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length < sample_distance * 0.1:
            return [p1, p2]
        num_samples = max(2, int(math.ceil(length / sample_distance)))
        points = [p1]
        for i in range(1, num_samples):
            t = i / num_samples
            points.append(Point(sc(x1 + t * dx), sc(y1 + t * dy)))
        points.append(p2)
        return points

    def sample_quadratic_bezier(p0: Point, p1: Point, p2: Point) -> List[Point]:
        x0, y0 = usc(p0.x), usc(p0.y)
        x1, y1 = usc(p1.x), usc(p1.y)
        x2, y2 = usc(p2.x), usc(p2.y)
        length = 0
        px, py = x0, y0
        for i in range(1, 11):
            t = i / 10
            x = (1-t)**2 * x0 + 2*(1-t)*t * x1 + t**2 * x2
            y = (1-t)**2 * y0 + 2*(1-t)*t * y1 + t**2 * y2
            length += math.sqrt((x - px)**2 + (y - py)**2)
            px, py = x, y
        if length < sample_distance * 0.1:
            return [p0, p2]
        num_samples = max(2, int(math.ceil(length / sample_distance)))
        points = [p0]
        for i in range(1, num_samples):
            t = i / num_samples
            x = (1-t)**2 * x0 + 2*(1-t)*t * x1 + t**2 * x2
            y = (1-t)**2 * y0 + 2*(1-t)*t * y1 + t**2 * y2
            points.append(Point(sc(x), sc(y)))
        points.append(p2)
        return points

    def sample_cubic_bezier(p0: Point, p1: Point, p2: Point, p3: Point) -> List[Point]:
        x0, y0 = usc(p0.x), usc(p0.y)
        x1, y1 = usc(p1.x), usc(p1.y)
        x2, y2 = usc(p2.x), usc(p2.y)
        x3, y3 = usc(p3.x), usc(p3.y)
        length = 0
        px, py = x0, y0
        for i in range(1, 11):
            t = i / 10
            x = (1-t)**3 * x0 + 3*(1-t)**2*t * x1 + 3*(1-t)*t**2 * x2 + t**3 * x3
            y = (1-t)**3 * y0 + 3*(1-t)**2*t * y1 + 3*(1-t)*t**2 * y2 + t**3 * y3
            length += math.sqrt((x - px)**2 + (y - py)**2)
            px, py = x, y
        if length < sample_distance * 0.1:
            return [p0, p3]
        num_samples = max(2, int(math.ceil(length / sample_distance)))
        points = [p0]
        for i in range(1, num_samples):
            t = i / num_samples
            x = (1-t)**3 * x0 + 3*(1-t)**2*t * x1 + 3*(1-t)*t**2 * x2 + t**3 * x3
            y = (1-t)**3 * y0 + 3*(1-t)**2*t * y1 + 3*(1-t)*t**2 * y2 + t**3 * y3
            points.append(Point(sc(x), sc(y)))
        points.append(p3)
        return points

    def execute_command(cmd, coords, current_pos, subpath_start, current_polyline, polylines):
        is_relative = cmd.islower()
        cmd_upper = cmd.upper()

        if cmd_upper == 'M':
            if current_polyline:
                polylines.append(current_polyline)
            if len(coords) >= 2:
                if is_relative and current_pos:
                    x = sc(usc(current_pos.x) + coords[0])
                    y = sc(usc(current_pos.y) + coords[1])
                else:
                    x, y = sc(coords[0]), sc(coords[1])
                current_pos = Point(x, y)
                subpath_start = current_pos
                current_polyline = [current_pos]
                for i in range(2, len(coords), 2):
                    if is_relative:
                        x = sc(usc(current_pos.x) + coords[i])
                        y = sc(usc(current_pos.y) + coords[i+1])
                    else:
                        x, y = sc(coords[i]), sc(coords[i+1])
                    new_pos = Point(x, y)
                    current_polyline.extend(sample_line(current_pos, new_pos)[1:])
                    current_pos = new_pos

        elif cmd_upper == 'L':
            for i in range(0, len(coords), 2):
                if is_relative and current_pos:
                    x = sc(usc(current_pos.x) + coords[i])
                    y = sc(usc(current_pos.y) + coords[i+1])
                else:
                    x, y = sc(coords[i]), sc(coords[i+1])
                new_pos = Point(x, y)
                if current_pos:
                    current_polyline.extend(sample_line(current_pos, new_pos)[1:])
                current_pos = new_pos

        elif cmd_upper == 'H':
            for x_coord in coords:
                if is_relative and current_pos:
                    x = sc(usc(current_pos.x) + x_coord)
                else:
                    x = sc(x_coord)
                if current_pos:
                    new_pos = Point(x, current_pos.y)
                    current_polyline.extend(sample_line(current_pos, new_pos)[1:])
                    current_pos = new_pos

        elif cmd_upper == 'V':
            for y_coord in coords:
                if is_relative and current_pos:
                    y = sc(usc(current_pos.y) + y_coord)
                else:
                    y = sc(y_coord)
                if current_pos:
                    new_pos = Point(current_pos.x, y)
                    current_polyline.extend(sample_line(current_pos, new_pos)[1:])
                    current_pos = new_pos

        elif cmd_upper == 'Q':
            for i in range(0, len(coords), 4):
                if i + 3 >= len(coords):
                    break
                if is_relative and current_pos:
                    cx = sc(usc(current_pos.x) + coords[i])
                    cy = sc(usc(current_pos.y) + coords[i+1])
                    ex = sc(usc(current_pos.x) + coords[i+2])
                    ey = sc(usc(current_pos.y) + coords[i+3])
                else:
                    cx, cy = sc(coords[i]), sc(coords[i+1])
                    ex, ey = sc(coords[i+2]), sc(coords[i+3])
                if current_pos:
                    end = Point(ex, ey)
                    current_polyline.extend(sample_quadratic_bezier(current_pos, Point(cx, cy), end)[1:])
                    current_pos = end

        elif cmd_upper == 'C':
            for i in range(0, len(coords), 6):
                if i + 5 >= len(coords):
                    break
                if is_relative and current_pos:
                    c1x = sc(usc(current_pos.x) + coords[i])
                    c1y = sc(usc(current_pos.y) + coords[i+1])
                    c2x = sc(usc(current_pos.x) + coords[i+2])
                    c2y = sc(usc(current_pos.y) + coords[i+3])
                    ex  = sc(usc(current_pos.x) + coords[i+4])
                    ey  = sc(usc(current_pos.y) + coords[i+5])
                else:
                    c1x, c1y = sc(coords[i]),   sc(coords[i+1])
                    c2x, c2y = sc(coords[i+2]), sc(coords[i+3])
                    ex,  ey  = sc(coords[i+4]), sc(coords[i+5])
                if current_pos:
                    end = Point(ex, ey)
                    current_polyline.extend(
                        sample_cubic_bezier(current_pos, Point(c1x, c1y), Point(c2x, c2y), end)[1:]
                    )
                    current_pos = end

        elif cmd_upper == 'Z':
            if current_pos and subpath_start and current_pos != subpath_start:
                current_polyline.extend(sample_line(current_pos, subpath_start)[1:])
                current_pos = subpath_start

        return current_pos, subpath_start, current_polyline

    def parse_path_data(d: str, transform: Transform) -> List[List[Point]]:
        tokens = re.findall(r'([MLHVQCZmlhvqcz])|(-?\d+\.?\d*)', d)
        polylines = []
        current_pos = None
        subpath_start = None
        current_polyline = []
        command = None
        coords = []

        for token, number in tokens:
            if token:
                if command and coords:
                    current_pos, subpath_start, current_polyline = execute_command(
                        command, coords, current_pos, subpath_start, current_polyline, polylines
                    )
                    coords = []
                elif command and command.upper() == 'Z':
                    current_pos, subpath_start, current_polyline = execute_command(
                        command, [], current_pos, subpath_start, current_polyline, polylines
                    )
                command = token
            elif number:
                coords.append(float(number))

        if command and coords:
            current_pos, subpath_start, current_polyline = execute_command(
                command, coords, current_pos, subpath_start, current_polyline, polylines
            )
        elif command and command.upper() == 'Z':
            current_pos, subpath_start, current_polyline = execute_command(
                command, [], current_pos, subpath_start, current_polyline, polylines
            )

        if current_polyline:
            polylines.append(current_polyline)

        is_identity = (transform.a == 1 and transform.b == 0 and transform.c == 0
                       and transform.d == 1 and transform.e == 0 and transform.f == 0)
        if not is_identity:
            polylines = [
                [Point(sc(x_new), sc(y_new))
                 for p in pl
                 for x_new, y_new in [transform.apply(usc(p.x), usc(p.y))]]
                for pl in polylines
            ]

        return polylines

    def extract_paths_recursive(element, polylines, parent_transform=None, clip_path=None):
        tag = element.tag.split('}')[-1]
        if tag in ['defs', 'clipPath', 'mask', 'pattern', 'symbol']:
            return

        element_clip = element.get('clip-path')
        if element_clip:
            clip_match = re.match(r'url\(#(.+)\)', element_clip)
            if clip_match:
                clip_path = clip_match.group(1)

        current_transform = parent_transform or Transform()
        element_transform_str = element.get('transform')
        if element_transform_str:
            current_transform = current_transform.compose(Transform.parse(element_transform_str))

        if element.tag.endswith('path') or element.tag == '{http://www.w3.org/2000/svg}path':
            d = element.get('d', '')
            if d:
                polylines.extend(parse_path_data(d, current_transform))

        for child in element:
            extract_paths_recursive(child, polylines, current_transform, clip_path)

    tree = ET.parse(path)
    root = tree.getroot()
    point_lists: List[List[Point]] = []
    extract_paths_recursive(root, point_lists)
    return [np.array([[p.x, p.y] for p in pl]) for pl in point_lists if len(pl) >= 2]
