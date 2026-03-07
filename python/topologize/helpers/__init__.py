import numpy as np


def load_svg(path: str, sample_distance: float = 5.0) -> list:
    """Load an SVG file and return sampled polylines as (N, 2) numpy arrays."""
    from topologize._internal import load_svg as _load_svg
    raw = _load_svg(path, sample_distance)
    return [np.array(curve) for curve in raw if len(curve) >= 2]


__all__ = ["load_svg"]
