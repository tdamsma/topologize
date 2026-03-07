import numpy as np


def load_svg(path: str, sample_distance: float = 5.0) -> list:
    """Load an SVG file and return sampled polylines as (N, 2) numpy arrays.

    Requires the library to be built with the ``svg`` Cargo feature (enabled
    by default in the provided pyproject.toml).
    """
    try:
        from topologize._internal import load_svg as _load_svg
    except ImportError:
        raise RuntimeError(
            "load_svg requires the 'svg' Cargo feature. "
            "Rebuild with: maturin develop --features svg"
        )
    raw = _load_svg(path, sample_distance)
    return [np.array(curve) for curve in raw if len(curve) >= 2]


__all__ = ["load_svg"]
