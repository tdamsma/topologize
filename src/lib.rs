//! PyO3 entry point — exposes topologize to Python as `topologize._internal`.

mod graph;
mod inflate;
mod python;
mod skeleton;
mod skeleton_voronoi;

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_internal")]
fn _internal(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python::topologize, m)?)?;
    Ok(())
}
