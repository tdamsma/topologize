//! PyO3 entry point — exposes topologize to Python as `topologize._internal`.

mod graph;
mod inflate;
mod python;
mod skeleton_cdt;
mod subdivide;

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_internal")]
fn _internal(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python::inflate_curves, m)?)?;
    m.add_function(wrap_pyfunction!(python::triangulate_curves, m)?)?;
    m.add_function(wrap_pyfunction!(python::topologize, m)?)?;
    m.add_function(wrap_pyfunction!(python::topologize_batch, m)?)?;
    Ok(())
}
