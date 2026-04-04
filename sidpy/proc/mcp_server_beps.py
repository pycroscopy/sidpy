"""
MCP server utilities for BEPS loop and SHO fitting.

This module wraps the existing ``SidpyFitter`` workflows in MCP-friendly
tools so an LLM client can fit nested-array BEPS loop data and SHO response
data without re-implementing the fitting logic.

The MCP runtime is optional. Importing this module does not require ``mcp``,
but running the server does.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import sidpy as sid
from scipy.spatial import ConvexHull
from scipy.special import erf

from .fitter_refactor import SidpyFitter

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - optional runtime dependency
    FastMCP = None


LOOP_PARAMETER_LABELS = [
    "offset",
    "amplitude",
    "coercive_left",
    "coercive_right",
    "slope",
    "branch_scale_1",
    "branch_scale_2",
    "branch_scale_3",
    "branch_scale_4",
]

SHO_PARAMETER_LABELS = ["amplitude", "resonance_frequency", "quality_factor", "phase"]


def loop_fit_function(vdc: Sequence[float], *coef_vec: float) -> np.ndarray:
    """Nine-parameter loop model used by the BEPS fitter tests."""
    vdc = np.asarray(vdc).squeeze()
    a = coef_vec[:5]
    b = coef_vec[5:]
    d = 1000

    v1 = np.asarray(vdc[: int(len(vdc) / 2)])
    v2 = np.asarray(vdc[int(len(vdc) / 2) :])

    g1 = (b[1] - b[0]) / 2 * (erf((v1 - a[2]) * d) + 1) + b[0]
    g2 = (b[3] - b[2]) / 2 * (erf((v2 - a[3]) * d) + 1) + b[2]

    y1 = (g1 * erf((v1 - a[2]) / g1) + b[0]) / (b[0] + b[1])
    y2 = (g2 * erf((v2 - a[3]) / g2) + b[2]) / (b[2] + b[3])

    f1 = a[0] + a[1] * y1 + a[4] * v1
    f2 = a[0] + a[1] * y2 + a[4] * v2
    return np.hstack((f1, f2)).squeeze()


def calculate_loop_centroid(vdc: Sequence[float], loop_vals: Sequence[float]) -> tuple[tuple[float, float], float]:
    """Calculate the polygon centroid for one unfolded loop."""
    vdc = np.squeeze(np.asarray(vdc))
    loop_vals = np.squeeze(np.asarray(loop_vals))
    num_steps = vdc.size

    x_vals = np.zeros(num_steps - 1)
    y_vals = np.zeros(num_steps - 1)
    area_vals = np.zeros(num_steps - 1)

    for index in range(num_steps - 1):
        x_i = vdc[index]
        x_i1 = vdc[index + 1]
        y_i = loop_vals[index]
        y_i1 = loop_vals[index + 1]

        x_vals[index] = (x_i + x_i1) * (x_i * y_i1 - x_i1 * y_i)
        y_vals[index] = (y_i + y_i1) * (x_i * y_i1 - x_i1 * y_i)
        area_vals[index] = x_i * y_i1 - x_i1 * y_i

    area = 0.5 * np.sum(area_vals)
    cent_x = (1.0 / (6.0 * area)) * np.sum(x_vals)
    cent_y = (1.0 / (6.0 * area)) * np.sum(y_vals)
    return (cent_x, cent_y), area


def generate_guess(vdc: Sequence[float], pr_vec: Sequence[float], show_plots: bool = False) -> np.ndarray:
    """
    Generate the initial BEPS loop parameter guess.

    This matches the tested heuristic in ``tests/proc/test_fitter.py``.
    ``show_plots`` is accepted for API compatibility but not used here.
    """
    del show_plots

    points = np.transpose(np.array([np.squeeze(vdc), pr_vec]))
    geom_centroid, _ = calculate_loop_centroid(points[:, 0], points[:, 1])
    hull = ConvexHull(points)

    def find_intersection(a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]):
        def ccw(p_a, p_b, p_c):
            return (p_c[1] - p_a[1]) * (p_b[0] - p_a[0]) > (p_b[1] - p_a[1]) * (p_c[0] - p_a[0])

        def line(p1, p2):
            coeff_a = p1[1] - p2[1]
            coeff_b = p2[0] - p1[0]
            coeff_c = p1[0] * p2[1] - p2[0] * p1[1]
            return coeff_a, coeff_b, -coeff_c

        def intersection(line_1, line_2):
            det = line_1[0] * line_2[1] - line_1[1] * line_2[0]
            det_x = line_1[2] * line_2[1] - line_1[1] * line_2[2]
            det_y = line_1[0] * line_2[2] - line_1[2] * line_2[0]
            if det == 0:
                return None
            return det_x / det, det_y / det

        intersects = (ccw(a, c, d) is not ccw(b, c, d)) and (ccw(a, b, c) is not ccw(a, b, d))
        if not intersects:
            return None
        return intersection(line(a, b), line(c, d))

    outline_1 = np.zeros((hull.simplices.shape[0], 2), dtype=float)
    outline_2 = np.zeros((hull.simplices.shape[0], 2), dtype=float)
    for index, pair in enumerate(hull.simplices):
        outline_1[index, :] = points[pair[0]]
        outline_2[index, :] = points[pair[1]]

    y_intersections = []
    for pair in range(outline_1.shape[0]):
        point = find_intersection(
            outline_1[pair],
            outline_2[pair],
            [geom_centroid[0], hull.min_bound[1]],
            [geom_centroid[0], hull.max_bound[1]],
        )
        if point is not None:
            y_intersections.append(point)

    x_intersections = []
    for pair in range(outline_1.shape[0]):
        point = find_intersection(
            outline_1[pair],
            outline_2[pair],
            [hull.min_bound[0], geom_centroid[1]],
            [hull.max_bound[0], geom_centroid[1]],
        )
        if point is not None:
            x_intersections.append(point)

    if len(y_intersections) < 2:
        min_y_intercept = min(pr_vec)
        max_y_intercept = max(pr_vec)
    else:
        min_y_intercept = min(y_intersections[0][1], y_intersections[1][1])
        max_y_intercept = max(y_intersections[0][1], y_intersections[1][1])

    if len(x_intersections) < 2:
        min_x_intercept = min(vdc) / 2.0
        max_x_intercept = max(vdc) / 2.0
    else:
        min_x_intercept = min(x_intersections[0][0], x_intersections[1][0])
        max_x_intercept = max(x_intersections[0][0], x_intersections[1][0])

    init_guess = np.zeros(shape=9)
    init_guess[0] = min_y_intercept
    init_guess[1] = max_y_intercept - min_y_intercept
    init_guess[2] = min_x_intercept
    init_guess[3] = max_x_intercept
    init_guess[4] = 0
    init_guess[5:] = 2
    return init_guess


def SHO_fit_flattened(wvec: Sequence[float], *params: float) -> np.ndarray:
    """Flattened complex SHO response used by the SHO fitter tests."""
    amp, w_0, quality_factor, phase = params[0], params[1], params[2], params[3]
    func = amp * np.exp(1j * phase) * w_0**2 / (wvec**2 - 1j * wvec * w_0 / quality_factor - w_0**2)
    return np.hstack([np.real(func), np.imag(func)])


def sho_guess_fn(freq_vec: Sequence[float], ydata: Sequence[complex]) -> list[float]:
    """Initial guess heuristic for SHO fitting."""
    ydata = np.asarray(ydata)
    amp_guess = np.abs(ydata)[np.argmax(np.abs(ydata))]
    phase_guess = np.angle(ydata)[np.argmax(np.abs(ydata))]
    w_guess = np.asarray(freq_vec)[np.argmax(np.abs(ydata))]

    q_values = [5, 10, 20, 50, 100, 200, 500]
    err_vals = []
    for q_val in q_values:
        p_test = [amp_guess / q_val, w_guess, q_val, phase_guess]
        func_out = SHO_fit_flattened(freq_vec, *p_test)
        complex_output = func_out[: len(func_out) // 2] + 1j * func_out[len(func_out) // 2 :]
        amp_output = np.abs(complex_output)
        err_vals.append(np.mean((amp_output - np.abs(ydata)) ** 2))

    q_guess = q_values[int(np.argmin(err_vals))]
    return [amp_guess / q_guess, w_guess, q_guess, phase_guess]


def _as_builtin(value: Any) -> Any:
    """Convert numpy-heavy structures to JSON-serializable builtins."""
    if isinstance(value, dict):
        return {str(key): _as_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_dataset(
    data: Sequence[Any],
    spectral_axis: Sequence[float],
    spectral_name: str,
    dataset_name: str,
    *,
    spectral_quantity: Optional[str] = None,
    spectral_units: Optional[str] = None,
) -> sid.Dataset:
    """Create a sidpy.Dataset assuming the last axis is the fitted spectral axis."""
    array = np.asarray(data)
    spectral_axis = np.asarray(spectral_axis)

    if array.ndim < 1:
        raise ValueError("Input data must have at least one dimension.")
    if array.shape[-1] != spectral_axis.size:
        raise ValueError(
            "The spectral axis length must match the size of the last data axis. "
            f"Received axis length {spectral_axis.size} and last axis {array.shape[-1]}."
        )

    dataset = sid.Dataset.from_array(array, name=dataset_name)
    for dim in range(array.ndim - 1):
        dataset.set_dimension(
            dim,
            sid.Dimension(np.arange(array.shape[dim]), name=f"dim_{dim}", dimension_type="spatial"),
        )
    dataset.set_dimension(
        array.ndim - 1,
        sid.Dimension(
            spectral_axis,
            name=spectral_name,
            quantity=spectral_quantity or spectral_name,
            units=spectral_units or "a.u.",
            dimension_type="spectral",
        ),
    )
    return dataset


def _package_result(result: Any) -> Dict[str, Any]:
    """Normalize fitter outputs into a JSON-friendly payload."""
    if isinstance(result, tuple):
        params_dataset, cov_dataset = result
    else:
        params_dataset, cov_dataset = result, None

    payload = {
        "parameters": np.asarray(params_dataset).tolist(),
        "parameter_shape": list(params_dataset.shape),
        "parameter_metadata": _as_builtin(params_dataset.metadata),
    }
    if cov_dataset is not None:
        payload["covariance"] = np.asarray(cov_dataset).tolist()
        payload["covariance_shape"] = list(cov_dataset.shape)
        payload["covariance_metadata"] = _as_builtin(cov_dataset.metadata)
    return payload


def fit_beps_loops(
    data: Sequence[Any],
    dc_voltage: Sequence[float],
    *,
    use_kmeans: bool = False,
    n_clusters: int = 6,
    return_cov: bool = False,
    loss: str = "linear",
    f_scale: float = 1.0,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None,
    chunks: Any = "auto",
    dataset_name: str = "beps_loop_data",
) -> Dict[str, Any]:
    """Fit BEPS loop data where the last axis contains the loop trace."""
    dataset = _build_dataset(
        data,
        dc_voltage,
        "DC Offset",
        dataset_name,
        spectral_quantity="Voltage",
        spectral_units="Volts",
    )
    fitter = SidpyFitter(
        dataset,
        loop_fit_function,
        generate_guess,
        ind_dims=(dataset.ndim - 1,),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    fitter.setup_calc(chunks=chunks)
    result = fitter.do_fit(
        use_kmeans=use_kmeans,
        n_clusters=n_clusters,
        return_cov=return_cov,
        loss=loss,
        f_scale=f_scale,
        fit_parameter_labels=LOOP_PARAMETER_LABELS,
    )
    payload = _package_result(result)
    payload["parameter_labels"] = LOOP_PARAMETER_LABELS
    payload["fit_kind"] = "beps_loop"
    return payload


def fit_sho_response(
    real_data: Sequence[Any],
    frequency: Sequence[float],
    *,
    imag_data: Optional[Sequence[Any]] = None,
    use_kmeans: bool = False,
    n_clusters: int = 10,
    return_cov: bool = False,
    loss: str = "linear",
    f_scale: float = 1.0,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None,
    chunks: Any = "auto",
    dataset_name: str = "sho_response_data",
) -> Dict[str, Any]:
    """Fit SHO response data where the last axis contains the frequency sweep."""
    real_array = np.asarray(real_data)
    if imag_data is None:
        if not np.iscomplexobj(real_array):
            raise ValueError("Pass complex-valued data or provide imag_data separately for SHO fitting.")
        complex_array = real_array
    else:
        imag_array = np.asarray(imag_data)
        if imag_array.shape != real_array.shape:
            raise ValueError("real_data and imag_data must have the same shape.")
        complex_array = real_array + 1j * imag_array

    dataset = _build_dataset(
        complex_array,
        frequency,
        "Frequency",
        dataset_name,
        spectral_quantity="Frequency",
        spectral_units="Hz",
    )
    fitter = SidpyFitter(
        dataset,
        SHO_fit_flattened,
        sho_guess_fn,
        ind_dims=(dataset.ndim - 1,),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    fitter.setup_calc(chunks=chunks)
    result = fitter.do_fit(
        use_kmeans=use_kmeans,
        n_clusters=n_clusters,
        return_cov=return_cov,
        loss=loss,
        f_scale=f_scale,
        fit_parameter_labels=SHO_PARAMETER_LABELS,
    )
    payload = _package_result(result)
    payload["parameter_labels"] = SHO_PARAMETER_LABELS
    payload["fit_kind"] = "sho"
    return payload


def create_mcp_server(server_name: str = "sidpy-beps-fitting"):
    """Create an MCP server exposing the BEPS loop and SHO fitting tools."""
    if FastMCP is None:  # pragma: no cover - optional runtime dependency
        raise ImportError("The 'mcp' package is required to create the BEPS MCP server.")

    server = FastMCP(server_name)

    @server.tool()
    def fit_beps_loops_tool(
        data: Sequence[Any],
        dc_voltage: Sequence[float],
        use_kmeans: bool = False,
        n_clusters: int = 6,
        return_cov: bool = False,
        loss: str = "linear",
        f_scale: float = 1.0,
        lower_bounds: Optional[Sequence[float]] = None,
        upper_bounds: Optional[Sequence[float]] = None,
        dataset_name: str = "beps_loop_data",
    ) -> Dict[str, Any]:
        """Fit BEPS loops from nested arrays with the spectral axis in the last dimension."""
        return fit_beps_loops(
            data,
            dc_voltage,
            use_kmeans=use_kmeans,
            n_clusters=n_clusters,
            return_cov=return_cov,
            loss=loss,
            f_scale=f_scale,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            dataset_name=dataset_name,
        )

    @server.tool()
    def fit_sho_response_tool(
        real_data: Sequence[Any],
        frequency: Sequence[float],
        imag_data: Optional[Sequence[Any]] = None,
        use_kmeans: bool = False,
        n_clusters: int = 10,
        return_cov: bool = False,
        loss: str = "linear",
        f_scale: float = 1.0,
        lower_bounds: Optional[Sequence[float]] = None,
        upper_bounds: Optional[Sequence[float]] = None,
        dataset_name: str = "sho_response_data",
    ) -> Dict[str, Any]:
        """Fit SHO data from nested real and imaginary arrays or a complex nested array."""
        return fit_sho_response(
            real_data,
            frequency,
            imag_data=imag_data,
            use_kmeans=use_kmeans,
            n_clusters=n_clusters,
            return_cov=return_cov,
            loss=loss,
            f_scale=f_scale,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            dataset_name=dataset_name,
        )

    return server


if FastMCP is not None:  # pragma: no cover - optional runtime dependency
    mcp = create_mcp_server()
else:  # pragma: no cover - optional runtime dependency
    mcp = None


def main():
    """Run the MCP server over the default transport."""
    if mcp is None:  # pragma: no cover - optional runtime dependency
        raise ImportError("The 'mcp' package is required to run the BEPS MCP server.")
    mcp.run()


__all__ = [
    "LOOP_PARAMETER_LABELS",
    "SHO_PARAMETER_LABELS",
    "SHO_fit_flattened",
    "calculate_loop_centroid",
    "create_mcp_server",
    "fit_beps_loops",
    "fit_sho_response",
    "generate_guess",
    "loop_fit_function",
    "main",
    "sho_guess_fn",
]
