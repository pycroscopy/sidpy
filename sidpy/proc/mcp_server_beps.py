"""
MCP server utilities for BEPS loop and SHO fitting.

This module wraps the existing ``SidpyFitterRefactor`` workflows in MCP-friendly
tools so an LLM client can fit nested-array BEPS loop data and SHO response
data without re-implementing the fitting logic.

The MCP runtime is optional. Importing this module does not require ``mcp``,
but running the server does.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence
from uuid import uuid4

import numpy as np
import sidpy as sid
from scipy.spatial import ConvexHull
from scipy.special import erf

from .fitter_refactor import SidpyFitterRefactor

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
DATASET_REGISTRY: Dict[str, sid.Dataset] = {}
WORKFLOW_EXAMPLES: Dict[str, list[Dict[str, Any]]] = {
    "analysis": [
        {
            "name": "fit_beps_dataset",
            "goal": (
                "Read a BEPS HDF5/NSID file with SciFiReaders, fit a BEPS loop map "
                "and an SHO slice, then save the fit-parameter maps as sidpy.Datasets."
            ),
            "inputs": {
                "file_path": "/path/to/PTO_5x5.h5",
                "channel_name": "Channel_000",
                "beps_frequency_index": 23,
                "beps_cycle_index": 0,
                "sho_dc_index": 49,
                "sho_cycle_index": 1,
                "use_kmeans": True,
                "n_clusters": 4,
                "beps_dataset_name": "beps_fit_parameters",
                "sho_dataset_name": "sho_fit_parameters",
            },
            "setup": [
                {
                    "kind": "external",
                    "tool": "SciFiReaders.NSIDReader",
                    "arguments": {
                        "file_path": "{{file_path}}",
                    },
                    "assign": "reader",
                    "notes": (
                        "Read the file outside MCP, then extract {{channel_name}} into a "
                        "sidpy.Dataset before calling the fit tools."
                    ),
                },
                {
                    "kind": "external",
                    "tool": "reader.read()",
                    "arguments": {},
                    "assign": "channel_data",
                    "notes": (
                        "Use the loaded dataset to build two slices: "
                        "BEPS data = data[:, :, beps_frequency_index, :, beps_cycle_index]; "
                        "SHO data = data[:, :, :, sho_dc_index, sho_cycle_index]."
                    ),
                },
            ],
            "steps": [
                {
                    "tool": "fit_beps_loops_tool",
                    "arguments": {
                        "data": "{{beps_data}}",
                        "dc_voltage": "{{dc_voltage}}",
                        "use_kmeans": "{{use_kmeans}}",
                        "n_clusters": "{{n_clusters}}",
                        "return_cov": False,
                        "loss": "linear",
                        "dataset_name": "beps_loop_fit",
                    },
                    "notes": "Fit the BEPS loop slice first so the loop map is captured cleanly.",
                },
                {
                    "tool": "create_dataset_tool",
                    "arguments": {
                        "data": "{{beps_parameters}}",
                        "dataset_name": "{{beps_dataset_name}}",
                        "quantity": "fit_parameter",
                        "units": "a.u.",
                        "dimensions": [
                            {
                                "axis": 0,
                                "name": "X",
                                "quantity": "X",
                                "units": "m",
                                "dimension_type": "spatial",
                                "values": "{{x_values}}",
                            },
                            {
                                "axis": 1,
                                "name": "Y",
                                "quantity": "Y",
                                "units": "m",
                                "dimension_type": "spatial",
                                "values": "{{y_values}}",
                            },
                            {
                                "axis": 2,
                                "name": "fit_parameter",
                                "quantity": "fit_parameter",
                                "units": "index",
                                "dimension_type": "spectral",
                                "values": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                            },
                        ],
                        "metadata": {
                            "fit_kind": "beps_loop",
                            "source_dataset": "{{file_path}}",
                        },
                    },
                    "notes": "Store the BEPS fit output as a sidpy.Dataset with spatial X/Y axes preserved.",
                },
                {
                    "tool": "fit_sho_response_tool",
                    "arguments": {
                        "real_data": "{{sho_real_data}}",
                        "imag_data": "{{sho_imag_data}}",
                        "frequency": "{{frequency}}",
                        "use_kmeans": "{{use_kmeans}}",
                        "n_clusters": "{{n_clusters}}",
                        "return_cov": False,
                        "loss": "linear",
                        "dataset_name": "sho_response_fit",
                    },
                    "notes": "Fit the complex SHO slice after the loop map has been built.",
                },
                {
                    "tool": "create_dataset_tool",
                    "arguments": {
                        "data": "{{sho_parameters}}",
                        "dataset_name": "{{sho_dataset_name}}",
                        "quantity": "fit_parameter",
                        "units": "a.u.",
                        "dimensions": [
                            {
                                "axis": 0,
                                "name": "X",
                                "quantity": "X",
                                "units": "m",
                                "dimension_type": "spatial",
                                "values": "{{x_values}}",
                            },
                            {
                                "axis": 1,
                                "name": "Y",
                                "quantity": "Y",
                                "units": "m",
                                "dimension_type": "spatial",
                                "values": "{{y_values}}",
                            },
                            {
                                "axis": 2,
                                "name": "fit_parameter",
                                "quantity": "fit_parameter",
                                "units": "index",
                                "dimension_type": "spectral",
                                "values": [0, 1, 2, 3],
                            },
                        ],
                        "metadata": {
                            "fit_kind": "sho",
                            "source_dataset": "{{file_path}}",
                        },
                    },
                    "notes": "Save the SHO fit output as a sidpy.Dataset so it can be inspected or reused.",
                },
            ],
        }
    ]
}


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

    This matches the tested heuristic in ``tests/proc/test_fitter_refactor.py``.
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


def _build_workflow_registry(include_internal: bool = False) -> Dict[str, Dict[str, Any]]:
    """Build a registry of named workflow examples."""
    del include_internal

    registry: Dict[str, Dict[str, Any]] = {}
    for group, workflows in WORKFLOW_EXAMPLES.items():
        for item in workflows:
            base_name = str(item.get("name", "")).strip()
            if not base_name:
                continue

            full_name = f"{group}.{base_name}"
            suffix = 2
            while full_name in registry:
                full_name = f"{group}.{base_name}_{suffix}"
                suffix += 1

            registry[full_name] = {
                "name": full_name,
                "group": group,
                "base_name": base_name,
                "goal": item.get("goal", ""),
                "inputs": item.get("inputs", {}),
                "setup": item.get("setup", []),
                "steps": item.get("steps", []),
            }
    return registry


def get_workflow_examples(include_internal: bool = False) -> Dict[str, Any]:
    """Return grouped workflow examples for MCP clients and LLM planners."""
    registry = _build_workflow_registry(include_internal=include_internal)
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for workflow in registry.values():
        grouped.setdefault(workflow["group"], []).append(workflow)
    return {"server": "sidpy-beps-fitting", "workflows": grouped}


def list_named_workflows(include_internal: bool = False) -> list[Dict[str, Any]]:
    """Return a compact list of named workflow examples."""
    registry = _build_workflow_registry(include_internal=include_internal)
    return [
        {
            "name": wf["name"],
            "group": wf["group"],
            "goal": wf["goal"],
            "input_names": sorted(wf.get("inputs", {}).keys()),
            "step_count": len(wf["steps"]),
        }
        for wf in registry.values()
    ]


def get_named_workflow(name: str, include_internal: bool = False) -> Dict[str, Any]:
    """Return one workflow by unique name or by unique base name."""
    registry = _build_workflow_registry(include_internal=include_internal)
    if name in registry:
        return registry[name]

    matches = [wf for wf in registry.values() if wf["base_name"] == name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = sorted(wf["name"] for wf in matches)
        raise ValueError(f"Workflow name {name!r} is ambiguous. Use one of: {names}")
    raise ValueError(f"Workflow {name!r} was not found.")


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


def _normalize_data_type(data_type: Optional[str]) -> str:
    return "UNKNOWN" if data_type is None else str(data_type)


def _dataset_dimension_payload(dimension: sid.Dimension, axis: int) -> Dict[str, Any]:
    values = np.asarray(dimension)
    return {
        "axis": axis,
        "name": dimension.name,
        "quantity": dimension.quantity,
        "units": dimension.units,
        "dimension_type": dimension.dimension_type.name,
        "length": int(values.size),
        "values": values.tolist(),
    }


def _dataset_payload(dataset: sid.Dataset, dataset_id: Optional[str] = None) -> Dict[str, Any]:
    payload = {
        "shape": list(dataset.shape),
        "ndim": int(dataset.ndim),
        "title": dataset.title,
        "quantity": dataset.quantity,
        "units": dataset.units,
        "data_type": dataset.data_type.name,
        "modality": dataset.modality,
        "source": dataset.source,
        "metadata": _as_builtin(dataset.metadata),
        "original_metadata": _as_builtin(dataset.original_metadata),
        "dimensions": [_dataset_dimension_payload(dataset._axes[axis], axis) for axis in sorted(dataset._axes)],
    }
    if dataset_id is not None:
        payload["dataset_id"] = dataset_id
    return payload


def _store_dataset(dataset: sid.Dataset, dataset_id: Optional[str] = None) -> str:
    if dataset_id is None:
        dataset_id = str(uuid4())
    DATASET_REGISTRY[dataset_id] = dataset
    return dataset_id


def _get_dataset(dataset_id: str) -> sid.Dataset:
    try:
        return DATASET_REGISTRY[dataset_id]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset_id '{dataset_id}'. Create or register a dataset first.") from exc


def _merge_nested_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


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

    dataset = sid.Dataset.from_array(array, title=dataset_name)
    for dim in range(array.ndim - 1):
        dataset.set_dimension(
            dim,
            sid.Dimension(
                np.arange(array.shape[dim]),
                name=f"position_{dim}",
                quantity=f"Position {dim}",
                units="a.u.",
                dimension_type="spatial",
            ),
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


def create_dataset(
    data: Sequence[Any],
    *,
    dataset_name: str = "sidpy_dataset",
    data_type: Optional[str] = None,
    quantity: str = "generic",
    units: str = "generic",
    modality: str = "generic",
    source: str = "generic",
    dimensions: Optional[Sequence[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    original_metadata: Optional[Dict[str, Any]] = None,
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create and register a sidpy.Dataset from nested array-like data."""
    dataset = sid.Dataset.from_array(
        data,
        title=dataset_name,
        datatype=_normalize_data_type(data_type),
        quantity=quantity,
        units=units,
        modality=modality,
        source=source,
    )

    if dimensions is not None:
        if len(dimensions) != dataset.ndim:
            raise ValueError(
                "dimensions must provide one entry per dataset axis. "
                f"Received {len(dimensions)} dimensions for dataset.ndim={dataset.ndim}."
            )
        specified_axes = [int(dimension_spec.get("axis", axis)) for axis, dimension_spec in enumerate(dimensions)]
        expected_axes = set(range(dataset.ndim))
        if len(set(specified_axes)) != len(specified_axes):
            raise ValueError("Each dimension axis may only be specified once.")
        if set(specified_axes) != expected_axes:
            raise ValueError(
                "dimensions must cover every dataset axis exactly once. "
                f"Received axes {specified_axes} for expected axes {sorted(expected_axes)}."
            )
        for axis, dimension_spec in enumerate(dimensions):
            axis_index = int(dimension_spec.get("axis", axis))
            values = dimension_spec.get("values", np.arange(dataset.shape[axis_index]))
            name = dimension_spec.get("name", f"dim_{axis_index}")
            quantity_value = dimension_spec.get("quantity", name)
            units_value = dimension_spec.get("units", "generic")
            dim_type = dimension_spec.get("dimension_type", "UNKNOWN")
            dataset.set_dimension(
                axis_index,
                sid.Dimension(
                    values,
                    name=name,
                    quantity=quantity_value,
                    units=units_value,
                    dimension_type=dim_type,
                ),
            )

    if metadata is not None:
        dataset.metadata = dict(metadata)
    if original_metadata is not None:
        dataset.original_metadata = dict(original_metadata)

    dataset_id = _store_dataset(dataset, dataset_id=dataset_id)
    return _dataset_payload(dataset, dataset_id=dataset_id)


def get_dataset(dataset_id: str) -> Dict[str, Any]:
    """Return a JSON-friendly summary of a registered dataset."""
    dataset = _get_dataset(dataset_id)
    return _dataset_payload(dataset, dataset_id=dataset_id)


def list_datasets() -> Dict[str, Any]:
    """List dataset ids currently stored in the in-memory MCP registry."""
    return {
        "datasets": [
            {
                "dataset_id": dataset_id,
                "title": dataset.title,
                "shape": list(dataset.shape),
                "data_type": dataset.data_type.name,
            }
            for dataset_id, dataset in DATASET_REGISTRY.items()
        ]
    }


def add_metadata(
    dataset_id: str,
    metadata: Dict[str, Any],
    *,
    merge: bool = True,
    target: str = "metadata",
) -> Dict[str, Any]:
    """Add or replace metadata on a registered dataset."""
    dataset = _get_dataset(dataset_id)
    target_name = str(target).lower()
    if target_name not in {"metadata", "original_metadata"}:
        raise ValueError("target must be either 'metadata' or 'original_metadata'.")

    current = getattr(dataset, target_name)
    next_value = _merge_nested_dict(current, metadata) if merge else dict(metadata)
    setattr(dataset, target_name, next_value)
    return _dataset_payload(dataset, dataset_id=dataset_id)


def update_dimension(
    dataset_id: str,
    axis: int,
    *,
    values: Sequence[float],
    name: Optional[str] = None,
    quantity: Optional[str] = None,
    units: Optional[str] = None,
    dimension_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Replace one dataset dimension using sidpy.Dimension and Dataset.set_dimension."""
    dataset = _get_dataset(dataset_id)
    axis = int(axis)
    existing_dimension = dataset._axes[axis]
    replacement = sid.Dimension(
        values,
        name=name or existing_dimension.name,
        quantity=quantity or existing_dimension.quantity,
        units=units or existing_dimension.units,
        dimension_type=existing_dimension.dimension_type if dimension_type is None else dimension_type,
    )
    dataset.set_dimension(axis, replacement)
    return _dataset_payload(dataset, dataset_id=dataset_id)


def rename_dimension(dataset_id: str, axis: int, name: str) -> Dict[str, Any]:
    """Rename one registered dataset dimension."""
    dataset = _get_dataset(dataset_id)
    dataset.rename_dimension(int(axis), name)
    return _dataset_payload(dataset, dataset_id=dataset_id)


def remove_dataset(dataset_id: str) -> Dict[str, Any]:
    """Remove a dataset from the in-memory registry."""
    dataset = _get_dataset(dataset_id)
    payload = {
        "dataset_id": dataset_id,
        "title": dataset.title,
        "removed": True,
    }
    del DATASET_REGISTRY[dataset_id]
    return payload


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
    fitter = SidpyFitterRefactor(
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
    fitter = SidpyFitterRefactor(
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
    """Create an MCP server exposing sidpy dataset and fitting tools."""
    if FastMCP is None:  # pragma: no cover - optional runtime dependency
        raise ImportError("The 'mcp' package is required to create the BEPS MCP server.")

    server = FastMCP(server_name)

    @server.tool()
    def create_dataset_tool(
        data: Sequence[Any],
        dataset_name: str = "sidpy_dataset",
        data_type: Optional[str] = None,
        quantity: str = "generic",
        units: str = "generic",
        modality: str = "generic",
        source: str = "generic",
        dimensions: Optional[Sequence[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        original_metadata: Optional[Dict[str, Any]] = None,
        dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a sidpy.Dataset and store it in the MCP server registry."""
        return create_dataset(
            data,
            dataset_name=dataset_name,
            data_type=data_type,
            quantity=quantity,
            units=units,
            modality=modality,
            source=source,
            dimensions=dimensions,
            metadata=metadata,
            original_metadata=original_metadata,
            dataset_id=dataset_id,
        )

    @server.tool()
    def get_dataset_tool(dataset_id: str) -> Dict[str, Any]:
        """Return a registered dataset summary including dimensions and metadata."""
        return get_dataset(dataset_id)

    @server.tool()
    def list_datasets_tool() -> Dict[str, Any]:
        """List the registered dataset ids in this MCP server process."""
        return list_datasets()

    @server.tool()
    def add_metadata_tool(
        dataset_id: str,
        metadata: Dict[str, Any],
        merge: bool = True,
        target: str = "metadata",
    ) -> Dict[str, Any]:
        """Add or replace metadata on a registered sidpy dataset."""
        return add_metadata(dataset_id, metadata, merge=merge, target=target)

    @server.tool()
    def update_dimension_tool(
        dataset_id: str,
        axis: int,
        values: Sequence[float],
        name: Optional[str] = None,
        quantity: Optional[str] = None,
        units: Optional[str] = None,
        dimension_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Replace one dataset dimension with new values and optional metadata."""
        return update_dimension(
            dataset_id,
            axis,
            values=values,
            name=name,
            quantity=quantity,
            units=units,
            dimension_type=dimension_type,
        )

    @server.tool()
    def rename_dimension_tool(dataset_id: str, axis: int, name: str) -> Dict[str, Any]:
        """Rename one dataset dimension."""
        return rename_dimension(dataset_id, axis, name)

    @server.tool()
    def remove_dataset_tool(dataset_id: str) -> Dict[str, Any]:
        """Remove a dataset from the in-memory MCP registry."""
        return remove_dataset(dataset_id)

    @server.tool()
    def get_workflow_examples_tool(include_internal: bool = False) -> Dict[str, Any]:
        """Return workflow examples that chain SciFiReaders with BEPS and SHO fitting tools."""
        return get_workflow_examples(include_internal=include_internal)

    @server.tool()
    def workflow_list_named_workflows_tool(include_internal: bool = False) -> list[Dict[str, Any]]:
        """Return compact summaries of the named workflow examples."""
        return list_named_workflows(include_internal=include_internal)

    @server.tool()
    def workflow_get_named_workflow_tool(name: str, include_internal: bool = False) -> Dict[str, Any]:
        """Return one named workflow example."""
        return get_named_workflow(name, include_internal=include_internal)

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
    try:
        mcp.run(transport="stdio")
    except TypeError:  # pragma: no cover - compatibility with older MCP releases
        mcp.run()


__all__ = [
    "DATASET_REGISTRY",
    "LOOP_PARAMETER_LABELS",
    "SHO_PARAMETER_LABELS",
    "SHO_fit_flattened",
    "add_metadata",
    "calculate_loop_centroid",
    "create_dataset",
    "create_mcp_server",
    "get_named_workflow",
    "get_workflow_examples",
    "fit_beps_loops",
    "fit_sho_response",
    "generate_guess",
    "get_dataset",
    "list_datasets",
    "list_named_workflows",
    "loop_fit_function",
    "main",
    "remove_dataset",
    "rename_dimension",
    "sho_guess_fn",
    "update_dimension",
]
