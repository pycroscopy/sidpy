"""
MCP server utilities for BEPS loop and SHO fitting.

This module wraps the existing ``SidpyFitterRefactor`` workflows in MCP-friendly
tools so an LLM client can fit nested-array BEPS loop data and SHO response
data without re-implementing the fitting logic.

The MCP runtime is optional. Importing this module does not require ``mcp``,
but running the server does.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
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
                "Read a BEPS HDF5/NSID file with the SciFiReaders MCP server, "
                "then run one sidpy workflow tool that fits SHO over the full DC "
                "sweep for one cycle, derives the loop input by projecting the "
                "fitted SHO amplitude and phase with BGlib projectLoop, and "
                "saves both fit-parameter maps as sidpy.Datasets."
            ),
            "inputs": {
                "file_path": "/path/to/PTO_5x5.h5",
                "channel_name": "Channel_000",
                "sho_cycle_index": 1,
                "use_kmeans": False,
                "n_clusters": 4,
                "scifireaders_return_mode": "data",
                "beps_dataset_name": "beps_fit_parameters",
                "sho_dataset_name": "sho_fit_parameters",
            },
            "setup": [
                {
                    "kind": "mcp",
                    "server": "scifireaders",
                    "tool": "read_scifireaders_file",
                    "arguments": {
                        "file_path": "{{file_path}}",
                        "return_mode": "{{scifireaders_return_mode}}",
                    },
                    "notes": (
                        "Read the file with the SciFiReaders MCP server so the file "
                        "never needs to be handled directly in Python."
                    ),
                },
            ],
            "steps": [
                {
                    "tool": "fit_beps_dataset_workflow_tool",
                    "arguments": {
                        "source_file_path": "{{file_path}}",
                        "channel_name": "{{channel_name}}",
                        "dataset_name": "{{file_path}}",
                        "cycle_index": "{{sho_cycle_index}}",
                        "use_kmeans": "{{use_kmeans}}",
                        "n_clusters": "{{n_clusters}}",
                        "return_cov": False,
                        "loss": "linear",
                        "sho_dataset_name": "{{sho_dataset_name}}",
                        "beps_dataset_name": "{{beps_dataset_name}}",
                    },
                    "assign": "workflow_result",
                    "notes": (
                        "Run the full fit workflow in one sidpy MCP call after "
                        "the SciFiReaders read step."
                    ),
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


def _dimension_type_value(dimension_type: Any) -> Any:
    if hasattr(dimension_type, "name"):
        return dimension_type
    return str(dimension_type)


def _is_dataset_like(value: Any) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "metadata")
        and hasattr(value, "get_dimension_by_number")
    )


def _restore_complex_payload(value: Any) -> Any:
    """Restore JSON-safe complex markers back into complex numbers."""
    if isinstance(value, dict):
        if set(value.keys()) == {"__complex__"}:
            real, imag = value["__complex__"]
            return complex(real, imag)
        return {key: _restore_complex_payload(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_restore_complex_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_complex_payload(item) for item in value)
    return value


def _dataset_from_payload(payload: Dict[str, Any], *, dataset_name: Optional[str] = None) -> sid.Dataset:
    if "data" not in payload:
        raise ValueError("The dataset payload must include a 'data' field.")

    data = _restore_complex_payload(payload["data"])
    dataset = sid.Dataset.from_array(
        np.asarray(data),
        title=dataset_name or payload.get("title", "sidpy_dataset"),
        datatype=_normalize_data_type(payload.get("data_type")),
        quantity=payload.get("quantity", "generic"),
        units=payload.get("units", "generic"),
        modality=payload.get("modality", "generic"),
        source=payload.get("source", "generic"),
    )

    for dim in payload.get("dimensions", []):
        axis = int(dim.get("axis", len(dataset._axes)))
        if axis >= dataset.ndim:
            continue
        values = dim.get("values", np.arange(dataset.shape[axis]))
        dataset.set_dimension(
            axis,
            sid.Dimension(
                values,
                name=dim.get("name", f"dim_{axis}"),
                quantity=dim.get("quantity", dim.get("name", f"dim_{axis}")),
                units=dim.get("units", "generic"),
                dimension_type=_dimension_type_value(dim.get("dimension_type", "UNKNOWN")),
            ),
        )

    if "metadata" in payload:
        dataset.metadata = dict(payload.get("metadata", {}))
    if "original_metadata" in payload:
        dataset.original_metadata = dict(payload.get("original_metadata", {}))
    return dataset


def _select_dataset_from_read_result(read_result: Any, channel_name: Optional[str] = None) -> sid.Dataset:
    """Pick one sidpy.Dataset out of a SciFiReaders read result."""
    def _collect_datasets(value: Any) -> list[sid.Dataset]:
        if _is_dataset_like(value):
            return [value]
        if isinstance(value, dict) and {"data", "dimensions"}.issubset(value.keys()):
            return [_dataset_from_payload(value, dataset_name=value.get("title"))]
        if isinstance(value, dict):
            datasets: list[sid.Dataset] = []
            for key, item in value.items():
                if isinstance(item, dict) and {"data", "dimensions"}.issubset(item.keys()):
                    datasets.append(_dataset_from_payload(item, dataset_name=str(key)))
                else:
                    datasets.extend(_collect_datasets(item))
            return datasets
        if isinstance(value, (list, tuple)):
            datasets = []
            for item in value:
                datasets.extend(_collect_datasets(item))
            return datasets
        return []

    if isinstance(read_result, sid.Dataset):
        return read_result
    if _is_dataset_like(read_result):
        return read_result  # type: ignore[return-value]

    if isinstance(read_result, dict):
        if channel_name is not None:
            if channel_name in read_result and isinstance(read_result[channel_name], sid.Dataset):
                return read_result[channel_name]
            dataset_values = _collect_datasets(read_result)
            if dataset_values:
                return dataset_values[0]
            raise KeyError(f"Channel {channel_name!r} was not found in the reader result.")

        dataset_values = _collect_datasets(read_result)
        if len(dataset_values) == 1:
            return dataset_values[0]
        if not dataset_values:
            raise ValueError("The reader result did not contain any sidpy.Dataset objects.")
        raise ValueError("channel_name is required when the reader result contains multiple datasets.")

    if isinstance(read_result, (list, tuple)):
        dataset_values = _collect_datasets(read_result)
        if len(dataset_values) == 1:
            return dataset_values[0]
        if not dataset_values:
            raise ValueError("The reader result did not contain any sidpy.Dataset objects.")
        raise ValueError("channel_name is required when the reader result contains multiple datasets.")

    raise TypeError("Unsupported reader result type.")


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


def register_external_dataset(
    reader_payload: Dict[str, Any],
    *,
    channel_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Register a SciFiReaders dataset payload as a sidpy.Dataset."""
    payload = reader_payload
    if "datasets" in reader_payload:
        datasets = reader_payload["datasets"]
        if channel_name is not None:
            try:
                payload = datasets[channel_name]
            except KeyError as exc:
                raise KeyError(f"Channel {channel_name!r} was not found in the reader payload.") from exc
        elif len(datasets) == 1:
            payload = next(iter(datasets.values()))
        else:
            raise ValueError("channel_name is required when the reader payload contains multiple datasets.")

    dataset = _dataset_from_payload(payload, dataset_name=dataset_name)
    dataset_id = _store_dataset(dataset, dataset_id=dataset_id)
    return _dataset_payload(dataset, dataset_id=dataset_id)


def register_external_dataset_from_file(
    output_file_path: str,
    *,
    channel_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Register a SciFiReaders-exported NeXus file as a sidpy.Dataset."""
    try:
        import SciFiReaders as sr
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError("SciFiReaders is required to register a reader output file.") from exc

    reader = sr.NSIDReader(str(Path(output_file_path)))
    read_result = reader.read()
    dataset = _select_dataset_from_read_result(read_result, channel_name=channel_name)
    if dataset_name:
        dataset = dataset.copy()
        dataset.title = dataset_name
    dataset_id = _store_dataset(dataset, dataset_id=dataset_id)
    return _dataset_payload(dataset, dataset_id=dataset_id)


@lru_cache(maxsize=1)
def _load_bglib_be_loop_module():
    """Load BGlib's be_loop helpers for piezoresponse projection."""
    module_name = "BGlib.be.analysis.utils.be_loop"
    try:
        return import_module(module_name)
    except ImportError:
        module_path = (
            Path(__file__).resolve().parents[2].parent
            / "BGlib"
            / "BGlib"
            / "be"
            / "analysis"
            / "utils"
            / "be_loop.py"
        )
        if not module_path.exists():
            raise ImportError(
                "BGlib is required to project SHO amplitude and phase into a "
                "piezoresponse loop."
            )
        spec = spec_from_file_location("_bglib_be_loop_mcp", module_path)
        if spec is None or spec.loader is None:
            raise ImportError("Unable to load BGlib be_loop helpers.")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _project_piezoresponse_loop(
    dc_voltage: Sequence[float],
    amplitude_vec: Sequence[float],
    phase_vec: Sequence[float],
) -> Dict[str, Any]:
    """Project SHO amplitude and phase into a BGlib piezoresponse loop."""
    be_loop = _load_bglib_be_loop_module()
    results = be_loop.projectLoop(
        np.asarray(dc_voltage, dtype=float).ravel().tolist(),
        np.asarray(amplitude_vec, dtype=float).ravel().tolist(),
        np.asarray(phase_vec, dtype=float).ravel().tolist(),
    )
    return dict(results)


def get_dataset(dataset_id: str) -> Dict[str, Any]:
    """Return a JSON-friendly summary of a registered dataset."""
    dataset = _get_dataset(dataset_id)
    return _dataset_payload(dataset, dataset_id=dataset_id)


def derive_loop_input_from_sho_result(sho_dataset_id: str) -> Dict[str, Any]:
    """Derive the BEPS loop input from a saved SHO fit dataset."""
    sho_dataset = _get_dataset(sho_dataset_id)
    metadata = dict(sho_dataset.metadata)
    source_dataset_id = metadata.get("source_dataset_id")
    source_slice = metadata.get("source_slice", {})
    if not source_dataset_id:
        raise ValueError("The SHO result dataset is missing source_dataset_id metadata.")

    source_dataset = _get_dataset(str(source_dataset_id))
    cycle_index = int(source_slice.get("cycle_index"))
    if sho_dataset.ndim < 4:
        raise ValueError("The SHO result dataset must include a DC axis and fit-parameter axis.")
    if source_dataset.ndim < 5:
        raise ValueError(
            "The source dataset must have at least five dimensions to derive a BEPS loop slice."
        )

    if sho_dataset.shape[-1] < 1:
        raise ValueError("The SHO result dataset must contain at least one fit parameter.")

    if sho_dataset.shape[-1] < 4:
        raise ValueError("The SHO result dataset must include amplitude and phase fit parameters.")

    dc_voltage = np.asarray(source_dataset._axes[3].values)
    sho_amplitude = np.asarray(sho_dataset[..., 0], dtype=float)
    sho_phase = np.asarray(sho_dataset[..., 3], dtype=float)

    projected = np.zeros(sho_amplitude.shape[:2] + (dc_voltage.size,), dtype=float)
    rotation = np.zeros(sho_amplitude.shape[:2] + (2,), dtype=float)
    centroid = np.zeros(sho_amplitude.shape[:2] + (2,), dtype=float)
    geometric_area = np.zeros(sho_amplitude.shape[:2], dtype=float)
    for row in range(sho_amplitude.shape[0]):
        for col in range(sho_amplitude.shape[1]):
            projection = _project_piezoresponse_loop(
                dc_voltage,
                sho_amplitude[row, col],
                sho_phase[row, col],
            )
            projected[row, col] = np.asarray(projection["Projected Loop"], dtype=float)
            rotation[row, col] = np.asarray(projection["Rotation Matrix"], dtype=float)
            centroid[row, col] = np.asarray(projection["Centroid"], dtype=float)
            geometric_area[row, col] = float(projection["Geometric Area"])

    dc_voltage = np.asarray(source_dataset._axes[3].values)

    return {
        "source_dataset_id": source_dataset_id,
        "source_slice": {
            "cycle_index": cycle_index,
            "signal": "projected_piezoresponse",
            "projection_tool": "BGlib.projectLoop",
        },
        "beps_data": projected.tolist(),
        "dc_voltage": dc_voltage.tolist(),
        "projected_loop": projected.tolist(),
        "projection": {
            "rotation_matrix": rotation.tolist(),
            "centroid": centroid.tolist(),
            "geometric_area": geometric_area.tolist(),
        },
    }


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


def fit_sho_response_from_dataset(
    dataset_id: str,
    *,
    dc_index: int,
    cycle_index: int,
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
    """Fit SHO response directly from a registered dataset using axis indices."""
    dataset = _get_dataset(dataset_id)
    if dataset.ndim < 5:
        raise ValueError("The source dataset must have at least five dimensions for SHO fitting.")
    return fit_sho_response(
        np.real(np.asarray(dataset[:, :, :, dc_index, cycle_index])),
        np.asarray(dataset._axes[2].values),
        imag_data=np.imag(np.asarray(dataset[:, :, :, dc_index, cycle_index])),
        use_kmeans=use_kmeans,
        n_clusters=n_clusters,
        return_cov=return_cov,
        loss=loss,
        f_scale=f_scale,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        chunks=chunks,
        dataset_name=dataset_name,
    )


def fit_sho_response_over_dc_from_dataset(
    dataset_id: str,
    *,
    cycle_index: int,
    use_kmeans: bool = False,
    n_clusters: int = 10,
    return_cov: bool = False,
    loss: str = "linear",
    f_scale: float = 1.0,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None,
    chunks: Any = "auto",
    dataset_name: str = "sho_response_dc_data",
) -> Dict[str, Any]:
    """Fit SHO response over the full DC sweep for one cycle."""
    dataset = _get_dataset(dataset_id)
    if dataset.ndim < 5:
        raise ValueError("The source dataset must have at least five dimensions for SHO fitting.")

    complex_array = np.asarray(dataset[:, :, :, :, cycle_index])
    complex_array = np.transpose(complex_array, (0, 1, 3, 2))
    return fit_sho_response(
        np.real(complex_array),
        np.asarray(dataset._axes[2].values),
        imag_data=np.imag(complex_array),
        use_kmeans=use_kmeans,
        n_clusters=n_clusters,
        return_cov=return_cov,
        loss=loss,
        f_scale=f_scale,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        chunks=chunks,
        dataset_name=dataset_name,
    )


def fit_beps_loops_from_dataset(
    dataset_id: str,
    *,
    frequency_index: int,
    cycle_index: int,
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
    """Fit BEPS loops directly from a registered dataset using axis indices."""
    dataset = _get_dataset(dataset_id)
    if dataset.ndim < 5:
        raise ValueError("The source dataset must have at least five dimensions for BEPS fitting.")
    return fit_beps_loops(
        np.real(np.asarray(dataset[:, :, frequency_index, :, cycle_index])),
        np.asarray(dataset._axes[3].values),
        use_kmeans=use_kmeans,
        n_clusters=n_clusters,
        return_cov=return_cov,
        loss=loss,
        f_scale=f_scale,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        chunks=chunks,
        dataset_name=dataset_name,
    )


def _r2_score(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Compute a simple R^2 score without depending on scikit-learn."""
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.size == 0:
        return float("nan")
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    if ss_tot == 0.0:
        return 1.0
    return 1.0 - (ss_res / ss_tot)


def _fit_parameter_dataset(
    parameters: Sequence[Any],
    *,
    dataset_name: str,
    x_values: Sequence[Any],
    y_values: Sequence[Any],
    parameter_labels: Sequence[str],
    fit_kind: str,
    source_dataset: str,
    source_dataset_id: str,
    source_slice: Dict[str, Any],
    fit_quality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Store a fit-parameter map with preserved X/Y axes and workflow metadata."""
    params = np.asarray(parameters)
    x_axis = np.asarray(x_values).tolist()
    y_axis = np.asarray(y_values).tolist()
    dims = [
        {
            "axis": 0,
            "name": "X",
            "quantity": "X",
            "units": "m",
            "dimension_type": "spatial",
            "values": x_axis,
        },
        {
            "axis": 1,
            "name": "Y",
            "quantity": "Y",
            "units": "m",
            "dimension_type": "spatial",
            "values": y_axis,
        },
        {
            "axis": 2,
            "name": "fit_parameter",
            "quantity": "fit_parameter",
            "units": "index",
            "dimension_type": "spectral",
            "values": list(range(params.shape[-1])),
        },
    ]
    metadata: Dict[str, Any] = {
        "fit_kind": fit_kind,
        "parameter_labels": list(parameter_labels),
        "source_dataset": source_dataset,
        "source_dataset_id": source_dataset_id,
        "source_slice": dict(source_slice),
    }
    if fit_quality is not None:
        metadata["fit_quality"] = dict(fit_quality)
    return create_dataset(
        params,
        dataset_name=dataset_name,
        quantity="fit_parameter",
        units="a.u.",
        dimensions=dims,
        metadata=metadata,
    )


def fit_beps_dataset_from_reader_payload(
    reader_payload: Dict[str, Any],
    *,
    channel_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    sho_dc_index: int,
    sho_cycle_index: int,
    beps_frequency_index: int,
    beps_cycle_index: int,
    use_kmeans: bool = False,
    n_clusters: int = 6,
    return_cov: bool = False,
    loss: str = "linear",
    f_scale: float = 1.0,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None,
    source_dataset_id: Optional[str] = None,
    sho_dataset_name: str = "sho_fit_parameters",
    beps_dataset_name: str = "beps_fit_parameters",
) -> Dict[str, Any]:
    """Run the BEPS workflow from a SciFiReaders payload in one tool chain."""
    registered = register_external_dataset(
        reader_payload,
        channel_name=channel_name,
        dataset_name=dataset_name,
        dataset_id=source_dataset_id,
    )
    source_dataset_id = registered["dataset_id"]
    source_dataset_title = registered.get("title", dataset_name or "sidpy_dataset")

    sho_result = fit_sho_response_over_dc_from_dataset(
        source_dataset_id,
        cycle_index=sho_cycle_index,
        use_kmeans=use_kmeans,
        n_clusters=n_clusters,
        return_cov=return_cov,
        loss=loss,
        f_scale=f_scale,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        dataset_name=sho_dataset_name,
    )
    sho_params = np.asarray(sho_result["parameters"])
    source_dataset = _get_dataset(source_dataset_id)
    sho_data = np.asarray(source_dataset[:, :, :, :, sho_cycle_index])
    sho_data = np.transpose(sho_data, (0, 1, 3, 2))
    sho_axis = np.asarray(source_dataset._axes[2].values)
    sho_pred_flat = []
    sho_true_flat = []
    sho_amp_true = []
    sho_amp_pred = []
    for row in range(sho_data.shape[0]):
        for col in range(sho_data.shape[1]):
            for dc_index in range(sho_data.shape[2]):
                pred_flat = SHO_fit_flattened(sho_axis, *sho_params[row, col, dc_index])
                true_flat = np.hstack([sho_data[row, col, dc_index].real, sho_data[row, col, dc_index].imag])
                sho_pred_flat.append(pred_flat)
                sho_true_flat.append(true_flat)
                pred_complex = pred_flat[: len(sho_axis)] + 1j * pred_flat[len(sho_axis) :]
                sho_amp_true.append(np.abs(sho_data[row, col, dc_index]))
                sho_amp_pred.append(np.abs(pred_complex))
    sho_pred_flat = np.asarray(sho_pred_flat)
    sho_true_flat = np.asarray(sho_true_flat)
    sho_amp_true = np.asarray(sho_amp_true)
    sho_amp_pred = np.asarray(sho_amp_pred)
    sho_quality = {
        "overall_r2": _r2_score(sho_true_flat, sho_pred_flat),
        "amplitude_overall_r2": _r2_score(sho_amp_true, sho_amp_pred),
    }
    sho_dataset = _fit_parameter_dataset(
        sho_params,
        dataset_name=sho_dataset_name,
        x_values=source_dataset._axes[0].values,
        y_values=source_dataset._axes[1].values,
        parameter_labels=SHO_PARAMETER_LABELS,
        fit_kind="sho",
        source_dataset=source_dataset_title,
        source_dataset_id=source_dataset_id,
        source_slice={
            "cycle_index": int(sho_cycle_index),
        },
        fit_quality=sho_quality,
    )

    loop_input = derive_loop_input_from_sho_result(sho_dataset["dataset_id"])
    beps_data = np.asarray(loop_input["beps_data"])
    dc_voltage = np.asarray(loop_input["dc_voltage"])
    beps_result = fit_beps_loops(
        beps_data,
        dc_voltage,
        use_kmeans=use_kmeans,
        n_clusters=n_clusters,
        return_cov=return_cov,
        loss=loss,
        f_scale=f_scale,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        dataset_name=beps_dataset_name,
    )
    beps_params = np.asarray(beps_result["parameters"])

    source_slice = loop_input["source_slice"]
    beps_pred = np.asarray([
        loop_fit_function(dc_voltage, *beps_params[row, col])
        for row in range(beps_data.shape[0])
        for col in range(beps_data.shape[1])
    ]).reshape(beps_data.shape)
    beps_quality = {
        "overall_r2": _r2_score(beps_data, beps_pred),
    }
    beps_dataset = _fit_parameter_dataset(
        beps_params,
        dataset_name=beps_dataset_name,
        x_values=source_dataset._axes[0].values,
        y_values=source_dataset._axes[1].values,
        parameter_labels=LOOP_PARAMETER_LABELS,
        fit_kind="beps_loop",
        source_dataset=source_dataset_title,
        source_dataset_id=source_dataset_id,
        source_slice=source_slice,
        fit_quality=beps_quality,
    )

    return {
        "source_dataset_id": source_dataset_id,
        "sho_result": sho_result,
        "sho_dataset": sho_dataset,
        "sho_dataset_id": sho_dataset["dataset_id"],
        "loop_input": {
            "source_dataset_id": loop_input["source_dataset_id"],
            "source_slice": loop_input["source_slice"],
        },
        "beps_result": beps_result,
        "beps_dataset": beps_dataset,
        "beps_dataset_id": beps_dataset["dataset_id"],
    }


def fit_beps_dataset_from_scifireaders_file(
    output_file_path: str,
    *,
    channel_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    sho_cycle_index: int,
    use_kmeans: bool = False,
    n_clusters: int = 6,
    return_cov: bool = False,
    loss: str = "linear",
    f_scale: float = 1.0,
    lower_bounds: Optional[Sequence[float]] = None,
    upper_bounds: Optional[Sequence[float]] = None,
    source_dataset_id: Optional[str] = None,
    sho_dataset_name: str = "sho_fit_parameters",
    beps_dataset_name: str = "beps_fit_parameters",
) -> Dict[str, Any]:
    """Run the full BEPS workflow from a SciFiReaders-exported file path."""
    registered = register_external_dataset_from_file(
        output_file_path,
        channel_name=None,
        dataset_name=dataset_name,
        dataset_id=source_dataset_id,
    )
    source_dataset_id = registered["dataset_id"]
    source_dataset_title = registered.get("title", dataset_name or "sidpy_dataset")
    source_dataset = _get_dataset(source_dataset_id)

    sho_result = fit_sho_response_over_dc_from_dataset(
        source_dataset_id,
        cycle_index=sho_cycle_index,
        use_kmeans=use_kmeans,
        n_clusters=n_clusters,
        return_cov=return_cov,
        loss=loss,
        f_scale=f_scale,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        dataset_name=sho_dataset_name,
    )
    sho_params = np.asarray(sho_result["parameters"])
    sho_data = np.asarray(source_dataset[:, :, :, :, sho_cycle_index])
    sho_data = np.transpose(sho_data, (0, 1, 3, 2))
    sho_axis = np.asarray(source_dataset._axes[2].values)
    sho_pred_flat = []
    sho_true_flat = []
    sho_amp_true = []
    sho_amp_pred = []
    for row in range(sho_data.shape[0]):
        for col in range(sho_data.shape[1]):
            for dc_index in range(sho_data.shape[2]):
                pred_flat = SHO_fit_flattened(sho_axis, *sho_params[row, col, dc_index])
                true_flat = np.hstack([sho_data[row, col, dc_index].real, sho_data[row, col, dc_index].imag])
                sho_pred_flat.append(pred_flat)
                sho_true_flat.append(true_flat)
                pred_complex = pred_flat[: len(sho_axis)] + 1j * pred_flat[len(sho_axis) :]
                sho_amp_true.append(np.abs(sho_data[row, col, dc_index]))
                sho_amp_pred.append(np.abs(pred_complex))
    sho_quality = {
        "overall_r2": _r2_score(sho_true_flat, sho_pred_flat),
        "amplitude_overall_r2": _r2_score(sho_amp_true, sho_amp_pred),
    }
    sho_dataset = create_dataset(
        sho_params,
        dataset_name=sho_dataset_name,
        quantity="fit_parameter",
        units="a.u.",
        dimensions=[
            {
                "axis": 0,
                "name": "X",
                "quantity": "X",
                "units": "m",
                "dimension_type": "spatial",
                "values": np.asarray(source_dataset._axes[0].values).tolist(),
            },
            {
                "axis": 1,
                "name": "Y",
                "quantity": "Y",
                "units": "m",
                "dimension_type": "spatial",
                "values": np.asarray(source_dataset._axes[1].values).tolist(),
            },
            {
                "axis": 2,
                "name": "DC Offset",
                "quantity": "DC Offset",
                "units": "Volts",
                "dimension_type": "spectral",
                "values": np.asarray(source_dataset._axes[3].values).tolist(),
            },
            {
                "axis": 3,
                "name": "fit_parameter",
                "quantity": "fit_parameter",
                "units": "index",
                "dimension_type": "spectral",
                "values": list(range(sho_params.shape[-1])),
            },
        ],
        metadata={
            "fit_kind": "sho_dc_sweep",
            "parameter_labels": SHO_PARAMETER_LABELS,
            "source_dataset": source_dataset_title,
            "source_dataset_id": source_dataset_id,
            "source_slice": {
                "cycle_index": int(sho_cycle_index),
            },
            "fit_quality": sho_quality,
        },
    )

    loop_input = derive_loop_input_from_sho_result(sho_dataset["dataset_id"])
    beps_data = np.asarray(loop_input["beps_data"])
    dc_voltage = np.asarray(loop_input["dc_voltage"])
    beps_result = fit_beps_loops(
        beps_data,
        dc_voltage,
        use_kmeans=use_kmeans,
        n_clusters=n_clusters,
        return_cov=return_cov,
        loss=loss,
        f_scale=f_scale,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        dataset_name=beps_dataset_name,
    )
    beps_params = np.asarray(beps_result["parameters"])
    beps_pred = np.asarray([
        loop_fit_function(dc_voltage, *beps_params[row, col])
        for row in range(beps_data.shape[0])
        for col in range(beps_data.shape[1])
    ]).reshape(beps_data.shape)
    beps_quality = {
        "overall_r2": _r2_score(beps_data, beps_pred),
    }
    beps_dataset = _fit_parameter_dataset(
        beps_params,
        dataset_name=beps_dataset_name,
        x_values=source_dataset._axes[0].values,
        y_values=source_dataset._axes[1].values,
        parameter_labels=LOOP_PARAMETER_LABELS,
        fit_kind="beps_loop",
        source_dataset=source_dataset_title,
        source_dataset_id=source_dataset_id,
        source_slice=loop_input["source_slice"],
        fit_quality=beps_quality,
    )

    return {
        "source_dataset_id": source_dataset_id,
        "sho_result": sho_result,
        "sho_dataset": sho_dataset,
        "sho_dataset_id": sho_dataset["dataset_id"],
        "loop_input": {
            "source_dataset_id": loop_input["source_dataset_id"],
            "source_slice": loop_input["source_slice"],
        },
        "beps_result": beps_result,
        "beps_dataset": beps_dataset,
        "beps_dataset_id": beps_dataset["dataset_id"],
    }


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
    def register_external_dataset_tool(
        reader_payload: Dict[str, Any],
        channel_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a SciFiReaders dataset payload as a sidpy.Dataset."""
        return register_external_dataset(
            reader_payload,
            channel_name=channel_name,
            dataset_name=dataset_name,
            dataset_id=dataset_id,
        )

    @server.tool()
    def get_dataset_tool(dataset_id: str) -> Dict[str, Any]:
        """Return a registered dataset summary including dimensions and metadata."""
        return get_dataset(dataset_id)

    @server.tool()
    def derive_loop_input_from_sho_result_tool(sho_dataset_id: str) -> Dict[str, Any]:
        """Derive BEPS loop input from a saved SHO fit dataset."""
        return derive_loop_input_from_sho_result(sho_dataset_id)

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
    def fit_sho_response_from_dataset_tool(
        dataset_id: str,
        dc_index: int,
        cycle_index: int,
        use_kmeans: bool = False,
        n_clusters: int = 10,
        return_cov: bool = False,
        loss: str = "linear",
        f_scale: float = 1.0,
        lower_bounds: Optional[Sequence[float]] = None,
        upper_bounds: Optional[Sequence[float]] = None,
        dataset_name: str = "sho_response_data",
    ) -> Dict[str, Any]:
        """Fit SHO response directly from a registered dataset."""
        return fit_sho_response_from_dataset(
            dataset_id,
            dc_index=dc_index,
            cycle_index=cycle_index,
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
    def fit_sho_response_over_dc_from_dataset_tool(
        dataset_id: str,
        cycle_index: int,
        use_kmeans: bool = False,
        n_clusters: int = 10,
        return_cov: bool = False,
        loss: str = "linear",
        f_scale: float = 1.0,
        lower_bounds: Optional[Sequence[float]] = None,
        upper_bounds: Optional[Sequence[float]] = None,
        dataset_name: str = "sho_response_dc_data",
    ) -> Dict[str, Any]:
        """Fit SHO response over the full DC sweep for one cycle."""
        return fit_sho_response_over_dc_from_dataset(
            dataset_id,
            cycle_index=cycle_index,
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
    def fit_beps_loops_from_dataset_tool(
        dataset_id: str,
        frequency_index: int,
        cycle_index: int,
        use_kmeans: bool = False,
        n_clusters: int = 6,
        return_cov: bool = False,
        loss: str = "linear",
        f_scale: float = 1.0,
        lower_bounds: Optional[Sequence[float]] = None,
        upper_bounds: Optional[Sequence[float]] = None,
        dataset_name: str = "beps_loop_data",
    ) -> Dict[str, Any]:
        """Fit BEPS loops directly from a registered dataset."""
        return fit_beps_loops_from_dataset(
            dataset_id,
            frequency_index=frequency_index,
            cycle_index=cycle_index,
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
    def fit_beps_dataset_workflow_tool(
        reader_payload: Optional[Dict[str, Any]] = None,
        scifireaders_result: Optional[Dict[str, Any]] = None,
        scifireaders_output_file_path: Optional[str] = None,
        source_file_path: Optional[str] = None,
        channel_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dc_index: int = 0,
        cycle_index: int = 0,
        frequency_index: int = 0,
        beps_cycle_index: int = 0,
        use_kmeans: bool = False,
        n_clusters: int = 6,
        return_cov: bool = False,
        loss: str = "linear",
        f_scale: float = 1.0,
        lower_bounds: Optional[Sequence[float]] = None,
        upper_bounds: Optional[Sequence[float]] = None,
        source_dataset_id: Optional[str] = None,
        sho_dataset_name: str = "sho_fit_parameters",
        beps_dataset_name: str = "beps_fit_parameters",
    ) -> Dict[str, Any]:
        """Run the full BEPS dataset workflow from one SciFiReaders payload."""
        if source_file_path is not None:
            return fit_beps_dataset_from_scifireaders_file(
                source_file_path,
                channel_name=channel_name,
                dataset_name=dataset_name,
                sho_cycle_index=cycle_index,
                use_kmeans=use_kmeans,
                n_clusters=n_clusters,
                return_cov=return_cov,
                loss=loss,
                f_scale=f_scale,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                source_dataset_id=source_dataset_id,
                sho_dataset_name=sho_dataset_name,
                beps_dataset_name=beps_dataset_name,
            )
        if scifireaders_output_file_path is not None:
            return fit_beps_dataset_from_scifireaders_file(
                scifireaders_output_file_path,
                channel_name=channel_name,
                dataset_name=dataset_name,
                sho_cycle_index=cycle_index,
                use_kmeans=use_kmeans,
                n_clusters=n_clusters,
                return_cov=return_cov,
                loss=loss,
                f_scale=f_scale,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                source_dataset_id=source_dataset_id,
                sho_dataset_name=sho_dataset_name,
                beps_dataset_name=beps_dataset_name,
            )

        payload = reader_payload if reader_payload is not None else scifireaders_result
        if payload is None:
            raise ValueError(
                "Provide the full SciFiReaders response as 'scifireaders_result' "
                "'reader_payload', a 'scifireaders_output_file_path', or a 'source_file_path'."
            )
        return fit_beps_dataset_from_reader_payload(
            payload,
            channel_name=channel_name,
            dataset_name=dataset_name,
            sho_dc_index=dc_index,
            sho_cycle_index=cycle_index,
            beps_frequency_index=frequency_index,
            beps_cycle_index=beps_cycle_index,
            use_kmeans=use_kmeans,
            n_clusters=n_clusters,
            return_cov=return_cov,
            loss=loss,
            f_scale=f_scale,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            source_dataset_id=source_dataset_id,
            sho_dataset_name=sho_dataset_name,
            beps_dataset_name=beps_dataset_name,
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
    "derive_loop_input_from_sho_result",
    "get_named_workflow",
    "get_workflow_examples",
    "fit_beps_loops",
    "fit_beps_loops_from_dataset",
    "fit_beps_dataset_from_scifireaders_file",
    "fit_beps_dataset_from_reader_payload",
    "fit_sho_response",
    "fit_sho_response_from_dataset",
    "fit_sho_response_over_dc_from_dataset",
    "generate_guess",
    "get_dataset",
    "list_datasets",
    "list_named_workflows",
    "loop_fit_function",
    "main",
    "remove_dataset",
    "register_external_dataset",
    "rename_dimension",
    "sho_guess_fn",
    "update_dimension",
]
