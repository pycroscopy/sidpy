import unittest
import json
from unittest.mock import patch

import numpy as np
from sklearn.metrics import r2_score

from sidpy.proc import mcp_server_beps as mcp_mod
from examples.mcp.run_beps_mcp_client import DEFAULT_SERVER_COMMAND, StdioMcpClient, _extract_tool_text


class TestSidpyCoreMCPTools(unittest.TestCase):

    def setUp(self):
        mcp_mod.DATASET_REGISTRY.clear()

    def tearDown(self):
        mcp_mod.DATASET_REGISTRY.clear()

    def test_create_dataset_registers_core_sidpy_dataset(self):
        result = mcp_mod.create_dataset(
            [[1, 2, 3], [4, 5, 6]],
            dataset_name="example",
            data_type="spectrum",
            quantity="intensity",
            units="a.u.",
            dimensions=[
                {
                    "name": "x",
                    "quantity": "position",
                    "units": "nm",
                    "dimension_type": "spatial",
                    "values": [0, 1],
                },
                {
                    "name": "energy",
                    "quantity": "energy",
                    "units": "eV",
                    "dimension_type": "spectral",
                    "values": [10, 20, 30],
                },
            ],
            metadata={"experiment": {"sample": "demo"}},
        )

        self.assertIn("dataset_id", result)
        self.assertEqual(result["title"], "example")
        self.assertEqual(result["data_type"], "SPECTRUM")
        self.assertEqual(result["metadata"]["experiment"]["sample"], "demo")
        self.assertEqual(result["dimensions"][0]["name"], "x")
        self.assertEqual(result["dimensions"][1]["dimension_type"], "SPECTRAL")
        self.assertEqual(len(mcp_mod.DATASET_REGISTRY), 1)

    def test_add_metadata_merges_nested_dicts(self):
        created = mcp_mod.create_dataset(np.arange(6).reshape(2, 3), dataset_name="meta")

        updated = mcp_mod.add_metadata(
            created["dataset_id"],
            {"experiment": {"sample": "PTO"}},
            merge=True,
        )
        updated = mcp_mod.add_metadata(
            created["dataset_id"],
            {"experiment": {"temperature_K": 300}, "operator": "tester"},
            merge=True,
        )

        self.assertEqual(updated["metadata"]["experiment"]["sample"], "PTO")
        self.assertEqual(updated["metadata"]["experiment"]["temperature_K"], 300)
        self.assertEqual(updated["metadata"]["operator"], "tester")

    def test_update_and_rename_dimension_use_sidpy_dimension_operations(self):
        created = mcp_mod.create_dataset(np.arange(6).reshape(2, 3), dataset_name="dims")
        dataset_id = created["dataset_id"]

        updated = mcp_mod.update_dimension(
            dataset_id,
            1,
            values=[100, 200, 300],
            name="bias",
            quantity="voltage",
            units="V",
            dimension_type="spectral",
        )
        renamed = mcp_mod.rename_dimension(dataset_id, 0, "row")

        self.assertEqual(updated["dimensions"][1]["name"], "bias")
        self.assertEqual(updated["dimensions"][1]["values"], [100.0, 200.0, 300.0])
        self.assertEqual(updated["dimensions"][1]["dimension_type"], "SPECTRAL")
        self.assertEqual(renamed["dimensions"][0]["name"], "row")

    def test_update_dimension_preserves_existing_dimension_type_when_unspecified(self):
        created = mcp_mod.create_dataset(
            np.arange(6).reshape(2, 3),
            dataset_name="preserve_dim_type",
            dimensions=[
                {
                    "axis": 0,
                    "name": "row",
                    "quantity": "row",
                    "units": "px",
                    "dimension_type": "spatial",
                    "values": [0, 1],
                },
                {
                    "axis": 1,
                    "name": "energy",
                    "quantity": "energy",
                    "units": "eV",
                    "dimension_type": "spectral",
                    "values": [1, 2, 3],
                },
            ],
        )

        updated = mcp_mod.update_dimension(
            created["dataset_id"],
            1,
            values=[10, 20, 30],
        )

        self.assertEqual(updated["dimensions"][1]["dimension_type"], "SPECTRAL")
        self.assertEqual(updated["dimensions"][1]["name"], "energy")

    def test_create_dataset_rejects_duplicate_or_incomplete_axis_specs(self):
        with self.assertRaises(ValueError):
            mcp_mod.create_dataset(
                np.arange(6).reshape(2, 3),
                dimensions=[
                    {"axis": 1, "name": "duplicate_1", "values": [0, 1, 2]},
                    {"axis": 1, "name": "duplicate_2", "values": [10, 20, 30]},
                ],
            )

        with self.assertRaises(ValueError):
            mcp_mod.create_dataset(
                np.arange(24).reshape(2, 3, 4),
                dimensions=[
                    {"axis": 0, "name": "x", "values": [0, 1]},
                    {"axis": 2, "name": "z", "values": [0, 1, 2, 3]},
                    {"axis": 2, "name": "z_again", "values": [0, 1, 2, 3]},
                ],
            )

    def test_list_and_remove_dataset(self):
        first = mcp_mod.create_dataset([[1, 2]], dataset_name="first")
        second = mcp_mod.create_dataset([[3, 4]], dataset_name="second")

        listed = mcp_mod.list_datasets()
        listed_ids = {item["dataset_id"] for item in listed["datasets"]}
        self.assertEqual(listed_ids, {first["dataset_id"], second["dataset_id"]})

        removed = mcp_mod.remove_dataset(first["dataset_id"])
        self.assertTrue(removed["removed"])
        self.assertNotIn(first["dataset_id"], mcp_mod.DATASET_REGISTRY)

    def test_fit_beps_dataset_workflow_example_is_published(self):
        workflows = mcp_mod.get_workflow_examples()
        self.assertIn("analysis", workflows["workflows"])

        summaries = mcp_mod.list_named_workflows()
        self.assertTrue(any(item["name"] == "analysis.fit_beps_dataset" for item in summaries))

        workflow = mcp_mod.get_named_workflow("fit_beps_dataset")
        self.assertEqual(workflow["name"], "analysis.fit_beps_dataset")
        self.assertEqual(workflow["group"], "analysis")
        self.assertEqual(len(workflow["steps"]), 1)
        step_tools = [step.get("tool") for step in workflow["steps"]]
        self.assertIn("fit_beps_dataset_workflow_tool", step_tools)
        self.assertEqual(workflow["setup"][0]["server"], "scifireaders")
        self.assertEqual(workflow["setup"][0]["tool"], "read_scifireaders_file")
        self.assertTrue(
            any("one sidpy MCP call" in step.get("notes", "") or "one sidpy workflow tool" in step.get("notes", "") for step in workflow["steps"] + workflow["setup"])
        )

    def test_fit_beps_dataset_workflow_forwards_channel_name_for_file_inputs(self):
        captured = {}

        class _StopWorkflow(Exception):
            pass

        def _fake_register_external_dataset_from_file(
            output_file_path,
            *,
            channel_name=None,
            dataset_name=None,
            dataset_id=None,
        ):
            captured["output_file_path"] = output_file_path
            captured["channel_name"] = channel_name
            captured["dataset_name"] = dataset_name
            captured["dataset_id"] = dataset_id
            raise _StopWorkflow

        with patch.object(mcp_mod, "register_external_dataset_from_file", side_effect=_fake_register_external_dataset_from_file):
            with self.assertRaises(_StopWorkflow):
                mcp_mod.fit_beps_dataset_from_scifireaders_file(
                    "/tmp/example.h5",
                    channel_name="Channel_000",
                    dataset_name="PTO_5x5.h5",
                    sho_cycle_index=1,
                    use_kmeans=False,
                    n_clusters=4,
                    return_cov=False,
                    loss="linear",
                    sho_dataset_name="sho_fit_parameters",
                    beps_dataset_name="beps_fit_parameters",
                )

        self.assertEqual(captured["output_file_path"], "/tmp/example.h5")
        self.assertEqual(captured["channel_name"], "Channel_000")
        self.assertEqual(captured["dataset_name"], "PTO_5x5.h5")

    def test_fit_parameter_dataset_preserves_dc_axis_for_4d_sho_maps(self):
        params = np.zeros((2, 3, 4, 5))
        result = mcp_mod._fit_parameter_dataset(
            params,
            dataset_name="sho_fit_parameters",
            x_values=[0.0, 1.0],
            y_values=[10.0, 20.0, 30.0],
            dc_values=[-1.0, 0.0, 1.0, 2.0],
            parameter_labels=mcp_mod.SHO_PARAMETER_LABELS,
            fit_kind="sho",
            source_dataset="demo",
            source_dataset_id="dataset-1",
            source_slice={"cycle_index": 1},
        )

        self.assertEqual(result["shape"], [2, 3, 4, 5])
        self.assertEqual(result["dimensions"][2]["name"], "DC Offset")
        self.assertEqual(result["dimensions"][2]["values"], [-1.0, 0.0, 1.0, 2.0])
        self.assertEqual(result["dimensions"][3]["name"], "fit_parameter")
        self.assertEqual(result["dimensions"][3]["values"], [0, 1, 2, 3, 4])

    def test_align_loop_trace_honors_explicit_roll_steps(self):
        dc_voltage = np.array([1.0, 2.0, -2.0, -1.0, 0.0])
        loop_trace = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

        aligned_dc, aligned_trace, applied_roll = mcp_mod._align_loop_trace(
            dc_voltage,
            loop_trace,
            roll_steps=-2,
        )

        np.testing.assert_array_equal(aligned_dc, np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(aligned_trace, np.array([12.0, 13.0, 14.0, 10.0, 11.0]))
        self.assertEqual(applied_roll, -2)

    def test_align_loop_trace_infers_minimum_dc_roll(self):
        dc_voltage = np.array([1.0, 2.0, -2.0, -1.0, 0.0])
        loop_trace = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

        aligned_dc, aligned_trace, applied_roll = mcp_mod._align_loop_trace(dc_voltage, loop_trace)

        np.testing.assert_array_equal(aligned_dc, np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(aligned_trace, np.array([12.0, 13.0, 14.0, 10.0, 11.0]))
        self.assertEqual(applied_roll, -2)

    def test_real_file_path_workflow_returns_sho_and_beps_datasets(self):
        from pathlib import Path
        import shutil
        import os
        import tempfile

        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        try:
            import SciFiReaders as sr
        except Exception as exc:
            self.skipTest(f"SciFiReaders is required for the workflow file-path regression test: {exc}")

        file_path = Path("/Users/rvv/Downloads/PTO_5x5.h5")
        if not file_path.exists():
            self.skipTest(f"Real BEPS fixture not found: {file_path}")

        fd, tmp_name = tempfile.mkstemp(prefix="pto_5x5_sidpy_", suffix=".h5")
        os.close(fd)
        workflow_path = Path(tmp_name)
        shutil.copy2(file_path, workflow_path)

        reader = sr.NSIDReader(str(workflow_path))
        data = reader.read()["Channel_000"]
        array = np.asarray(data)

        result = mcp_mod.fit_beps_dataset_from_scifireaders_file(
            str(workflow_path),
            channel_name="Channel_000",
            dataset_name="PTO_5x5.h5",
            sho_cycle_index=1,
            use_kmeans=False,
            n_clusters=4,
            return_cov=False,
            loss="linear",
            sho_dataset_name="pto_5x5_sho_workflow",
            beps_dataset_name="pto_5x5_beps_workflow",
        )

        self.assertIn("sho_dataset_id", result)
        self.assertIn("beps_dataset_id", result)
        self.assertEqual(result["sho_dataset"]["shape"], [5, 5, array.shape[3], 4])
        self.assertEqual(result["beps_dataset"]["shape"], [5, 5, 9])
        self.assertEqual(
            result["loop_input"]["source_slice"],
            {
                "cycle_index": 1,
                "signal": "projected_piezoresponse",
                "projection_tool": "BGlib.projectLoop",
            },
        )
        self.assertEqual(result["sho_dataset"]["metadata"]["fit_kind"], "sho")
        self.assertEqual(result["beps_dataset"]["metadata"]["fit_kind"], "beps_loop")
        self.assertGreater(result["sho_dataset"]["metadata"]["fit_quality"]["overall_r2"], 0.8)
        self.assertGreater(result["sho_dataset"]["metadata"]["fit_quality"]["amplitude_overall_r2"], 0.9)
        self.assertGreater(result["beps_dataset"]["metadata"]["fit_quality"]["overall_r2"], 0.75)
        self.assertIn("increasing_branch_r2", result["beps_dataset"]["metadata"]["fit_quality"])
        self.assertIn("decreasing_branch_r2", result["beps_dataset"]["metadata"]["fit_quality"])

    def test_real_file_scifireaders_payload_can_drive_the_executable_workflow(self):
        from pathlib import Path
        import shutil
        import os
        import tempfile

        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        try:
            from SciFiReaders.mcp import scifireaders_mcp
        except Exception as exc:
            self.skipTest(f"SciFiReaders MCP is required for the executable workflow test: {exc}")

        file_path = Path("/Users/rvv/Downloads/PTO_5x5.h5")
        if not file_path.exists():
            self.skipTest(f"Real BEPS fixture not found: {file_path}")

        fd, tmp_name = tempfile.mkstemp(prefix="pto_5x5_sidpy_", suffix=".h5")
        os.close(fd)
        working_path = Path(tmp_name)
        shutil.copy2(file_path, working_path)

        reader_payload = scifireaders_mcp.read_file(str(working_path), return_mode="data")
        self.assertIn("datasets", reader_payload)

        registered = mcp_mod.register_external_dataset(
            reader_payload,
            channel_name="Channel_000",
            dataset_name="pto_5x5_from_scifireaders",
        )
        source_dataset_id = registered["dataset_id"]
        array = np.asarray(reader_payload["datasets"]["Channel_000"]["data"])

        sho_cycle_idx = 1
        sho_result = mcp_mod.fit_sho_response_over_dc_from_dataset(
            source_dataset_id,
            cycle_index=sho_cycle_idx,
            use_kmeans=False,
            n_clusters=4,
            return_cov=False,
            loss="linear",
            dataset_name="pto_5x5_sho_from_dataset",
        )
        sho_params = np.asarray(sho_result["parameters"])
        sho_data = np.transpose(array[:, :, :, :, sho_cycle_idx], (0, 1, 3, 2))
        sho_axis = np.asarray(reader_payload["datasets"]["Channel_000"]["dimensions"][2]["values"])
        sho_pred_concat = []
        sho_true_concat = []
        sho_amp_true = []
        sho_amp_pred = []
        for row in range(sho_data.shape[0]):
            for col in range(sho_data.shape[1]):
                for dc_idx in range(sho_data.shape[2]):
                    pred_flat = mcp_mod.SHO_fit_flattened(sho_axis, *sho_params[row, col, dc_idx])
                    true_flat = np.hstack([sho_data[row, col, dc_idx].real, sho_data[row, col, dc_idx].imag])
                    sho_pred_concat.append(pred_flat)
                    sho_true_concat.append(true_flat)
                    pred_complex = pred_flat[: len(sho_axis)] + 1j * pred_flat[len(sho_axis) :]
                    sho_amp_true.append(np.abs(sho_data[row, col, dc_idx]))
                    sho_amp_pred.append(np.abs(pred_complex))
        sho_overall_r2 = r2_score(np.asarray(sho_true_concat).reshape(-1), np.asarray(sho_pred_concat).reshape(-1))
        sho_amp_overall_r2 = r2_score(np.asarray(sho_amp_true).reshape(-1), np.asarray(sho_amp_pred).reshape(-1))
        self.assertGreater(sho_overall_r2, 0.8)
        self.assertGreater(sho_amp_overall_r2, 0.9)

        sho_dataset = mcp_mod.create_dataset(
            sho_params,
            dataset_name="pto_5x5_sho_fit_parameters",
            quantity="fit_parameter",
            units="a.u.",
            dimensions=[
                {
                    "axis": 0,
                    "name": "X",
                    "quantity": "X",
                    "units": "m",
                    "dimension_type": "spatial",
                    "values": np.asarray(registered["dimensions"][0]["values"]).tolist(),
                },
                {
                    "axis": 1,
                    "name": "Y",
                    "quantity": "Y",
                    "units": "m",
                    "dimension_type": "spatial",
                    "values": np.asarray(registered["dimensions"][1]["values"]).tolist(),
                },
                {
                    "axis": 2,
                    "name": "DC Offset",
                    "quantity": "DC Offset",
                    "units": "Volts",
                    "dimension_type": "spectral",
                    "values": np.asarray(registered["dimensions"][3]["values"]).tolist(),
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
                "source_dataset": str(file_path),
                "source_dataset_id": source_dataset_id,
                "source_slice": {
                    "cycle_index": sho_cycle_idx,
                },
            },
        )

        loop_input = mcp_mod.derive_loop_input_from_sho_result(sho_dataset["dataset_id"])
        self.assertIn("beps_data", loop_input)
        self.assertIn("dc_voltage", loop_input)
        expected_roll_steps = -int(np.argmin(beps_axis))
        expected_dc_voltage = np.roll(beps_axis, expected_roll_steps)
        self.assertEqual(
            loop_input["source_slice"],
            {
                "cycle_index": sho_cycle_idx,
                "loop_roll_steps": expected_roll_steps,
                "signal": "projected_piezoresponse",
                "projection_tool": "BGlib.projectLoop",
            },
        )
        expected_projected = np.asarray(
            [
                mcp_mod._project_piezoresponse_loop(
                    beps_axis,
                    sho_params[row, col, :, 0],
                    sho_params[row, col, :, 3],
                )["Projected Loop"]
                for row in range(sho_params.shape[0])
                for col in range(sho_params.shape[1])
            ]
        ).reshape(beps_data.shape)
        expected_projected = np.roll(expected_projected, expected_roll_steps, axis=-1)
        expected_projected = expected_projected * 1e3
        np.testing.assert_allclose(np.asarray(loop_input["beps_data"]), expected_projected)
        np.testing.assert_allclose(np.asarray(loop_input["dc_voltage"]), expected_dc_voltage)

        beps_result = mcp_mod.fit_beps_loops(
            loop_input["beps_data"],
            loop_input["dc_voltage"],
            use_kmeans=False,
            n_clusters=4,
            return_cov=False,
            loss="linear",
            dataset_name="pto_5x5_beps_from_derived_input",
        )
        self.assertEqual(beps_result["fit_kind"], "beps_loop")
        self.assertEqual(beps_result["parameter_shape"], [5, 5, 9])
        beps_params = np.asarray(beps_result["parameters"])
        beps_pred = np.asarray([
            mcp_mod.loop_fit_function(loop_input["dc_voltage"], *beps_params[row, col])
            for row in range(beps_params.shape[0])
            for col in range(beps_params.shape[1])
        ]).reshape(np.asarray(loop_input["beps_data"]).shape)
        beps_r2 = r2_score(np.asarray(loop_input["beps_data"]).reshape(-1), beps_pred.reshape(-1))
        self.assertGreater(beps_r2, 0.75)
        branch_quality = mcp_mod._branch_r2_scores(
            np.asarray(loop_input["dc_voltage"]),
            np.asarray(loop_input["beps_data"]),
            beps_pred,
        )
        self.assertIn("increasing_branch_r2", branch_quality)
        self.assertIn("decreasing_branch_r2", branch_quality)

    def test_fit_beps_loops_rejects_multi_cycle_dc_voltage(self):
        dc_voltage = np.array([
            -1.0, -0.5, 0.0, 0.5, 1.0,
            0.5, 0.0, -0.5, -1.0,
            -0.5, 0.0, 0.5, 1.0,
            0.5, 0.0, -0.5, -1.0,
        ])
        data = np.zeros((1, 1, dc_voltage.size))

        with self.assertRaises(ValueError):
            mcp_mod.fit_beps_loops(data, dc_voltage)

    def test_real_file_fits_round_trip_to_sidpy_datasets(self):
        from pathlib import Path
        import os

        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

        try:
            import SciFiReaders as sr
        except Exception as exc:
            self.skipTest(f"SciFiReaders is required for the real-file MCP integration test: {exc}")

        file_path = Path(
            "/Users/rvv/Downloads/PTO_5x5.h5"
        )
        if not file_path.exists():
            self.skipTest(f"Real BEPS fixture not found: {file_path}")

        reader = sr.NSIDReader(str(file_path))
        data = reader.read()["Channel_000"]
        array = np.asarray(data)

        # Pick representative slices from the real file.
        # The best slices are determined by a lightweight proxy score so the test
        # remains tied to the real dataset while still being deterministic.
        beps_candidates = []
        for freq_idx in range(array.shape[2]):
            for cycle_idx in range(array.shape[4]):
                loop_slice = np.real(array[:, :, freq_idx, :, cycle_idx])
                score = float(np.mean(np.std(loop_slice, axis=2)))
                beps_candidates.append((score, freq_idx, cycle_idx))
        _, beps_freq_idx, beps_cycle_idx = max(beps_candidates, key=lambda item: item[0])
        beps_data = np.real(array[:, :, beps_freq_idx, :, beps_cycle_idx])
        beps_axis = np.asarray(data._axes[3].values)

        sho_candidates = []
        for dc_idx in range(array.shape[3]):
            for cycle_idx in range(array.shape[4]):
                sho_slice = np.abs(array[:, :, :, dc_idx, cycle_idx])
                score = float(np.mean(np.std(sho_slice, axis=2)))
                sho_candidates.append((score, dc_idx, cycle_idx))
        _, sho_dc_idx, sho_cycle_idx = max(sho_candidates, key=lambda item: item[0])
        sho_data = array[:, :, :, sho_dc_idx, sho_cycle_idx]
        sho_axis = np.asarray(data._axes[2].values)

        source_dataset = mcp_mod.create_dataset(
            array,
            dataset_name="pto_5x5_source_dataset",
            quantity="signal",
            units="a.u.",
            dimensions=[
                {
                    "axis": 0,
                    "name": "X",
                    "quantity": "X",
                    "units": "m",
                    "dimension_type": "spatial",
                    "values": np.asarray(data._axes[0].values).tolist(),
                },
                {
                    "axis": 1,
                    "name": "Y",
                    "quantity": "Y",
                    "units": "m",
                    "dimension_type": "spatial",
                    "values": np.asarray(data._axes[1].values).tolist(),
                },
                {
                    "axis": 2,
                    "name": "Frequency",
                    "quantity": "Frequency",
                    "units": "Hz",
                    "dimension_type": "spectral",
                    "values": np.asarray(data._axes[2].values).tolist(),
                },
                {
                    "axis": 3,
                    "name": "DC Offset",
                    "quantity": "Voltage",
                    "units": "Volts",
                    "dimension_type": "spectral",
                    "values": np.asarray(data._axes[3].values).tolist(),
                },
                {
                    "axis": 4,
                    "name": "Cycle",
                    "quantity": "Cycle",
                    "units": "index",
                    "dimension_type": "spectral",
                    "values": np.asarray(data._axes[4].values).tolist(),
                },
            ],
            metadata={"source_file": "PTO_5x5.h5"},
        )
        source_dataset_id = source_dataset["dataset_id"]

        # --- BEPS fit ---
        beps_result = mcp_mod.fit_beps_loops(
            beps_data,
            beps_axis,
            use_kmeans=False,
            n_clusters=4,
            return_cov=False,
            loss="linear",
            dataset_name="pto_5x5_beps_fixture",
        )
        self.assertEqual(beps_result["fit_kind"], "beps_loop")
        self.assertEqual(beps_result["parameter_shape"], [5, 5, 9])
        self.assertEqual(beps_result["parameter_labels"], mcp_mod.LOOP_PARAMETER_LABELS)

        beps_params = np.asarray(beps_result["parameters"])
        beps_pred = np.zeros_like(beps_data, dtype=float)
        beps_r2_values = []
        for row in range(beps_data.shape[0]):
            for col in range(beps_data.shape[1]):
                pred = mcp_mod.loop_fit_function(beps_axis, *beps_params[row, col])
                beps_pred[row, col] = pred
                beps_r2_values.append(r2_score(beps_data[row, col], pred))

        beps_overall_r2 = r2_score(beps_data.reshape(-1), beps_pred.reshape(-1))
        self.assertGreater(beps_overall_r2, 0.85)
        self.assertGreater(np.median(beps_r2_values), 0.80)

        beps_dataset = mcp_mod.create_dataset(
            beps_params,
            dataset_name="pto_5x5_beps_fit_parameters",
            quantity="fit_parameter",
            units="a.u.",
            dimensions=[
                {
                    "axis": 0,
                    "name": "X",
                    "quantity": "X",
                    "units": "m",
                    "dimension_type": "spatial",
                    "values": np.asarray(data._axes[0].values).tolist(),
                },
                {
                    "axis": 1,
                    "name": "Y",
                    "quantity": "Y",
                    "units": "m",
                    "dimension_type": "spatial",
                    "values": np.asarray(data._axes[1].values).tolist(),
                },
                {
                    "axis": 2,
                    "name": "fit_parameter",
                    "quantity": "fit_parameter",
                    "units": "index",
                    "dimension_type": "spectral",
                    "values": list(range(beps_params.shape[-1])),
                },
            ],
            metadata={
                "fit_kind": "beps_loop",
                "parameter_labels": mcp_mod.LOOP_PARAMETER_LABELS,
                "source_dataset": "PTO_5x5.h5",
                "source_slice": {
                    "frequency_index": beps_freq_idx,
                    "cycle_index": beps_cycle_idx,
                },
                "fit_quality": {
                    "overall_r2": float(beps_overall_r2),
                    "median_r2": float(np.median(beps_r2_values)),
                },
            },
        )
        self.assertEqual(beps_dataset["shape"], [5, 5, 9])
        self.assertEqual(beps_dataset["metadata"]["fit_kind"], "beps_loop")
        self.assertEqual(beps_dataset["dimensions"][2]["dimension_type"], "SPECTRAL")

        beps_dataset_id = beps_dataset["dataset_id"]
        beps_round_trip = mcp_mod.get_dataset(beps_dataset_id)
        self.assertEqual(beps_round_trip["shape"], [5, 5, 9])
        self.assertEqual(beps_round_trip["metadata"]["fit_quality"]["overall_r2"], float(beps_overall_r2))

        # --- SHO fit ---
        sho_result = mcp_mod.fit_sho_response(
            sho_data.real,
            sho_axis,
            imag_data=sho_data.imag,
            use_kmeans=False,
            n_clusters=4,
            return_cov=False,
            loss="linear",
            dataset_name="pto_5x5_sho_fixture",
        )
        self.assertEqual(sho_result["fit_kind"], "sho")
        self.assertEqual(sho_result["parameter_shape"], [5, 5, 4])
        self.assertEqual(sho_result["parameter_labels"], mcp_mod.SHO_PARAMETER_LABELS)

        sho_params = np.asarray(sho_result["parameters"])
        sho_pred_concat = []
        sho_true_concat = []
        sho_amp_true = []
        sho_amp_pred = []
        sho_complex_r2_values = []
        for row in range(sho_data.shape[0]):
            for col in range(sho_data.shape[1]):
                pred_flat = mcp_mod.SHO_fit_flattened(sho_axis, *sho_params[row, col])
                true_flat = np.hstack([sho_data[row, col].real, sho_data[row, col].imag])
                sho_pred_concat.append(pred_flat)
                sho_true_concat.append(true_flat)
                sho_complex_r2_values.append(r2_score(true_flat, pred_flat))

                pred_complex = pred_flat[: len(sho_axis)] + 1j * pred_flat[len(sho_axis) :]
                sho_amp_true.append(np.abs(sho_data[row, col]))
                sho_amp_pred.append(np.abs(pred_complex))

        sho_pred_concat = np.asarray(sho_pred_concat)
        sho_true_concat = np.asarray(sho_true_concat)
        sho_amp_true = np.asarray(sho_amp_true)
        sho_amp_pred = np.asarray(sho_amp_pred)

        sho_overall_r2 = r2_score(sho_true_concat.reshape(-1), sho_pred_concat.reshape(-1))
        sho_amp_overall_r2 = r2_score(sho_amp_true.reshape(-1), sho_amp_pred.reshape(-1))
        self.assertGreater(sho_overall_r2, 0.85)
        self.assertGreater(sho_amp_overall_r2, 0.94)

        sho_dataset = mcp_mod.create_dataset(
            sho_params,
            dataset_name="pto_5x5_sho_fit_parameters",
            quantity="fit_parameter",
            units="a.u.",
            dimensions=[
                {
                    "axis": 0,
                    "name": "X",
                    "quantity": "X",
                    "units": "m",
                    "dimension_type": "spatial",
                    "values": np.asarray(data._axes[0].values).tolist(),
                },
                {
                    "axis": 1,
                    "name": "Y",
                    "quantity": "Y",
                    "units": "m",
                    "dimension_type": "spatial",
                    "values": np.asarray(data._axes[1].values).tolist(),
                },
                {
                    "axis": 2,
                    "name": "fit_parameter",
                    "quantity": "fit_parameter",
                    "units": "index",
                    "dimension_type": "spectral",
                    "values": list(range(sho_params.shape[-1])),
                },
            ],
            metadata={
                "fit_kind": "sho",
                "parameter_labels": mcp_mod.SHO_PARAMETER_LABELS,
                "source_dataset": "PTO_5x5.h5",
                "source_dataset_id": source_dataset_id,
                "source_slice": {
                    "frequency_index": beps_freq_idx,
                    "cycle_index": beps_cycle_idx,
                },
                "fit_quality": {
                    "overall_r2": float(sho_overall_r2),
                    "amplitude_overall_r2": float(sho_amp_overall_r2),
                },
            },
        )
        self.assertEqual(sho_dataset["shape"], [5, 5, 4])
        self.assertEqual(sho_dataset["metadata"]["fit_kind"], "sho")
        self.assertEqual(sho_dataset["dimensions"][2]["dimension_type"], "SPECTRAL")

        sho_dataset_id = sho_dataset["dataset_id"]
        sho_round_trip = mcp_mod.get_dataset(sho_dataset_id)
        self.assertEqual(sho_round_trip["shape"], [5, 5, 4])
        self.assertEqual(sho_round_trip["metadata"]["fit_quality"]["overall_r2"], float(sho_overall_r2))

        loop_input = mcp_mod.derive_loop_input_from_sho_result(sho_dataset_id)
        self.assertEqual(
            loop_input["source_slice"],
            {"frequency_index": beps_freq_idx, "cycle_index": beps_cycle_idx},
        )
        self.assertEqual(loop_input["beps_data"], beps_data.tolist())

    def test_stdio_mcp_client_can_fit_real_file_and_create_datasets(self):
        from pathlib import Path
        import shutil
        import os
        import tempfile

        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        try:
            import SciFiReaders as sr
        except Exception as exc:
            self.skipTest(f"SciFiReaders is required for the stdio MCP integration test: {exc}")

        file_path = Path("/Users/rvv/Downloads/PTO_5x5.h5")
        if not file_path.exists():
            self.skipTest(f"Real BEPS fixture not found: {file_path}")

        fd, tmp_name = tempfile.mkstemp(prefix="pto_5x5_sidpy_", suffix=".h5")
        os.close(fd)
        analysis_path = Path(tmp_name)
        shutil.copy2(file_path, analysis_path)

        fd, tmp_name = tempfile.mkstemp(prefix="pto_5x5_sidpy_", suffix=".h5")
        os.close(fd)
        workflow_path = Path(tmp_name)
        shutil.copy2(file_path, workflow_path)

        reader = sr.NSIDReader(str(analysis_path))
        data = reader.read()["Channel_000"]
        array = np.asarray(data)

        beps_candidates = []
        for freq_idx in range(array.shape[2]):
            for cycle_idx in range(array.shape[4]):
                loop_slice = np.real(array[:, :, freq_idx, :, cycle_idx])
                score = float(np.mean(np.std(loop_slice, axis=2)))
                beps_candidates.append((score, freq_idx, cycle_idx))
        _, beps_freq_idx, beps_cycle_idx = max(beps_candidates, key=lambda item: item[0])
        beps_data = np.real(array[:, :, beps_freq_idx, :, beps_cycle_idx])
        beps_axis = np.asarray(data._axes[3].values)

        sho_candidates = []
        for dc_idx in range(array.shape[3]):
            for cycle_idx in range(array.shape[4]):
                sho_slice = np.abs(array[:, :, :, dc_idx, cycle_idx])
                score = float(np.mean(np.std(sho_slice, axis=2)))
                sho_candidates.append((score, dc_idx, cycle_idx))
        _, sho_dc_idx, sho_cycle_idx = max(sho_candidates, key=lambda item: item[0])
        sho_data = array[:, :, :, sho_dc_idx, sho_cycle_idx]
        sho_axis = np.asarray(data._axes[2].values)

        xy_axes = [
            {
                "axis": 0,
                "name": "X",
                "quantity": "X",
                "units": "m",
                "dimension_type": "spatial",
                "values": np.asarray(data._axes[0].values).tolist(),
            },
            {
                "axis": 1,
                "name": "Y",
                "quantity": "Y",
                "units": "m",
                "dimension_type": "spatial",
                "values": np.asarray(data._axes[1].values).tolist(),
            },
        ]

        with StdioMcpClient(DEFAULT_SERVER_COMMAND, cwd=Path("/Users/rvv/Github/sidpy")) as client:
            tools = client.list_tools()
            tool_names = {tool["name"] for tool in tools}
            self.assertIn("fit_beps_loops_tool", tool_names)
            self.assertIn("fit_sho_response_tool", tool_names)
            self.assertIn("create_dataset_tool", tool_names)
            self.assertIn("get_dataset_tool", tool_names)

            beps_tool_result = client.call_tool(
                "fit_beps_loops_tool",
                {
                    "data": beps_data.tolist(),
                    "dc_voltage": beps_axis.tolist(),
                    "use_kmeans": True,
                    "n_clusters": 4,
                    "return_cov": False,
                    "loss": "linear",
                    "dataset_name": "pto_5x5_beps_stdio",
                },
            )
            beps_payload = json.loads(_extract_tool_text(beps_tool_result))
            self.assertEqual(beps_payload["fit_kind"], "beps_loop")
            self.assertEqual(beps_payload["parameter_shape"], [5, 5, 9])

            beps_params = np.asarray(beps_payload["parameters"])
            beps_dataset = json.loads(
                _extract_tool_text(
                    client.call_tool(
                        "create_dataset_tool",
                        {
                            "data": beps_params.tolist(),
                            "dataset_name": "pto_5x5_beps_stdio_parameters",
                            "quantity": "fit_parameter",
                            "units": "a.u.",
                            "dimensions": xy_axes
                            + [
                                {
                                    "axis": 2,
                                    "name": "fit_parameter",
                                    "quantity": "fit_parameter",
                                    "units": "index",
                                    "dimension_type": "spectral",
                                    "values": list(range(beps_params.shape[-1])),
                                }
                            ],
                            "metadata": {
                                "fit_kind": "beps_loop",
                                "source_dataset": "PTO_5x5.h5",
                            },
                        },
                    )
                )
            )
            beps_round_trip = json.loads(
                _extract_tool_text(client.call_tool("get_dataset_tool", {"dataset_id": beps_dataset["dataset_id"]}))
            )
            self.assertEqual(beps_dataset["shape"], [5, 5, 9])
            self.assertEqual(beps_round_trip["shape"], [5, 5, 9])

            beps_pred = np.zeros_like(beps_data, dtype=float)
            beps_r2_values = []
            for row in range(beps_data.shape[0]):
                for col in range(beps_data.shape[1]):
                    pred = mcp_mod.loop_fit_function(beps_axis, *beps_params[row, col])
                    beps_pred[row, col] = pred
                    beps_r2_values.append(r2_score(beps_data[row, col], pred))
            self.assertGreater(r2_score(beps_data.reshape(-1), beps_pred.reshape(-1)), 0.85)
            self.assertGreater(np.median(beps_r2_values), 0.80)

            sho_tool_result = client.call_tool(
                "fit_sho_response_tool",
                {
                    "real_data": sho_data.real.tolist(),
                    "imag_data": sho_data.imag.tolist(),
                    "frequency": sho_axis.tolist(),
                    "use_kmeans": False,
                    "n_clusters": 4,
                    "return_cov": False,
                    "loss": "linear",
                    "dataset_name": "pto_5x5_sho_stdio",
                },
            )
            sho_payload = json.loads(_extract_tool_text(sho_tool_result))
            self.assertEqual(sho_payload["fit_kind"], "sho")
            self.assertEqual(sho_payload["parameter_shape"], [5, 5, 4])

            sho_params = np.asarray(sho_payload["parameters"])
            sho_dataset = json.loads(
                _extract_tool_text(
                    client.call_tool(
                        "create_dataset_tool",
                        {
                            "data": sho_params.tolist(),
                            "dataset_name": "pto_5x5_sho_stdio_parameters",
                            "quantity": "fit_parameter",
                            "units": "a.u.",
                            "dimensions": xy_axes
                            + [
                                {
                                    "axis": 2,
                                    "name": "fit_parameter",
                                    "quantity": "fit_parameter",
                                    "units": "index",
                                    "dimension_type": "spectral",
                                    "values": list(range(sho_params.shape[-1])),
                                }
                            ],
                            "metadata": {
                                "fit_kind": "sho",
                                "source_dataset": "PTO_5x5.h5",
                            },
                        },
                    )
                )
            )
            sho_round_trip = json.loads(
                _extract_tool_text(client.call_tool("get_dataset_tool", {"dataset_id": sho_dataset["dataset_id"]}))
            )
            self.assertEqual(sho_dataset["shape"], [5, 5, 4])
            self.assertEqual(sho_round_trip["shape"], [5, 5, 4])

            sho_pred_concat = []
            sho_true_concat = []
            sho_amp_true = []
            sho_amp_pred = []
            for row in range(sho_data.shape[0]):
                for col in range(sho_data.shape[1]):
                    pred_flat = mcp_mod.SHO_fit_flattened(sho_axis, *sho_params[row, col])
                    true_flat = np.hstack([sho_data[row, col].real, sho_data[row, col].imag])
                    sho_pred_concat.append(pred_flat)
                    sho_true_concat.append(true_flat)
                    pred_complex = pred_flat[: len(sho_axis)] + 1j * pred_flat[len(sho_axis) :]
                    sho_amp_true.append(np.abs(sho_data[row, col]))
                    sho_amp_pred.append(np.abs(pred_complex))

            sho_pred_concat = np.asarray(sho_pred_concat)
            sho_true_concat = np.asarray(sho_true_concat)
            sho_amp_true = np.asarray(sho_amp_true)
            sho_amp_pred = np.asarray(sho_amp_pred)
            self.assertGreater(r2_score(sho_true_concat.reshape(-1), sho_pred_concat.reshape(-1)), 0.85)
            self.assertGreater(r2_score(sho_amp_true.reshape(-1), sho_amp_pred.reshape(-1)), 0.94)

    def test_stdio_mcp_client_can_run_full_beps_dataset_workflow(self):
        from pathlib import Path
        import shutil
        import os
        import tempfile

        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        try:
            import SciFiReaders as sr
            from SciFiReaders.mcp import scifireaders_mcp
        except Exception as exc:
            self.skipTest(f"SciFiReaders MCP is required for the workflow stdio integration test: {exc}")

        file_path = Path("/Users/rvv/Downloads/PTO_5x5.h5")
        if not file_path.exists():
            self.skipTest(f"Real BEPS fixture not found: {file_path}")

        fd, tmp_name = tempfile.mkstemp(prefix="pto_5x5_sidpy_", suffix=".h5")
        os.close(fd)
        working_path = Path(tmp_name)
        shutil.copy2(file_path, working_path)

        fd, tmp_name = tempfile.mkstemp(prefix="pto_5x5_sidpy_", suffix=".h5")
        os.close(fd)
        workflow_path = Path(tmp_name)
        shutil.copy2(file_path, workflow_path)

        reader = sr.NSIDReader(str(working_path))
        data = reader.read()["Channel_000"]
        array = np.asarray(data)

        with StdioMcpClient(DEFAULT_SERVER_COMMAND, cwd=Path("/Users/rvv/Github/sidpy")) as client:
            tools = client.list_tools()
            tool_names = {tool["name"] for tool in tools}
            self.assertIn("fit_beps_dataset_workflow_tool", tool_names)

            result = client.call_tool(
                "fit_beps_dataset_workflow_tool",
                {
                    "source_file_path": str(workflow_path),
                    "channel_name": "Channel_000",
                    "dataset_name": "PTO_5x5.h5",
                    "cycle_index": 1,
                    "use_kmeans": False,
                    "n_clusters": 4,
                    "return_cov": False,
                    "loss": "linear",
                    "sho_dataset_name": "pto_5x5_sho_workflow",
                    "beps_dataset_name": "pto_5x5_beps_workflow",
                },
            )
            payload = json.loads(_extract_tool_text(result))

            self.assertIn("sho_dataset_id", payload)
            self.assertIn("beps_dataset_id", payload)
            self.assertEqual(payload["sho_dataset"]["shape"], [5, 5, array.shape[3], 4])
            self.assertEqual(payload["beps_dataset"]["shape"], [5, 5, 9])
            self.assertEqual(
                payload["loop_input"]["source_slice"],
                {
                    "cycle_index": 1,
                    "loop_roll_steps": expected_roll_steps,
                    "signal": "projected_piezoresponse",
                    "projection_tool": "BGlib.projectLoop",
                },
            )
            self.assertEqual(payload["sho_dataset"]["metadata"]["fit_kind"], "sho_dc_sweep")
            self.assertGreater(payload["sho_dataset"]["metadata"]["fit_quality"]["overall_r2"], 0.8)
            self.assertGreater(payload["sho_dataset"]["metadata"]["fit_quality"]["amplitude_overall_r2"], 0.9)
            self.assertGreater(payload["beps_dataset"]["metadata"]["fit_quality"]["overall_r2"], 0.75)
            expected_projected = np.asarray(
                [
                    mcp_mod._project_piezoresponse_loop(
                        np.asarray(data._axes[3].values),
                        sho_params[row, col, :, 0],
                        sho_params[row, col, :, 3],
                    )["Projected Loop"]
                    for row in range(sho_params.shape[0])
                    for col in range(sho_params.shape[1])
                ]
            ).reshape(beps_data.shape)
            expected_roll_steps = -int(np.argmin(beps_axis))
            expected_projected = np.roll(expected_projected, expected_roll_steps, axis=-1)
            expected_projected = expected_projected * 1e3
            np.testing.assert_allclose(np.asarray(payload["loop_input"]["beps_data"]), expected_projected)


if __name__ == "__main__":
    unittest.main()
