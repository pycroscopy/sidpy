import unittest
import json

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
        self.assertGreaterEqual(len(workflow["steps"]), 4)
        self.assertTrue(any(step["tool"] == "fit_beps_loops_tool" for step in workflow["steps"]))
        self.assertTrue(any(step["tool"] == "fit_sho_response_tool" for step in workflow["steps"]))
        self.assertTrue(any(step["tool"] == "create_dataset_tool" for step in workflow["steps"]))
        self.assertIn("SciFiReaders.NSIDReader", workflow["setup"][0]["tool"])

    def test_real_file_fits_round_trip_to_sidpy_datasets(self):
        from pathlib import Path

        try:
            import SciFiReaders as sr
        except ImportError:
            self.skipTest("SciFiReaders is required for the real-file MCP integration test.")

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
                "source_slice": {
                    "dc_index": sho_dc_idx,
                    "cycle_index": sho_cycle_idx,
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

    def test_stdio_mcp_client_can_fit_real_file_and_create_datasets(self):
        from pathlib import Path

        try:
            import SciFiReaders as sr
        except ImportError:
            self.skipTest("SciFiReaders is required for the stdio MCP integration test.")

        file_path = Path("/Users/rvv/Downloads/PTO_5x5.h5")
        if not file_path.exists():
            self.skipTest(f"Real BEPS fixture not found: {file_path}")

        reader = sr.NSIDReader(str(file_path))
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


if __name__ == "__main__":
    unittest.main()
