import unittest

import numpy as np

from sidpy.proc import mcp_server_beps as mcp_mod


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


if __name__ == "__main__":
    unittest.main()
