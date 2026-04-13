# Fit BEPS Dataset Workflow

This is the small-model-friendly workflow for the common BEPS fitting task.

## Workflow

1. Read the file with the SciFiReaders MCP server.
2. Pass the original file path to the sidpy workflow tool.
3. Fit SHO over the full DC sweep for one cycle.
4. Derive the loop input by projecting the fitted SHO amplitude and phase
   with BGlib `projectLoop`.
5. Save the SHO and BEPS fit-parameter datasets.

The published workflow name is:

`analysis.fit_beps_dataset`

## Inputs

Use these as the starting placeholders:

- `file_path`: `/path/to/PTO_5x5.h5`
- `channel_name`: `Channel_000`
- `sho_cycle_index`: `1`
- `use_kmeans`: `false`
- `n_clusters`: `4`
- `scifireaders_return_mode`: `file`
- `sho_dataset_name`: `sho_fit_parameters`
- `beps_dataset_name`: `beps_fit_parameters`

## Tool Call Sequence

### 1. Read the file with SciFiReaders MCP

Call:

```python
read_scifireaders_file(
    file_path=file_path,
    return_mode="file",
)
```

This returns a small result object with an `output_file_path` field. Pass that
path if you want, but the sidpy workflow tool only needs the original
`source_file_path`.

### 2. Run the BEPS workflow in sidpy

Call:

```python
fit_beps_dataset_workflow_tool(
    source_file_path=file_path,
    channel_name=channel_name,
    dataset_name=file_path,
    cycle_index=sho_cycle_index,
    use_kmeans=use_kmeans,
    n_clusters=n_clusters,
    return_cov=False,
    loss="linear",
    sho_dataset_name=sho_dataset_name,
    beps_dataset_name=beps_dataset_name,
)
```

This one sidpy tool performs the full sequence internally:

1. Register the source file as a sidpy dataset.
2. Fit SHO over the full DC sweep for the selected cycle.
3. Save the SHO fit-parameter map as a `sidpy.Dataset` with a DC axis.
4. Project the fitted SHO amplitude and phase into a piezoresponse loop with
   BGlib `projectLoop`.
5. Fit the BEPS loop slice from that projected piezoresponse.
6. Save the BEPS fit-parameter map as a `sidpy.Dataset`.

## Expected Result

The tool returns:

- `source_dataset_id`
- `sho_dataset_id`
- `beps_dataset_id`
- `sho_dataset` with `fit_quality`
- `beps_dataset` with `fit_quality`

## Practical Notes

- Keep the workflow in this exact order: optionally read with SciFiReaders, then one sidpy workflow call.
- The SHO metadata stores the loop slice indices used by the derive step.
- The SHO metadata stores the selected cycle index used by the derive step.
- The loop fit uses the projected piezoresponse, not the raw loop waveform.
- The saved fit datasets preserve the `X` and `Y` axes from the source file.
