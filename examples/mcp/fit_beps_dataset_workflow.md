# Fit BEPS Dataset Workflow

This is a small-model-friendly workflow for the common task:

1. Read a BEPS file with `SciFiReaders`
2. Fit a loop slice with `fit_beps_loops_tool`
3. Fit an SHO slice with `fit_sho_response_tool`
4. Save both parameter maps as `sidpy.Dataset` objects

The workflow name published by the MCP server is:

`analysis.fit_beps_dataset`

## Inputs

Use these as the starting placeholders:

- `file_path`: `/path/to/PTO_5x5.h5`
- `channel_name`: `Channel_000`
- `beps_frequency_index`: `23`
- `beps_cycle_index`: `0`
- `sho_dc_index`: `49`
- `sho_cycle_index`: `1`
- `use_kmeans`: `true`
- `n_clusters`: `4`

## Workflow Shape

### 1. Read the file

Use `SciFiReaders.NSIDReader(file_path)` and read the channel named `channel_name`.

Expected result:

- a `sidpy.Dataset`
- shape similar to `(x, y, frequency, dc_offset, cycle)`

### 2. Build the BEPS slice

Take the real-valued loop slice:

```python
beps_data = data[:, :, beps_frequency_index, :, beps_cycle_index].real
dc_voltage = data._axes[3].values
```

Then call:

```python
fit_beps_loops_tool(
    data=beps_data,
    dc_voltage=dc_voltage,
    use_kmeans=use_kmeans,
    n_clusters=n_clusters,
    return_cov=False,
    loss="linear",
    dataset_name="beps_loop_fit",
)
```

Expected output:

- `fit_kind = "beps_loop"`
- `parameter_shape = [x, y, 9]`
- `parameter_labels = LOOP_PARAMETER_LABELS`

### 3. Save the BEPS fit parameters

Call `create_dataset_tool` with the BEPS parameter array.

Use:

- `quantity = "fit_parameter"`
- `units = "a.u."`
- spatial dimensions for `X` and `Y`
- spectral dimension `fit_parameter` with 9 entries

Attach metadata such as:

- `fit_kind`
- `source_dataset`
- `source_slice`

### 4. Build the SHO slice

Take the complex frequency slice:

```python
sho_data = data[:, :, :, sho_dc_index, sho_cycle_index]
frequency = data._axes[2].values
```

Then call:

```python
fit_sho_response_tool(
    real_data=sho_data.real,
    imag_data=sho_data.imag,
    frequency=frequency,
    use_kmeans=use_kmeans,
    n_clusters=n_clusters,
    return_cov=False,
    loss="linear",
    dataset_name="sho_response_fit",
)
```

Expected output:

- `fit_kind = "sho"`
- `parameter_shape = [x, y, 4]`
- `parameter_labels = SHO_PARAMETER_LABELS`

### 5. Save the SHO fit parameters

Call `create_dataset_tool` again with the SHO parameter array.

Use:

- `quantity = "fit_parameter"`
- `units = "a.u."`
- spatial dimensions for `X` and `Y`
- spectral dimension `fit_parameter` with 4 entries

Attach metadata such as:

- `fit_kind`
- `source_dataset`
- `source_slice`

## Practical Notes

- Keep the number of tool calls small and linear.
- Read once, slice once, fit once, save once.
- If the model gets confused, prefer this exact order:
  1. read
  2. fit BEPS
  3. save BEPS
  4. fit SHO
  5. save SHO
- The saved fit-parameter datasets should preserve the `X` and `Y` axes from the source file.
