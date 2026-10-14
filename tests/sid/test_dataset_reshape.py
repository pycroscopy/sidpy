import numpy as np
import dask.array as da
import pytest
from dask.callbacks import Callback

from sidpy import Dataset


@pytest.mark.parametrize('shape', [(2, 3), (6,), (-1,), ((2, 3),)])
def test_reshape_shape(shape):
    values = np.arange(6, dtype=np.float32)
    dataset = Dataset.from_array(da.from_array(values, chunks=3))
    expected_shape = shape[0] if isinstance(shape[0], tuple) else shape

    with pytest.warns(UserWarning, match='Dimensional information will be lost'):
        result = dataset.reshape(shape)

    expected = values.reshape(expected_shape)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result.compute(), expected)


@pytest.mark.parametrize('dtype', [np.float32, np.complex128, np.uint64])
def test_reshape_preserves_dtype_and_values(dtype):
    values = np.arange(6, dtype=dtype)
    if dtype == np.complex128:
        values += 2j
    elif dtype == np.uint64:
        values += np.uint64(2 ** 63)
    dataset = Dataset.from_array(values)

    with pytest.warns(UserWarning, match='Dimensional information will be lost'):
        result = dataset.reshape((2, 3))

    assert isinstance(result, da.Array)
    assert result.shape == (2, 3)
    assert result.dtype == values.dtype
    np.testing.assert_array_equal(result.compute(), values.reshape(2, 3))


@pytest.mark.parametrize('merge_chunks', [True, False])
@pytest.mark.parametrize('limit', [None, 32])
def test_reshape_options_remain_lazy(merge_chunks, limit):
    values = np.arange(24, dtype=np.int64).reshape(4, 6)
    source = da.from_array(values, chunks=(2, 3))
    dataset = Dataset.from_array(source)
    tasks = []
    with Callback(pretask=lambda key, dsk, state: tasks.append(key)):
        with pytest.warns(UserWarning, match='Dimensional information will be lost'):
            result = dataset.reshape((24,), merge_chunks=merge_chunks, limit=limit)

    expected = source.reshape((24,), merge_chunks=merge_chunks, limit=limit)
    assert not tasks
    assert result.shape == expected.shape
    assert result.chunks == expected.chunks
    np.testing.assert_array_equal(result.compute(), expected.compute())


def test_reshape_rejects_incompatible_size():
    dataset = Dataset.from_array(np.arange(6))
    with pytest.warns(UserWarning, match='Dimensional information will be lost'):
        with pytest.raises(ValueError, match='total size'):
            dataset.reshape((5,))
