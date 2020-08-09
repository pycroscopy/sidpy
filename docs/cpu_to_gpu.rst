GPU Computing Using CuPy
========================
:Authors: Emily Costa
:Created on: 08/07/2019

Enabling GPU computing, by implementing CuPy in the Bayesian Inference package of pycroscopy, was completed. 
The following are lessons learned during the exploration of implementing CuPy for GPU computing in pyUSID functions:

Expanding Dimensions
--------------------
``CuPy`` does not have a ``newaxis`` function, like ``NumPy`` does.
Instead of using new axis to add an additional dimension, you need to use ``cupy.expand_dims()``.
Also, note that cupy does not lose a dimension during operations with vectors, like ``numpy``,
so adding another dimension is often unnecessary as there are no singular dimensions in ``cupy``.
All vectors are converted into row vectors in ``numpy`` after being operated on,
which can be dealt with by adding a new axis and converting back into a column vector for further matrix operations.

The following is an example of how ``numpy``'s ``newawaxis`` function and how to use ``cupy``'s ``expand_dims`` in its place:

``numpy.newaxis``
~~~~~~~~~~~~~~~~~
  
Import all necessary modules

.. code-block:: python

  import numpy as np

1D array

.. code-block:: python

  arr = np.arange(5)
  arr.shape
  
``(5,)``

Make the 1D array becomes a row vector when an axis is inserted along 1st dimension
  
.. code-block:: python

  row_vec = arr[np.newaxis, :]
  row_vec.shape
  
``(1, 5)``

Make the 1D array becomes a column vector when an axis is inserted along 1st dimension
  
.. code-block:: python

  col_vec = arr[:, np.newaxis]
  col_vec.shape

``(5, 1)``
  
``cupy.expand_dims``
~~~~~~~~~~~~~~~~~~~~

Import all necessary modules

.. code-block:: python

  import cupy as cp

1D array
  
.. code-block:: python

  cp_arr = cp.arange(5)
  cp_arr.shape
  
``(5,)``

Make the 1D array becomes a row vector when an axis is inserted along 1st dimension
  
.. code-block:: python

  cp_row_vec = cp.expand_dims(cp_arr, axis=0)
  cp_row_vec.shape
  
``(1, 5)``

Make the 1D array becomes a column vector when an axis is inserted along 1st dimension
  
.. code-block:: python

  cp_col_vec = cp.expand_dims(cp_arr, axis=1)
  cp_col_vec.shape
  
``(5, 1)``
  
Append
------
``CuPy`` does not have an ``append`` function like ``numpy`` does. The ``append`` function in ``numpy`` appends values to the end of an array.

The following is an example of ``numpy``'s ``append`` function and how to use ``cupy``'s ``concatonate`` instead:

``numpy.append``
~~~~~~~~~~~~~~~~
  
.. code-block:: python

  x = np.array([1,2,3])
  y = [4,5,6]
  xy = np.append(x, y)
  xy
  
``array([1, 2, 3, 4, 5, 6])``
  
``cupy.concatenate``
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

  cp_x = cp.array([1,2,3])
  cp_y = cp.array([4,5,6])
  cp_xy = cp.concatenate([cp_x,cp_y], axis=0)
  cp_xy
  
``array([1, 2, 3, 4, 5, 6])``
