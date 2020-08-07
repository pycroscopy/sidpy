"""
The sidpy package

Submodules
----------

.. autosummary::
    :toctree: _autosummary

"""
from . import base, hdf, io, proc, sid, viz
from .__version__ import version as __version__

__all__ = ['__version__']
__all__ += base.__all__
__all__ += hdf.__all__
__all__ += io.__all__
__all__ += proc.__all__
__all__ += sid.__all__
__all__ += viz.__all__
