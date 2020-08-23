"""
The sidpy package

Submodules
----------

.. autosummary::
    :toctree: _autosummary

"""
from .__version__ import version as __version__
from . import base, hdf, io, proc, sid, viz
from .base import dict_utils, num_utils, string_utils
from .hdf import hdf_utils, reg_ref, dtype_utils, prov_utils
from .io import interface_utils
from .proc import comp_utils
from .sid.dimension import Dimension
from .sid.translator import Translator
from .sid.reader import Reader
from .sid.dataset import Dataset
from .viz import plot_utils
from .viz import jupyter_utils

__all__ = ['__version__']
# Traditional hierarchical approach - importing submodules
__all__ += base.__all__
__all__ += hdf.__all__
__all__ += io.__all__
__all__ += proc.__all__
__all__ += sid.__all__
__all__ += viz.__all__

# Making things easier by surfacing all low-level modules directly:
__all__ += ['dict_utils', 'num_utils', 'string_utils']
__all__ += ['hdf_utils', 'reg_ref', 'dtype_utils', 'prov_utils']
__all__ += ['interface_utils']
__all__ += ['comp_utils']
__all__ += ['Dimension', 'Translator', 'Dataset', 'Reader']
__all__ += ['plot_utils', 'jupyter_utils']
