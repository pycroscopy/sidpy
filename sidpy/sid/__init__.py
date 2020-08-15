"""
Spectroscopy and Imaging Data related classes

Submodules
----------

.. autosummary::
    :toctree: _autosummary

    Dimension
    Dataset
    Translator
"""

from .dimension import Dimension
from .translator import Translator
from .dataset import Dataset

__all__ = ['Dimension', 'Translator', 'Dataset']
