"""
Basic computational utilities
"""
fitter_present = True
try:
    from dask import distributed
except:
    fitter_present = False
if not fitter_present:
    __all__ = []
else:
    from . import fitter

    __all__ = ['fitter']

