"""
===============================================================================
User Interface utilities
===============================================================================

**Suhas Somnath**

8/12/2017

**This is a short walk-through of useful user interface utilities available in sidpy**

Introduction
--------------

.. tip::
    You can download and run this document as a Jupyter notebook using the link at the bottom of this page.
"""

# Ensure python 3 compatibility:
from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
from warnings import warn
import matplotlib.pyplot as plt
import subprocess
import sys


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
# Package for downloading online files:
try:
    import sidpy
except ImportError:
    warn('sidpy not found.  Will install with pip.')
    import pip
    install('sidpy')
    import sidpy

########################################################################################################################
# Communication utilities
# ========================
#
# check_ssh()
# -----------
# When developing workflows that need to work on remote or virtual machines in addition to one's own personal computer
# such as a laptop, this function is handy at letting the developer know where the code is being executed

print('Running on remote machine: {}'.format(sidpy.interface_utils.check_ssh()))