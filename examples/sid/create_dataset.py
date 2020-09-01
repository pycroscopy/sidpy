# Ensure python 3 compatibility:
from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
import sys

sys.path.append('../../')
import sidpy as sid
print(sid.__version__)

data_set = sid.Dataset.from_array(np.zeros([4, 5, 10]), name='zeros')
print(data_set)

