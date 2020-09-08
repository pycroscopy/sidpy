.. SIDpy documentation master file, created by
   sphinx-quickstart on Sun Jul 12 16:57:01 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

sidpy
=====

**Python utilities for storing, visualizing, and processing Spectroscopic and Imaging Data (SID)**

This utilities package supports other packages such as:

* Data formatting:

  * `pyNSID <https://pycroscopy.github.io/pyNSID/about.html>`_
  * `pyUSID <https://pycroscopy.github.io/pyUSID/about.html>`_

* Scientific analysis:

  * `pycroscopy <https://pycroscopy.github.io/pycroscopy/about.html>`_
  * `ScopeReaders <https://pycroscopy.github.io/ScopeReaders/about.html>`_
  * `BGlib <https://pycroscopy.github.io/BGlib/about.html>`_

Please use the side panel on the left for other documentation pages on ``sidpy``

Jump to our `GitHub project page <https://github.com/pycroscopy/sidpy>`_

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: SIDpy

   install
   getting_started
   external_guides
   contribute
   matlab
   contact


Source code API
---------------
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   sidpy

* :ref:`modindex`

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Examples

   notebooks/**/index
