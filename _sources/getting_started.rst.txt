Getting Started
===============
* Follow these :ref:`instructions <Installation>` to install SIDpy
* We have compiled a list of :ref:`handy tutorials <Tutorials on Basics>` on basic / prerequisite topics such as programming in python, hdf5 handling, etc.
* See our `examples <./notebooks/00_basic_usage/index.html>`_ to get started on creating and using your own SIDpy datasets.

  * Please see this `pyUSID tutorial for beginners <https://github.com/pycroscopy/pyUSID_Tutorial>`_ based on the examples on this project.
* Details regarding the definition, implementation, and guidelines for N-Dimensional Spectroscopy and Imaging Data (NSID) are available in `this document <https://pycroscopy.github.io/pyNSID/nsid.html>`_.
* If you are interested in contributing your code to SIDpy, please look at our :ref:`guidelines <Contributing code>`
* We also have a handy document for converting your :ref:`matlab code to python <Upgrading from Matlab>`.
* If you need detailed documentation on what is where and why, all our classes, functions, etc., please visit our :ref:`API <API Reference>`
* For a concise change-log, please see the `release history <https://github.com/pycroscopy/SIDpy/releases>`_.
* Please :ref:`get in touch <Contact us>` if you would like to use SIDpy and pyNSID for other new or mature scientific packages.
* Have questions? See our `FAQ <./faq.html>`_ to see if we have already answered them.dd
* Need more information? Please see our `Arxiv <https://arxiv.org/abs/1903.09515>`_ paper.
* Need help or need to get in touch with us? See our :ref:`contact <Contact us>` information.

Guide for python novices
~~~~~~~~~~~~~~~~~~~~~~~~
For the python novices by a python novice - **Nick Mostovych, Brown University**

#. Watch the video on `installing Anaconda <https://www.youtube.com/watch?v=YJC6ldI3hWk>`_
#. Follow instructions on the :ref:`installation <Installation>` page to install Anaconda.
#. Watch the `video tutorial <https://www.youtube.com/watch?v=HW29067qVWk>`_ on the Jupyter Notebooks
#. Read the whole :ref:`Tutorial on Basics page <Tutorials on Basics>`. Do NOT proceed unless you are familiar with basic python programming and usage.
#. Read `the document on the pyNSID data format <https://pycroscopy.github.io/pyNSID/nsid.html>`_. This is very important and highlights the advantages of using NSID. New users should not jump to the examples until they have a good understanding of the data format.
#. Depending on your needs, go through the recommended sequence of tutorials and examples (see 'EXAMPLES' on the side panel on the left)

Tips and pitfalls
~~~~~~~~~~~~~~~~~
For the python novices by a python novice - **Nick Mostovych, Brown University**

* Documentation and examples on this website are for the **latest** version of SIDpy. If something does not work as shown on this website,
  chances are that you may be using an older version of pyUSID. Follow the instructions to :ref:`update SIDpy to the latest version <updating sidpy>`
* pyUSID has excellent documentation (+ examples too) for all functions. If you are ever confused with the usage of a
  function or class, you can get help in numerous ways:

  * If you are using jupyter notebooks, just hit the ``Shift+Tab`` keys after typing the name of your function.
    See `this quick video <https://www.youtube.com/watch?v=TgqMK1SG7XI>`_ for a demo.
    E.g. - type ``sidpy.Dataset(``. Hit ``Shift+Tab`` twice or four times. You should be able to see the documentation for the
    class / function to learn how to supply inputs / extract outputs
  * Use the search function and reference the source code in the :ref:`API section <API Reference>` for detailed comments.
    Most detailed questions are answered there.
* Many functions in SIDpy have a ``verbose`` keyword argument that can be set to ``True`` to get detailed print logs of intermediate steps in the function.
  This is **very** handy for debugging code

If there are tips or pitfalls you would like to add to this list, please :ref:`get in touch to us <Contact us>`
