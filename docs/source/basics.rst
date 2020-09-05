About SIDpy
===========

**Python utilities for storing, visualizing, and processing Spectroscopic and Imaging Data (SID)**

This package supports other scientific packages such as:

* `pyNSID <https://pycroscopy.github.io/pyNSID/about.html>`_
* `pyUSID <https://pycroscopy.github.io/pyUSID/about.html>`_
* `pycroscopy <https://pycroscopy.github.io/pycroscopy/about.html>`_

Please use the side panel on the left for other documentation pages on ``sidpy``


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
    E.g. - type ``px.USIDataset(``. Hit ``Shift+Tab`` twice or four times. You should be able to see the documentation for the
    class / function to learn how to supply inputs / extract outputs
  * Use the search function and reference the source code in the :ref:`API section <API Reference>` for detailed comments.
    Most detailed questions are answered there.
* Many functions in SIDpy and pyNSID have a ``verbose`` keyword argument that can be set to ``True`` to get detailed print logs of intermediate steps in the function.
  This is **very** handy for debugging code

If there are tips or pitfalls you would like to add to this list, please :ref:`get in touch to us <Contact us>`

Installation
============

Preparing for sidpy
-------------------
`sidpy <https://github.com/pycroscopy/sidpy>`_ requires many commonly used scientific and numeric python packages such as numpy, h5py etc.
To simplify the installation process, we recommend the installation of
`Anaconda <https://www.anaconda.com/distribution/>`_ which contains most of the prerequisite packages,
`conda <https://conda.io/docs/>`_ - a package / environment manager,
as well as an `interactive development environment <https://en.wikipedia.org/wiki/Integrated_development_environment>`_ - `Spyder <https://www.coursera.org/learn/python-programming-introduction/lecture/ywcuv/introduction-to-the-spyder-ide>`_.

Do you already have Anaconda installed?

- No?

  - `Download and install Anaconda <https://www.anaconda.com/download/>`_ for Python 3.6

- Yes?

  - Is your Anaconda based on python 2.7, 3.4+?

    - No?

      - Uninstall existing Python / Anaconda distribution(s).
      - Restart computer
    - Yes?

      - Proceed to install sidpy

Compatibility
~~~~~~~~~~~~~
* sidpy is compatible with python 2.7, and 3.4 onwards. Please raise an issue if you find a bug.
* We do not support 32 bit architectures
* We only support text that is UTF-8 compliant due to restrictions posed by HDF5

Terminal
--------
Installing, uninstalling, or updating sidpy (or any other python package for that matter) can be performed using the ``Terminal`` application.
You will need to open the Terminal to type any command shown on this page.
Here is how you can access the Terminal on your computer:

* Windows - Open ``Command Prompt`` by clicking on the Start button on the bottom left and typing ``cmd`` in the search box.
  You can either click on the ``Command Prompt`` that appears in the search result or just hit the Enter button on your keyboard.

  * Note - be sure to install in a location where you have write access.  Do not install as administrator unless you are required to do so.
* MacOS - Click on the ``Launchpad``. You will be presented a screen with a list of all your applications with a search box at the top.
  Alternatively, simultaneously hold down the ``Command`` and ``Space`` keys on the keyboard to launch the ``Spotlight search``.
  Type ``terminal`` in the search box and click on the ``Terminal`` application.
* Linux (e.g - Ubuntu) - Open the Dash by clicking the Ubuntu (or equivalent) icon in the upper-left, type "terminal".
  Select the Terminal application from the results that appear.

Installing sidpy
-----------------
1. Ensure that a compatible Anaconda distribution has been successfully installed
2. Open a `terminal <#terminal>`_ window.
3. You can now install sidpy via **either** the ``pip`` or ``conda`` methods shown below.
   Type the following commands into the terminal / command prompt and hit the Return / Enter key:

   * pip:

     .. code:: bash

        pip install sidpy

   * conda:

     .. code:: bash

        conda config --add channels conda-forge
        conda install sidpy

Offline installation
~~~~~~~~~~~~~~~~~~~~
In certain cases, you may need your python packages to work on a computer
(typically the computer that controls a scientific instrument) that is not connected to the internet.
In such cases, the aforementioned routes will not work. Please follow these instructions instead:

#. Recall that sidpy requires python and several other packages. Therefore, you will need to:

   #. Download the `Anaconda installer <https://www.anaconda.com/download/>`_ from a computer is online
   #. Copy the installer onto the target computer via a USB pen drive
   #. Install Anaconda
#. Download the sidpy repository from GitHub via `this link <https://github.com/pycroscopy/sidpy/archive/master.zip>`_
#. Copy the resultant zip file to the offline computer via a portable storage device like a USB pen drive
#. Unzip the zip file in the offline computer.
#. Open a `terminal <#terminal>`_ window
#. Navigate to the folder where you unzipped the contents of the zip file via ``cd`` commands
#. Type the following command:

   .. code:: bash

     python setup.py install

  
Installing from a specific branch (advanced users **ONLY**)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Note that we do not recommend installing sidpy this way since branches other than the master branch may contain bugs.

.. note::
   Windows users will need to install ``git`` before proceeding. Please type the following command in the Command Prompt:

   .. code:: bash

     conda install git

Install a specific branch of sidpy (``dev`` in this case):

.. code:: bash

  pip install -U git+https://github.com/pycroscopy/sidpy@dev

  
Updating sidpy
--------------

We recommend periodically updating your conda / anaconda distribution. Please see :ref:`these instructions to update anaconda <Updating packages>`.

If you already have sidpy installed and want to update to the latest version, use the following command in a terminal window:

* If you originally installed sidpy via ``pip``:

  .. code:: bash

    pip install -U --no-deps sidpy
  
  If it does not work try reinstalling the package:

  .. code:: bash

    pip uninstall sidpy
    pip install sidpy
* If you originally installed sidpy via ``conda``:

  .. code:: bash

    conda update sidpy

Other software
--------------
We recommend `HDF View <https://support.hdfgroup.org/products/java/hdfview/>`_ for exploring HDF5 files generated by and used in sidpy.


Tutorials on Basics
====================
For those who are new to python and data analytics, we highly encourage you to
go through `Prof. Josh Agar's tutorials <https://github.com/jagar2/Fall_2019_Data_Analysis_and_Machine_Learning_for_Experimentalists>`_
for a throrough primer on all the basic concepts.

Here are a list of other tutorials from other websites and sources that describe some of the many important topics
on reading, using / running and writing code:

.. contents:: :local:

Python and  packages
--------------------
There are several concepts such as file operations, parallel computing, etc.
that are heavily used and applied in pyUSID. Most of these concepts are realized using add-ons or packages in
python. Here is a compilation of useful tutorials:

Python
~~~~~~
The following tutorials go over the basics of python programming:

* `Official Python tutorial <https://docs.python.org/3/tutorial/>`_
* The `Hitchhiker guide to Python <http://docs.python-guide.org/en/latest/>`_
* Introduction to programming in `Python 3 <https://pythonprogramming.net/beginner-python-programming-tutorials/>`_
* Tutorials on a broad spectrum of `real-world use topics <https://automatetheboringstuff.com>`_
* `O'Riley <https://greenteapress.com/wp/think-python/>`_ has a nice book on Python too.
* A nice guide on `intermediate Python <http://book.pythontips.com/en/latest/index.html>`_
* Our own `crash course on the basics of python <https://github.com/pycroscopy/CNMS_UM_2018_SPIMA>`_

HDF5 and h5py
~~~~~~~~~~~~~
Our software packages - ``sidpy``, ``pyUSID``, ``pyNSID`` are all
designed to be file-centric, we highly recommend learning more about HDF5 and h5py:

* `Basics of HDF5 <https://portal.hdfgroup.org/display/HDF5/Learning+HDF5>`_ (especially the last three tutorials)
* `Quick start <http://docs.h5py.org/en/latest/quick.html>`_ to h5py
* Another `tutorial on HDF5 and h5py <https://www.nersc.gov/assets/Uploads/H5py-2017-Feb23.pdf>`_
* The `O-Reilly book <http://shop.oreilly.com/product/0636920030249.do>`_ where we learnt h5py

Installing software
-------------------
python
~~~~~~~
`Anaconda <https://www.anaconda.com/download/>`_ is a popular source for python which also comes with a large number of popular scientific python packages that are all correctly compiled and installed in one go.
Tutorial for `installing Anaconda <https://www.youtube.com/watch?v=YJC6ldI3hWk>`_ (Python + all necessary packages)

python packages
~~~~~~~~~~~~~~~~
Two popular methods for installing packages in python are:

* `pip <https://packaging.python.org/tutorials/installing-packages/>`_:
    * included with basic python and standard on Linux and Mac OS
    * Works great for installing pure python and other simple packages
* `conda <https://conda.io/docs/user-guide/tasks/manage-pkgs.html>`_
    * included with Anaconda installation
    * Ideally suited for installing packages that have complex dependencies
* Here's a nice tutorial on `installing packages using both pip and conda <https://www.youtube.com/watch?v=Z_Kxg-EYvxM>`_

Updating packages
~~~~~~~~~~~~~~~~~
Following `these instructions <https://stackoverflow.com/questions/45197777/how-do-i-update-anaconda>`_, open a terminal or the command prompt (Windows) and type:

.. code:: bash

    conda update conda
    conda update anaconda

Note that you could use the following line instead of or in addition to ``conda update anaconda`` but it can lead to incompatible package versions

.. code:: bash

    conda update --all

Note that this does **not** update python itself.

Upgrading python
~~~~~~~~~~~~~~~~
Follow these instructions to `upgrade python using conda <https://conda.io/docs/user-guide/tasks/manage-python.html#updating-or-upgrading-python>`_ to the latest or specific version

Writing code
------------
Text Editors
~~~~~~~~~~~~
These software often do not have any advanced features found in IDEs such as syntax highlighting,
real-time code-checking etc. but are simple, and most importantly, open files quickly.  Here are some excellent
text editors for each class of operating system:

* Mac OS - `Atom <https://atom.io/>`_
* Linux - `gEdit <https://wiki.gnome.org/Apps/Gedit>`_
* Windows - `Notepad++ <https://notepad-plus-plus.org/>`_

Integrated Development Environments (IDE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These applications often come with a built-in text editor, code management
capabilities, a python console, a terminal, integration with software repositories, etc. that make them ideal for
executing and developing code. We only recommend two IDEs at this point: Spyder for users, PyCharm for developers.
Both of these work in Linux, Mac OS, and Windows.

* `Spyder <https://en.wikipedia.org/wiki/Spyder_(software)>`_ is a great IDE that is simple and will be immediately
  familiar for users of Matlab.

    * `Basics of Spyder <https://www.youtube.com/watch?v=a1P_9fGrfnU>`_
    * `Python  with Spyder <http://datasciencesource.com/python-with-spyder-tutorial/>`_ - this was written with
      Python 2.7 in mind, but most concepts will still apply

* `Pycharm <https://www.jetbrains.com/pycharm/>`_

    * Official `PyCharm Tutorial <https://confluence.jetbrains.com/display/PYH/PyCharm+Tutorials>`_ from Jetbrains

Jupyter Notebooks
~~~~~~~~~~~~~~~~~
These are `interactive documents <http://jupyter.org/>`_ containing live cells with code, equations,
visualizations, and narrative text. The interactive nature of the document makes Jupyter notebooks an ideal medium for
conveying information and a narrative. These documents are neither text editors nor IDEs and are a separate category.

* Notebook `basics <http://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb>`_
* `Video <https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook>`_ tutorial
* Another `video overview <https://www.youtube.com/watch?v=HW29067qVWk>`_.

Software development basics
---------------------------
This section is mainly focused on the other tools that are mainly necessary for those interested in developing their own
code and possibly contributing back to sidpy.

Environments
~~~~~~~~~~~~
Environments allow users to set up and segregate software sandboxes. For example, one could set up separate environments
in python 2 and 3 to ensure that a certain desired code works in both python 2 and 3. For python users, there are two
main and popular modes of creating and managing environments - **virtual environments** and **conda environments**.

* `Virtual environment <https://docs.python.org/3/tutorial/venv.html>`_
    * Basic python ships with virtual enviroments. Anaconda is not required for this
    * How to `use venv <http://www.pythonforbeginners.com/basics/how-to-use-python-virtualenv>`_

* Conda environments
    * `Basics  <https://conda.io/docs/user-guide/getting-started.html>`_ of Conda
    * How to `manage environments in conda <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_
    * `Managing Python Environments <https://www.youtube.com/watch?v=EGaw6VXV3GI>`_ with Conda

Version control
~~~~~~~~~~~~~~~
`Version control <https://vimeo.com/41027679>`_ is a tool used for managing changes in code over time. It lifts the
burden of having to check for changes line-by-line when multiple people are working on the same project. For example,
sidpy uses `Git <https://git-scm.com/>`_, the most popular version control software (VCS) for tracking changes etc. By default, git
typically only comes with a command-line interface. However, there are several software packages that provide a
graphical user interface on top of git. One other major benefit of using an IDE over jupyter or a text editor is that
(some) IDEs come with excellent integration with VCS like Git. Here are a collection of useful resources to get you
started on git:

* Tutorial on the `basics of git <https://www.atlassian.com/git/tutorials>`_
* Our favorite git client - `GitKraken <https://support.gitkraken.com/>`_
* Our favorite IDE with `excellent integration with Git: PyCharm <https://www.youtube.com/watch?v=vIReqoQYud8>`_
* Our own guide to `setting up and using git with PyCharm <https://github.com/pycroscopy/sidpy/blob/master/docs/Using%20PyCharm%20to%20manage%20repository.pdf>`_


Guidelines for Contribution
============================

We would like to thank you and several others who have offered / are willing to contribute their code.
We are more than happy to add your code to this project.
Just as we strive to ensure that you get the best possible software from us, we ask that you do the same for others.
We do **NOT** ask that your code be as efficient as possible. Instead, we have some simpler and easier requests.
We have compiled a list of best practices below with links to additional information.
If you are confused or need more help, please feel free to `contact us <./contact.html>`_.

Before you begin
----------------
Please consider familiarizing yourself with the `examples <./auto_examples/index.html>`_ and `documentation <./api.html>`_
on functionality available in sidpy so that you can use the available functionality to simplify your code
in addition to avoiding the development of duplicate code.


Structuring code
----------------

General guidelines
~~~~~~~~~~~~~~~~~~
* Encapsulate independent sections of your code into functions that can be used individually if required.
* Ensure that your code (functions) is well documented (`numpy format <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_)
  - expected inputs and outputs, purpose of functions
* Please avoid very short names for variables like ``i`` or ``k``. This makes it challenging to follow code, find and fix bugs.
* Please consider using packages that are easy to install on Windows, Mac, and Linux.
  It is quite likely that packages included within Anaconda
  (which has a comprehensive list packages for science and data analysis + visualization) can handle most needs.
  If this is not possible, try to use packages that are easy to to install (pip install).
  If even this is not possible, try to use packages that at least have conda installers.
* Follow best practices for `PEP8 compatibility <https://www.datacamp.com/community/tutorials/pep8-tutorial-python-code>`_.
  The easiest way to ensure compatibility is to set it up in your code editor.
  `PyCharm <https://blog.jetbrains.com/pycharm/2013/02/long-awaited-pep-8-checks-on-the-fly-improved-doctest-support-and-more-in-pycharm-2-7/>`_ does this by default.
  So, as long as PyCharm does not raise many warning, your code is beautiful!

sidpy-specific guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~
* Recall that sidpy is a general collection of tools that can help store, analyze, visualize, and process spectroscopy and imaging data.
  Any code specific to the Universal Spectroscopic and Imaging Data (USID) or
  N-Dimensional Spectroscopic and Imaging Data (NSID) should go into pyUSID or pyNSID respectively.
  Code that provides scientific functionality goes into pycroscopy.
* Please ensure that your code files fit into our :ref:`package structure <API Reference>` (``base``, ``hdf``, ``io``, ``proc``, ``sid`` and ``viz``)
* Once you decide where your code will sit, please use relative import statements instead of absolute / external paths.
  For example, if you are contributing code for a new submodule within ``sidpy.hdf``, you will need to turn your import statements and code from something like:

  .. code-block:: python

     import sidpy
     ...
     sidpy.hdf.hdf_utils.print_tree(hdf_file_handle)
     x_dim = sidpy.sid.Dimension(...)

  to:

  .. code-block:: python

     from ..hdf.hdf_utils import print_tree
     from ..sid import Dimension
     ...
     print_tree(hdf_file_handle)
     x_dim = Dimension(...)

You can look at our code in our `GitHub project <https://github.com/pycroscopy/sidpy>`_ to get an idea of how we organize, document, and submit our code.

Contributing code
-----------------
We recommend that you follow the steps below. Again, if you are ever need help, please contact us:

1. Learn ``git`` if you are not already familiar with it. See our :ref:`compilation of tutorials and guides <Tutorials on Basics>`,
   especially `this one <https://github.com/pycroscopy/sidpy/blob/master/docs/Using%20PyCharm%20to%20manage%20repository.pdf>`_.
2. Create a ``fork`` of sidpy - this creates a separate copy of the entire sidpy repository under your user ID.
   For more information see `instructions here <https://help.github.com/articles/fork-a-repo/>`_.
3. Once inside your own fork, you can either work directly off ``master`` or create a new branch.
4. Add / modify code
5. ``Commit`` your changes (equivalent to saving locally on your laptop). Do this regularly.
6. Repeat steps 4-5.
7. After you reach a certain milestone, ``push`` your commits to your ``remote branch``.
   This synchronizes your changes with the GitHub website and is similar to the Dropbox website /service making note of changes in your documents.
   To avoid losing work due to problems with your computer, consider ``pushing commits`` once at least every day / every few days.
8. Repeat steps 4-7 till you are ready to have your code added to the parent sidpy repository.
   At this point, `create a pull request <https://help.github.com/articles/creating-a-pull-request-from-a-fork/>`_.
   Someone on the development team will review your ``pull request``. If any changes are req and then ``merge`` these changes to ``master``.

Writing tests
-------------
Software can become complicated very quickly through a complex interconnected web of dependencies, etc.
Adding or modifying code at one location may break some use case or code in a different location.
Unit tests are short functions that test to see if functions / classes respond in the expected way given some known inputs.
Unit tests are a good start for ensuring that you spend more time using code than fixing it. New functions / classes must be accompanied with unit tests.

Writing examples
----------------
Additionally, examples on how to use the new code must also be added so others are aware about how to use the code.
You can now do it by simply adding a Jupyter notebook with your tutorial/example to the `notebooks <https://github.com/ziatdinovmax/sidpy/tree/master/notebooks>`_ folder.

Contact us
----------
If you find any bugs or if you want a feature added to sidpy, please raise an
`issue <https://github.com/pycroscopy/sidpy/issues>`_. You will need a (free) Github account to do this.

When reporting bugs, please provide a (minimal) script / snippet that reproduces

Credits
-------
The core sidpy team consists of:

* `@ssomnath <https://github.com/ssomnath>`_ (Suhas Somnath)
* `@gduscher <https://github.com/gduscher>`_ (Prof. Gerd Duscher)

Substantial contributions from many developers including:

* `@CompPhysChris <https://github.com/CompPhysChris>`_ (Chris R. Smith)
* and many more
the error(s) you are facing, the full description of the error(s), details
regarding your computer, operating system, python, sidpy and other related package versions etc.
These details will help us solve your problem a lot faster.

Upgrading from Matlab
=====================
**Chris R. Smith**

Here are some one-to-one translations for many popular functions in Matlab and python that should make it easier to switch from Matlab to Python

System functions
----------------
+------------------+-------------------+-------------+
| Matlab Function  | Python Equivalent | Description |
+==================+===================+=============+
| addpath          | sys.path.append   | Add to path |
+------------------+-------------------+-------------+

File I/O
--------
+-----------------+--------------------------------------------+-------------------------------------------------------+
| Matlab Function | Python Equivalent                          | Description                                           |
+=================+============================================+=======================================================+
| dlmread         | either read and parse or skimage.io.imread | Read ASCII-delimited file of numeric data into matrix |
+-----------------+--------------------------------------------+-------------------------------------------------------+
| imread          | pyplot.imread                              | read image file; N is number of files used            |
+-----------------+--------------------------------------------+-------------------------------------------------------+

Data Type
---------
+-----------------+-------------------+-----------------------------------------------+
| Matlab Function | Python Equivalent | Description                                   |
+=================+===================+===============================================+
| int             | numpy.int         | Convert data to signed integer                |
+-----------------+-------------------+-----------------------------------------------+
| double          | numpy.float       | Convert data to double                        |
+-----------------+-------------------+-----------------------------------------------+
| real            | numpy.real        | Return the real part of a complex number      |
+-----------------+-------------------+-----------------------------------------------+
| imag            | numpy.imag        | Return the imaginary part of a complex number |
+-----------------+-------------------+-----------------------------------------------+

Mathematics
-----------
+------------------+-------------------------------+-------------------------------+
| Matlab Function  | Python Equivalent             | Description                   |
+==================+===============================+===============================+
| sqrt             | math.sqrt or numpy.sqrt       | Square root                   |
+------------------+-------------------------------+-------------------------------+
| erf              | math.erf or scipy.special.erf | Error function                |
+------------------+-------------------------------+-------------------------------+
| atan2            | math.erf or numpy.atan2       | Four-quadrant inverse tangent |
+------------------+-------------------------------+-------------------------------+
| abs              | abs or numpy.abs              | Absolute value                |
+------------------+-------------------------------+-------------------------------+
| exp              | exp or numpy.exp              | Exponential function          |
+------------------+-------------------------------+-------------------------------+
| sin              | sin or numpy.sin              | Sine function                 |
+------------------+-------------------------------+-------------------------------+

Array Creation
--------------
+-----------------+----------------------------+-------------------------------------------------+
| Matlab Function | Python Equivalent          | Description                                     |
+=================+============================+=================================================+
| zeros           | numpy.zeros                | Create an array of zeros                        |
+-----------------+----------------------------+-------------------------------------------------+
| meshgrid        | numpy.meshgrid             | Create grid of coordinates in 2 or 3 dimensions |
+-----------------+----------------------------+-------------------------------------------------+
| ndgrid          | numpy.mgrid or numpy.ogrid | Rectangular grid in N-D space                   |
+-----------------+----------------------------+-------------------------------------------------+

Advanced functions
------------------
+-----------------+------------------------------------------------------+------------------------------------------------------------------------------------------------------+
| Matlab Function | Python Equivalent                                    | Description                                                                                          |
+=================+======================================================+======================================================================================================+
| permute         | numpy.transpose                                      | Rearrange dimensions of N-dimensional array                                                          |
+-----------------+------------------------------------------------------+------------------------------------------------------------------------------------------------------+
| angle           | numpy.angle                                          | Phase angles for elements in complex array                                                           |
+-----------------+------------------------------------------------------+------------------------------------------------------------------------------------------------------+
| max             | numpy.max                                            | Return the maximum element in an array                                                               |
+-----------------+------------------------------------------------------+------------------------------------------------------------------------------------------------------+
| min             | numpy.min                                            | Return the minimum element in an array                                                               |
+-----------------+------------------------------------------------------+------------------------------------------------------------------------------------------------------+
| reshape         | numpy.reshape                                        | Reshape array                                                                                        |
+-----------------+------------------------------------------------------+------------------------------------------------------------------------------------------------------+
| mean            | numpy.mean                                           | Take mean along specified dimension                                                                  |
+-----------------+------------------------------------------------------+------------------------------------------------------------------------------------------------------+
| size            | numpy.size                                           | get the total number of entries in an array                                                          |
+-----------------+------------------------------------------------------+------------------------------------------------------------------------------------------------------+
| cell2mat        | numpy.vstack([numpy.hstack(cell) for cell in cells]) | converts data structure from cell to mat; joins multiple arrays of different sizes into single array |
+-----------------+------------------------------------------------------+------------------------------------------------------------------------------------------------------+
| repmat          | numpy.tile                                           | Repeat copies of an array                                                                            |
+-----------------+------------------------------------------------------+------------------------------------------------------------------------------------------------------+
| unwrap          | np.unwrap                                            | Shift the phase of an array so that there are no jumps of more than the desired angle (default pi)   |
+-----------------+------------------------------------------------------+------------------------------------------------------------------------------------------------------+

Array Indexing
--------------
+-----------------+-------------------+--------------------------------------------------------------------+
| Matlab Function | Python Equivalent | Description                                                        |
+=================+===================+====================================================================+
| find            | numpy.where       | Find all indices of a matrix for which a logical statement is true |
+-----------------+-------------------+--------------------------------------------------------------------+
| isnan           | numpy.isnan       | checks each array entry to see if it is NaN                        |
+-----------------+-------------------+--------------------------------------------------------------------+
| isinf           | numpy.isinf       | checks each array entry to see if it is Inf                        |
+-----------------+-------------------+--------------------------------------------------------------------+
| ischar          | numpy.ischar      | checks each array entry to see if it is a character                |
+-----------------+-------------------+--------------------------------------------------------------------+

Advanced functions
------------------
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Matlab Function | Python Equivalent                                                            | Description                                                                                                                                                                                                                          |
+=================+==============================================================================+======================================================================================================================================================================================================================================+
| fft2            | numpy.fft.fft2                                                               | 2D fast Fourier transform                                                                                                                                                                                                            |
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| fftshift        | numpy.fft.fftshift                                                           | shift zero-frequency component to the center of the spectrum                                                                                                                                                                         |
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ifftshift       | numpy.fft.ifftshift                                                          | inverse fftshift                                                                                                                                                                                                                     |
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ifft2           | numpy.fft.fifft2                                                             | inverse 2d fft                                                                                                                                                                                                                       |
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| interp2         | scipy.interpolate.RectBivariateSpline or scipy.interpolate.interp2           | Interpolation for 2-D gridded data in meshgrid format                                                                                                                                                                                |
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| imshowpair      | skimage.measure.structural_similarity                                        | Compare differences between 2 images                                                                                                                                                                                                 |
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| imregconfig     |                                                                              | Creates configurations to perform intensity-based image registration                                                                                                                                                                 |
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| imregister      |                                                                              | Intensity-based image registration                                                                                                                                                                                                   |
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| imregtform      | skimage.feature.register_translation or skimage.transform.estimate_transform | Estimate geometric transfomation to align two images                                                                                                                                                                                 |
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| imwarp          | skimage.transform.warp                                                       | Apply geometric transformation to an image                                                                                                                                                                                           |
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| imref2d         |                                                                              | Reference 2d image to xy-coordinates                                                                                                                                                                                                 |
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| corr2           | scipy.signal.correlate2d                                                     | 2d correlation coefficient                                                                                                                                                                                                           |
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| optimset        |                                                                              | Create of edit optimizations options for passing to fminbnd, fminsearch, fzero, or lsqnonneg                                                                                                                                         |
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| lsqcurvefit     | scipy.optimize.curve_fit                                                     | Solve nonlinear curve-fitting problems                                                                                                                                                                                               |
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| fastica         | sklearn.decomposition.FastICA                                                | fast fixed-point algorithm for independent component analysis and projection pursuit                                                                                                                                                 |
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| kmeans          | sklearn.cluster.Kmeans                                                       | kmeans clustering                                                                                                                                                                                                                    |
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| fsolve          | scipy.optimize.root(func, x0, method='anderson')                             | Root finding.  Scipy does not have a trust-region dogleg method that functions exactly like Matlab's fsolve.  The 'anderson' method reproduces the results in many cases.  Other methods may need to be explored for other problems. |
+-----------------+------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Basic Plotting
--------------
+-----------------+--------------------------------------------+-------------------------------------------------------------------------------------------------------+
| Matlab Function | Python Equivalent                          | Description                                                                                           |
+=================+============================================+=======================================================================================================+
| figure          | matplotlib.pyplot.figure                   | Create a new figure object                                                                            |
+-----------------+--------------------------------------------+-------------------------------------------------------------------------------------------------------+
| clf             | figure.clf                                 | clear figure; shouldn't be needed in Python since each figure will be a unique object                 |
+-----------------+--------------------------------------------+-------------------------------------------------------------------------------------------------------+
| subplot         | figure.subplots or figure.add_subplot      | 1st creates a set of subplots in the figure, 2nd creates one subplot and adds it to the figure        |
+-----------------+--------------------------------------------+-------------------------------------------------------------------------------------------------------+
| plot            | figure.plot or axes.plot                   | Add lineplot to current figure                                                                        |
+-----------------+--------------------------------------------+-------------------------------------------------------------------------------------------------------+
| title           | object.title                               | Title of plot; better to define on object creation if possible                                        |
+-----------------+--------------------------------------------+-------------------------------------------------------------------------------------------------------+
| xlabel          | axes.xlabel                                | Label for the x-axis of plot                                                                          |
+-----------------+--------------------------------------------+-------------------------------------------------------------------------------------------------------+
| ylabel          | axes.ylabel                                | Label for the y-axis of plot                                                                          |
+-----------------+--------------------------------------------+-------------------------------------------------------------------------------------------------------+
| imagesc         | pyplot.imshow or pyplot.matshow            | Scale image data to full range of colormap and display                                                |
+-----------------+--------------------------------------------+-------------------------------------------------------------------------------------------------------+
| axis            | axes.axis                                  | Axis properties                                                                                       |
+-----------------+--------------------------------------------+-------------------------------------------------------------------------------------------------------+
| surf            | axes3d.plot_surface or axes3d.plot_trisurf | Plot a 3d surface, need to uses mpl_toolkits.mplot3d and Axes3d; which you use depends on data format |
+-----------------+--------------------------------------------+-------------------------------------------------------------------------------------------------------+
| shading         |                                            | Set during plot creation as argument                                                                  |
+-----------------+--------------------------------------------+-------------------------------------------------------------------------------------------------------+
| view            | axes3d.view_init                           | Change the viewing angle for a 3d plot                                                                |
+-----------------+--------------------------------------------+-------------------------------------------------------------------------------------------------------+
| colormap        | plot.colormap                              | Set the colormap; better to do so at plot creation if possible                                        |
+-----------------+--------------------------------------------+-------------------------------------------------------------------------------------------------------+
| colorbar        | figure.add_colorbar(axes)                  | Add colorbar to selected axes                                                                         |
+-----------------+--------------------------------------------+-------------------------------------------------------------------------------------------------------+

API Reference
=============

.. currentmodule:: sidpy

.. automodule:: sidpy
    :no-members:
    :no-inherited-members:

:py:mod:`sidpy`:

.. autosummary::
    :toctree: _autosummary/
    :template: module.rst

    sidpy.base
    sidpy.hdf
    sidpy.io
    sidpy.proc
    sidpy.sid
    sidpy.viz
