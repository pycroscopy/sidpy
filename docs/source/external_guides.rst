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

HDF5 and h5py
~~~~~~~~~~~~~
Our software packages - ``sidpy``, ``pyUSID``, ``pyNSID`` are all
designed to be file-centric, we highly recommend learning more about HDF5 and h5py:

* `Basics of HDF5 <https://portal.hdfgroup.org/display/HDF5/Learning+HDF5>`_ (especially the last three tutorials)
* `Quick start <http://docs.h5py.org/en/latest/quick.html>`_ to h5py
* Another `tutorial on HDF5 and h5py <https://www.nersc.gov/assets/Uploads/H5py-2017-Feb23.pdf>`_

Installing software
-------------------
python
~~~~~~~
`uv <https://docs.astral.sh/uv/>`_ is a fast Python package and environment manager that can create isolated environments and install scientific dependencies reproducibly.
Tutorial for `installing uv <https://docs.astral.sh/uv/getting-started/installation/>`_.

python packages
~~~~~~~~~~~~~~~~
Two popular methods for installing packages in python are:

* `pip <https://packaging.python.org/tutorials/installing-packages/>`_:
    * included with basic python and standard on Linux and Mac OS
    * Works great for installing pure python and other simple packages
* `uv <https://docs.astral.sh/uv/>`_
    * creates and manages project environments
    * can install packages from `pyproject.toml` or requirements files

Updating packages
~~~~~~~~~~~~~~~~~
If you are working in a `uv` managed project, open a terminal and run:

.. code:: bash

    uv lock --upgrade
    uv sync

Upgrading python
~~~~~~~~~~~~~~~~
Use `uv python install <https://docs.astral.sh/uv/reference/cli/#uv-python-install>`_ to add a specific Python version, then recreate the environment with that interpreter.

Writing code
------------
Text Editors
~~~~~~~~~~~~
These software often do not have any advanced features found in IDEs such as syntax highlighting,
real-time code-checking etc. but are simple, and most importantly, open files quickly.  Here are some excellent
text editors for each class of operating system:

* Mac OS - `Atom <https://atom.io/>`_
* Linux - `gEdit <https://wiki.gnome.org/Apps/Gedit>`_, `vim <https://www.vim.org/>`_, `neovim <https://neovim.io/>`_
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

* `VS Code <https://code.visualstudio.com/>`_
    * Completely free and open-source editor by Microsoft. Much faster and extremely lightweight compared to Pycharm.

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
in python 2 and 3 to ensure that a certain desired code works in both python 2 and 3. For Python users, common
approaches are **virtual environments** and tools like ``uv`` that manage them.

* `Virtual environment <https://docs.python.org/3/tutorial/venv.html>`_
    * Basic python ships with virtual enviroments.
    * How to `use venv <http://www.pythonforbeginners.com/basics/how-to-use-python-virtualenv>`_

* `uv`-managed environments
    * `uv documentation <https://docs.astral.sh/uv/>`_
    * use ``uv venv`` and ``uv sync`` to create and populate a project environment

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
