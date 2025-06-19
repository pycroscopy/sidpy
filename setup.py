from codecs import open
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst')) as f:
    long_description = f.read()

with open(os.path.join(here, 'sidpy/__version__.py')) as f:
    __version__ = f.read().split("'")[1]

# TODO: Move requirements to requirements.txt
requirements = ['numpy>=1.10',
                'toolz',  # dask installation failing without this
                'cytoolz',  # dask installation failing without this
                'dask',
                'h5py>=2.6.0',
                'matplotlib>=2.0.0',
                'six',
                'joblib>=0.11.0',
                'ipywidgets',
                'ipython',  # Beginning with IPython 6.0, Python 3.3 and above is required.
                'scikit-learn',
                'ase',
                'ipympl',
                'dill'
                ]

setup(
    name='sidpy',
    version=__version__,
    description='Python utilities for storing, visualizing, and processing Spectroscopic and Imaging Data (SID)',
    long_description=long_description,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',

        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Information Analysis'],
    keywords=['imaging', 'spectra', 'multidimensional', 'scientific',
              'visualization', 'processing', 'storage', 'hdf5'],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    url='https://pycroscopy.github.io/sidpy/about.html',
    license='MIT',
    author='Pycroscopy contributors',
    author_email='pycroscopy@gmail.com',
    install_requires=requirements,
    
    tests_require=['pytest'],
    platforms=['Linux', 'Mac OSX', 'Windows 11/10'],
    # package_data={'sample':['dataset_1.dat']}
    test_suite='pytest',
    # dependency='',
    # dependency_links=[''],
    include_package_data=True,
    # https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-dependencies
    extras_require={
        'MPI': ["mpi4py"]
    },
    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # package_data={
    #     'sample': ['package_data.dat'],
    # },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
)
