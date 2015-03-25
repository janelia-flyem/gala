Installation
============

Requirements
------------

-  Python 3.4 or 2.7
-  numpy 1.7+
-  scipy 0.10+
-  Image (a.k.a. Python Imaging Library or PIL) 1.1.7 or Pillow 2.5+
-  networkx 1.6+
-  HDF5 and h5py 1.5+
-  cython 0.17+
-  scikit-learn 0.15+
-  matplotlib 1.2+
-  scikit-image 0.11+
-  viridis 0.3+

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

-  progressbar 2.3 (Python 2.7 only, currently)
-  `vigra/vigranumpy <hci.iwr.uni-heidelberg.de/vigra/>`__ (1.9.0)
-  libtiff 0.4+

In its original incarnation, this project used Vigra for the random
forest classifier. Installation is less simple than scikit-learn, which
has emerged in the last few years as a truly excellent implementation and is
now recommended. Tests in the test suite expect scikit-learn rather than
Vigra. You can also use any of the scikit-learn classifiers, including
their newly-excellent random forest.

Installing with distutils
~~~~~~~~~~~~~~~~~~~~~~~~~

Gala is a Python library with limited Cython extensions and can be
installed in two ways:

- Use the command ``python setup.py build_ext -i`` in the gala directory,
  then add the gala directory to your PYTHONPATH environment variable, or
- Install it into your preferred python environment with
  ``python setup.py install``.

Installing requirements
~~~~~~~~~~~~~~~~~~~~~~~

Though you can install all the requirements yourself, as most are
available in the Python Package Index (PyPI) and can be installed with
simple commands, the easiest way to get up and running is to use the
`Continuum Anaconda <http://www.continuum.io/downloads>`__ Python
distribution. An example of installing all dependencies for gala on
linux is shown in the ``.travis.yml`` test recipe.

Installing with Buildem
~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can use Janelia's own `buildem
system <http://github.com/janelia-flyem/buildem#readme>`__ to
automatically download, compile, test, and install requirements into a
specified buildem prefix directory. (You will need CMake.)

::

    $ cmake -D BUILDEM_DIR=/path/to/platform-specific/build/dir <gala directory>
    $ make

You might have to run the above steps twice if this is the first time
you are using the buildem system.

On Mac, you might have to install compilers (such as gcc, g++, and
gfortran).

Testing
~~~~~~~

The test coverage is rather tiny, but it is still a nice way to check
you haven't completely screwed up your installation. The tests do cover
the fundamental functionality of agglomeration learning.

We use `pytest <https://pytest.org>`__ for testing. Run the tests by building
gala in-place and running the ``py.test`` command. (You need to have installed
pytest and pytest-cov for this to work. Both are readily available in PyPI.)

Alternatively, you can run individual test files independently:

.. code:: bash

    $ cd tests
    $ python test_agglo.py
    $ python test_features.py
    $ python test_watershed.py
    $ python test_optimized.py
    $ python test_gala.py
