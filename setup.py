from __future__ import absolute_import
#from distutils.core import setup
from setuptools import setup
from Cython.Build import cythonize
import numpy

descr = """Graph-based active learning of agglomeration

Gala implements a suite of hierarchical segmentation algorithms,
including the eponymous Graph-based active learning of agglomeration.

In addition, gala contains implementations of many common segmentation
evaluation functions, such as the variation of information (VI) and the
adjusted Rand index (ARI).
"""

DISTNAME            = 'gala'
DESCRIPTION         = 'Hierarchical nD image segmentation algorithms'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Juan Nunez-Iglesias'
MAINTAINER_EMAIL    = 'juan.n@unimelb.edu.au'
URL                 = 'https://gala.readthedocs.org'
LICENSE             = 'Janelia (BSD-like)'
DOWNLOAD_URL        = 'https://github.com/janelia-flyem/gala'
VERSION             = '0.2dev'
PYTHON_VERSION      = (2, 7)
INST_DEPENDENCIES   = {} 


if __name__ == '__main__':

    setup(name=DISTNAME,
        version=VERSION,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        license=LICENSE,
        packages=['gala', 'gala.features'],
        package_data={'gala':
                      ['testdata/*.*', 'testdata/original_grayscales/*']},
        install_requires=INST_DEPENDENCIES,
        scripts=["bin/gala-segmentation-stitch",
                 "bin/gala-segmentation-pipeline",
                 "bin/gala-train", "bin/gala-test-package",
                 "bin/gala-pixel", "bin/comparestacks",
                 "bin/gala-valprob", "bin/gala-auto"],
        ext_modules = cythonize(["gala/*.pyx","gala/features/*.pyx"],
                                annotate=True),
        include_dirs=[numpy.get_include()]
    )

