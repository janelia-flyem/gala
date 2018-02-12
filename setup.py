import sys
import os
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
VERSION             = '0.4.2'
PYTHON_VERSION      = (3, 5)
INST_DEPENDENCIES   = ['cython']


if __name__ == '__main__':

    # Massive hack: installing on RTD fails, so we change installation to
    # building in-place
    on_rtd = (os.environ.get('READTHEDOCS', None) == 'True')
    if on_rtd:
        sys.argv[1] = 'build_ext'
        sys.argv.append('-i')

    setup(name=DISTNAME,
        version=VERSION,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        license=LICENSE,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Healthcare Industry',
            'Intended Audience :: Science/Research',
            'Programming Language :: Cython',
            'Programming Language :: Python :: 3.5',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Image Recognition',
            'Topic :: Scientific/Engineering :: Medical Science Apps.',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],
        packages=['gala', 'gala.features'],
        package_data={'gala':
                      ['testdata/*.*', 'testdata/original_grayscales/*']},
        install_requires=INST_DEPENDENCIES,
        scripts=["bin/gala-segment", "bin/gala-segmentation-stitch",
                 "bin/gala-segmentation-pipeline",
                 "bin/gala-train", "bin/gala-test-package",
                 "bin/gala-pixel", "bin/comparestacks",
                 "bin/gala-valprob", "bin/gala-auto", "bin/gala-serve"],
        ext_modules = cythonize(["gala/*.pyx","gala/features/*.pyx"],
                                annotate=True),
        include_dirs=[numpy.get_include()]
    )

