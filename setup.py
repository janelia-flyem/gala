#from distutils.core import setup
from setuptools import setup, find_packages



setup(name = "gala",
    version = "1.0",
    url = "https://github.com/jni/gala",
    description = "Gala is a python library for performing and evaluating image segmentation.",
    long_description = "Gala is a python library for performing and evaluating of image segmentation. It supports n-dimensional images (images, volumes, videos, videos of volumes...) and multiple channels per image.",
    author = "Juan Nunez-Iglesias",
    author_email = 'jni@janelia.hhmi.org',
    license = 'LICENSE.txt',
    packages = ['gala', 'gala.features'],
    package_data = {'gala': ['testdata/*.*', 'testdata/original_grayscales/*'] },
    install_requires = [ ],
    scripts = ["bin/gala-segmentation-stitch", "bin/gala-segmentation-pipeline", "bin/gala-train", "bin/gala-test-package", "bin/gala-pixel", "bin/comparestacks", "bin/gala-valprob", "bin/gala-auto"]
)
