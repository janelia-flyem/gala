#from distutils.core import setup
from setuptools import setup, find_packages



setup(name = "ray",
    version = "1.0",
    description = "Ray is a python library for performance and evaluation of image segmentation, distributed under the open-source MIT license. It supports n-dimensional images (images, volumes, videos, videos of volumes...) and multiple channels per image.",
    author = "Juan Nunez-Iglesias",
    url = "https://github.com/jni/ray",
    author_email = 'jni@janelia.hhmi.org',
    packages = ['ray', 'ray.features'],
    install_requires = ['scikit-learn', 'progressbar', 'scikits-image>=0.5', 'matplotlib', 'h5py>=1.5.0', 'networkx>=1.6', 'scipy>=0.10.0',  'numpy>=1.6.0', 'f2py'
              ],
    scripts = ["bin/ray-segmentation-pipeline", "bin/ray-train"]
)
