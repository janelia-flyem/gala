#from distutils.core import setup
from setuptools import setup, find_packages



setup(name = "ray",
    version = "1.0",
    url = "https://github.com/jni/ray",
    description = "Ray is a python library for performing and evaluating image segmentation.",
    long_description = "Ray is a python library for performing and evaluating of image segmentation. It supports n-dimensional images (images, volumes, videos, videos of volumes...) and multiple channels per image.",
    author = "Juan Nunez-Iglesias",
    author_email = 'jni@janelia.hhmi.org',
    license = 'LICENSE.txt',
    packages = ['ray', 'ray.features'],
    package_data = {'ray': ['testdata/*.*', 'testdata/original_grayscales/*'] },
    install_requires = [ ],
    scripts = ["bin/ray-segmentation-pipeline", "bin/ray-train", "bin/ray-test-package", "bin/ray-pixel", "bin/comparestacks", "bin/ray-valprob"]
)
