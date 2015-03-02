.. gala documentation master file, created by
   sphinx-quickstart on Mon Mar  2 18:55:08 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

gala: segmentation of nD images
===============================

Gala is a python library for performing and evaluating image
segmentation, distributed under the open-source, BSD-like `Janelia Farm
license <http://janelia-flyem.github.com/janelia_farm_license.html>`__.
It implements the algorithm described in `Nunez-Iglesias et
al. <http://arxiv.org/abs/1303.6163>`__, PLOS ONE, 2013.

If you use this library in your research, please cite:

    Nunez-Iglesias J, Kennedy R, Plaza SM, Chakraborty A and Katz WT
    (2014) `Graph-based active learning of agglomeration (GALA): a
    Python library to segment 2D and 3D
    neuroimages. <http://journal.frontiersin.org/Journal/10.3389/fninf.2014.00034/abstract>`__
    *Front. Neuroinform. 8:34.* doi:10.3389/fninf.2014.00034

If you use or compare to the GALA algorithm in your research, please
cite:

    Nunez-Iglesias J, Kennedy R, Parag T, Shi J, Chklovskii DB (2013)
    `Machine Learning of Hierarchical Clustering to Segment 2D and 3D
    Images. <http://journal.frontiersin.org/Journal/10.3389/fninf.2014.00034/abstract>`__
    *PLoS ONE 8(8): e71715.* doi:10.1371/journal.pone.0071715

Gala supports n-dimensional images (images, volumes, videos, videos of
volumes...) and multiple channels per image.

|Build Status| |Coverage Status|

Contents:

.. toctree::
   :maxdepth: 2

   installation
   gettingstarted


.. |Build Status| image:: https://travis-ci.org/janelia-flyem/gala.png?branch=master
   :target: https://travis-ci.org/janelia-flyem/gala
.. |Coverage Status| image:: https://img.shields.io/coveralls/janelia-flyem/gala.svg
   :target: https://coveralls.io/r/janelia-flyem/gala



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

