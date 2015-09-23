=============
Release notes
=============

0.2
===

0.2.3
-----

Minor feature addition: enable exporting segmentation results *after*
agglomeration is complete.

0.2.2
-----

This maintenance release contains several bug fixes:

- package Cython source files (.pyx) for PyPI
- package the gala-segment command-line interface for PyPI
- include viridis in ``requirements.txt``
- update libtiff usage

0.2
---

This release owes much of its existence to Neal Donnelly (@NealJMD on GitHub),
who bravely delved into gala and reduced its memory and time footprints by
over 20% each. The other highlights are Python 3 support and much better
continuous integration.

Major changes:
--------------

- gala now uses an ultrametric tree backend to represent the merge hierarchy.
  This speeds up merges and will allow more sophisticated editing operations
  in the future.
- gala is now **fully compatible with Python 3.4**! That's a big tick in the
  "being a good citizen of the Python community" box. =) The downside is that a
  lot of the operations are slower in Py3.
- As mentioned above, gala is 20% faster and 20% smaller than before. That's
  thanks to extensive benchmarking and Cythonizing by @NealJMD
- We are now measuring code coverage, and although it's a bit low at 40%, the
  major gala functions (RAG building, learning, agglomerating) are covered.
  And we're only going up from here!
- We now have `documentation on ReadTheDocs <http://gala.readthedocs.org>`__!

Minor changes:
--------------

- @anirbanchakraborty added the concepts of "frozen nodes" and "frozen edges",
  which are never merged. This is useful to
  temporarily ignore mitochondria during the first stages of agglomeration,
  which can dramatically reduce errors. (See
  `A Context-aware Delayed Agglomeration Framework for EM Segmentation <http://arxiv.org/abs/1406.1476>`__.)
- @anirbanchakraborty added the inclusiveness feature, a measure of how much
  a region is "surrounded" by another.
- The `gala.evaluate` module now supports the Adapted Rand Error, as used by
  the `SNEMI3D challenge <http://brainiac2.mit.edu/SNEMI3D>`__.
- Improvements to the `gala.morphology` module.
