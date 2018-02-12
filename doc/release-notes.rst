=============
Release notes
=============

0.4
===

0.4.2
-----

This release updates setup.py to conform to a new requirement in setuptools
38.0 that package requirements listings should be ordered (list or tuple, but
not set).

0.4.1
-----

This release provides initial support for an interactive proofreading service.
`(#64) <https://github.com/janelia-flyem/gala/pull/68>`__.

Gala's performance, while still slow, is much improved:

- 30x speedup in RAG building
- 5x speedup in flat learning
- 6x speedup in agglomerative learning
- 7x speedup in test segmentation
- 30% reduction in RAM usage by the RAG class

Some of these improvements are thanks to Michal Janusz's (@mjanusz) idea of
batching calls to scikit-learn.

This release also includes many bug fixes (thanks to Tobias Maier!):

- Broken ``best_possible_segmentation`` when labels were not continuous (`(#71)
  <https://github.com/janelia-flyem/gala/issues/71>`__).
- Broken ``split_vi`` and ``set_ground_truth`` functions (`(#72)
  <https://github.com/janelia-flyem/gala/issues/72>`__ and `(#73)
  <https://github.com/janelia-flyem/gala/issues/71>`__).

Finally, we also made a `conda environment file
<https://conda.io/docs/user-guide/tasks/manage-environments.html>`__ to make it
easy to get started with gala and dependencies.

0.3
===

0.3.2
-----

- Bug fix: missing import in ``test_gala.py``. This was caused by rebasing
  commits from post-0.3 onto 0.3.


0.3.1
-----

This is a major bug fix release addressing
`issue #63 on GitHub <https://github.com/janelia-flyem/gala/issues/63>`__.
You can read more there and in the related
`mailing list thread <http://gala.30861.n7.nabble.com/issue-with-learn-agglomerate-td81.html>`__,
but the gist is that the "learning mode" parameter did nothing in previous
releases of gala. The gala library in fact was not implementing the algorithm
described in the GALA paper, but rather, a variant of
`LASH <http://papers.nips.cc/paper/4249-learning-to-agglomerate-superpixel-hierarchies>`__
with memory across epochs. (LASH only retains data from the most recent
learning epoch.) It remains to be determined whether
the "strict" learning mode described in our paper indeed yields
improvements in segmentation accuracy. 

Note that the included tests pass when using scikit-learn 0.16, but not with
the recently-released 0.17, because of changes in the implementation of
``GaussianNB``.


0.3.0
-----

Announcing the third release of gala!

I want to thank Paul Watkins, Sean Colby, Larissa Heinrich,
Joergen Kornfeld, and Jan Funke for their bug reports and mailing
list discussions, which prompted almost all of the improvements in
this release.

I must also thank the
`Saalfeld lab <https://www.janelia.org/lab/saalfeld-lab>`__ for financial
support while I was making these improvements.

This release focuses on performance improvements, but also includes some
API and behavior changes.

**This is the last release of gala supporting Python 2.** Upcoming work
will focus on asynchronous learning to enable interactive proofreading,
for which Python 3.4 and 3.5 offer compelling features and libraries. If
you absolutely *need* Python 2.7 support in gala, get in touch!

On to the changes in this version!


Major changes:
--------------

- 2x memory reduction and 3x RAG construction speedup.
- Add support for masked volumes: use a boolean array of the same shape
  as the image to inspect only ``True`` positions.
- **API break:** The label "0" is no longer considered a boundary label;
  volumes with a single-voxel-thick boundary are no longer supported.
- **API break:** The Ultrametric Contour Map (UCM) is gone, because it is
  inaccurate without a voxel-thick boundary, and was computationally
  expensive to maintain.

Minor changes:
--------------

- Add ``paper_em`` and ``snemi3d`` default feature managers (in
  ``gala.features.default``) to reproduce previous gala results.
- Bug fix: passing a label array of type floating point no longer
  causes a crash. (But you really should use integers for labels!)


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
