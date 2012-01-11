# ray: segmentation of nD images

Ray is a python library for performance and evaluation of image segmentation.  
It supports n-dimensional images (images, volumes, videos, videos of 
volumes...) and multiple channels per image.

## Requirements (tested versions)

* Python 2.x (2.6, 2.7)
* numpy (1.5.1, 1.6.0)
* Image (a.k.a. Python Imaging Library or PIL) (1.3.1)
* networkx (1.4, 1.5)
* h5py (1.5.0)
* scipy (0.7.0, 0.9.0)

All of the above are included in the Enthought Python Distribution, so I would
recommend you just install that if you can.

### Recommended

* progressbar 2.3-dev
* [vigra/vigranumpy](hci.iwr.uni-heidelberg.de/vigra/) (1.7.1)
* scikits.learn (0.7.1, 0.8)

progressbar is in PyPi and trivial to install:

```
sudo easy_install progressbar
```

For vigra, you are on your own. It is used for the random forest classifier,
but if you don't install it you can still use SVM or AdaBoost classifiers.

## Installation

Well, there's nothing to install per se (distutils support coming at some point
in the far future). Download the source and add whatever path you downloaded it
to to your Python path.

## Usage

### Mean agglomeration

Suppose you have already trained a pixel level boundary detector, and want to
perform mean agglomeration on it. This is the simplest form of agglomeration
and was the initial design spec for Ray. Now:

```
from ray import imio, agglo, morpho
prob = imio.read_image_stack('probabilities-*.png') 
# prob is a numpy ndarray
# probabilities-* can be one file for 2D segmentation, or many files for 3D.
label_field = morpho.watershed(prob)
# Make the region adjacency graph (RAG)
g = agglo.Rag(label_field, prob)
threshold = 0.5
g.agglomerate(threshold) # agglomerate until the boundary mean is 0.5
seg = g.get_segmentation() # the label field resulting from the agglomeration
g.agglomerate(inf) # agglomerate to completion
ucm = g.get_ucm() # get the ultrametric contour map (UCM) for the segmentation
```

A UCM can be thresholded to provide the segmentation at any threshold of 
agglomeration. It may, however, result in a split when a segment becomes
thinner than one pixel.
