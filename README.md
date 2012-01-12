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

### Testing

The test coverage is rather tiny, but it is still a nice way to check you
haven't completely screwed up your installation. From the Ray root directory,
run `python test/test_ray.py` to run some regression tests.

## Usage

### Agglomeration

Suppose you have already trained a pixel level boundary detector, and want to
perform mean agglomeration on it. This is the simplest form of agglomeration
and was the initial design spec for Ray. Now:

```python
from ray import imio, agglo, morpho
# prob is a numpy ndarray
# probabilities-* can be one file for 2D segmentation, or many files for 3D.
prob = imio.read_image_stack('probabilities-*.png') 
label_field = morpho.watershed(prob)
# Make the region adjacency graph (RAG)
g = agglo.Rag(label_field, prob)
threshold = 0.5
# agglomerate until the given threshold
g.agglomerate(threshold)
# get the label field resulting from the agglomeration
seg = g.get_segmentation() 
# now agglomerate to completion and get the UCM
g.agglomerate(inf)
ucm = g.get_ucm()
```

An ultrametric contour map (UCM) can be thresholded to provide the segmentation
at any threshold of agglomeration. (It may, however, result in a split when a
segment becomes thinner than one pixel.)

The mean agglomeration may be too simple. What if we want to use the median?
We can specify this with the `merge_priority_function` argument to the RAG
constructor:

```python
# merge by boundary median instead of mean
g = agglo.Rag(label_field, prob, merge_priority_function=agglo.boundary_median)
```

A user can specify their own merge priority function. A valid merge priority
function is a callable Python object that takes a graph and two nodes from that
graph as input, and returns a real number. (Technically, any object that
satisfies the basic comparison operations, such as `__lt__`, will work.)

### Learning agglomeration

A whole new set of tools is needed to apply machine learning to agglomeration.
These are provided by the `classify` module, and built into the `agglo.Rag`
class.

```python
from ray import classify
gs = imio.read_h5_stack('gold-standard-segmentation.h5')
fm = classify.MomentsFeatureManager()
fh = classify.HistogramFeatureManager()
fc = classify.CompositeFeatureManager(children=[fm, fh])
```

A _feature manager_ is a callable object that computes feature vectors from
graph edges. The object has the following responsibilities, which it can inherit
from `classify.NullFeatureManager`:
- create a (possibly empty) _feature cache_ on each edge and node, precomputing
  some of the calculations needed for feature computation;
- maintain the feature cache throughout node merges during agglomeration;
- compute the feature vector from the feature caches when called with the
  inputs of a graph and two nodes.

Feature managers can be chained through the `classify.CompositeFeatureManager`
class.

We can then extract feature vectors from the graph as follows:

```python
g = agglo.Rag(label_field, prob, feature_manager=fc)
n1, n2 = 1, 2
feature_vector = fc(g, n1, n2)
```

This gives us the rudimentary tools to do some machine learning, when combined
with a labeling system. Given a gold standard segmentation (`gs`, above)
assumed to be a correct segmentation of the image, do:

```python
training_data, all_training_data = g.learn_agglomerate(gs, fc)
```

The training data is a tuple with four elements:
- an nsamples x nfeatures numpy array with the feature vectors for each
  learned edge.
- an nsamples x 4 numpy array with the associated lables for each edge: -1 for
  "correct merge", and +1 for "incorrect merge". The four columns are four
  different labeling systems. They mostly agree (and certainly do in the case
  of a perfect oversegmentation); using column 0 is fine for most purposes.
- an nsamples x 2 numpy array of weights (VI and RI) associated with each 
  learned edge, for weighted learning.
- an nsamples x 2 numpy array of edge ids, the sample history during learning.

`all_training_data` is a list of such tuples, one for each training epoch.
This will be ignored for this tutorial. Briefly, learning takes place by
agglomerating while comparing the present segmentation to the gold standard.
Once the volume has been fully agglomerated, learning starts over, which can
result in repeated elements. `training_data` is the set of unique learning
samples encountered over all epochs, while `all_training_data` is a list of
the samples encountered in each epoch (including repeats).


