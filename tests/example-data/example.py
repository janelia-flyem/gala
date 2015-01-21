from __future__ import absolute_import
from __future__ import print_function
# imports
from gala import imio, classify, features, agglo, evaluate as ev
from six.moves import map

# read in training data
gt_train, pr_train, ws_train = (map(imio.read_h5_stack,
                                ['train-gt.lzf.h5', 'train-p1.lzf.h5',
                                 'train-ws.lzf.h5']))

# create a feature manager
fm = features.moments.Manager()
fh = features.histogram.Manager()
fc = features.base.Composite(children=[fm, fh])

# create graph and obtain a training dataset
g_train = agglo.Rag(ws_train, pr_train, feature_manager=fc)
(X, y, w, merges) = g_train.learn_agglomerate(gt_train, fc)[0]
y = y[:, 0] # gala has 3 truth labeling schemes, pick the first one
print((X.shape, y.shape)) # standard scikit-learn input format

# train a classifier, scikit-learn syntax
rf = classify.DefaultRandomForest().fit(X, y)
# a policy is the composition of a feature map and a classifier
learned_policy = agglo.classifier_probability(fc, rf)

# get the test data and make a RAG with the trained policy
pr_test, ws_test = (map(imio.read_h5_stack,
                        ['test-p1.lzf.h5', 'test-ws.lzf.h5']))
g_test = agglo.Rag(ws_test, pr_test, learned_policy, feature_manager=fc)
g_test.agglomerate(0.5) # best expected segmentation
seg_test1 = g_test.get_segmentation()

# the same approach works with a multi-channel probability map
p4_train = imio.read_h5_stack('train-p4.lzf.h5')
# note: the feature manager works transparently with multiple channels!
g_train4 = agglo.Rag(ws_train, p4_train, feature_manager=fc)
(X4, y4, w4, merges4) = g_train4.learn_agglomerate(gt_train, fc)[0]
y4 = y4[:, 0]
print((X4.shape, y4.shape))
rf4 = classify.DefaultRandomForest().fit(X4, y4)
learned_policy4 = agglo.classifier_probability(fc, rf4)
p4_test = imio.read_h5_stack('test-p4.lzf.h5')
g_test4 = agglo.Rag(ws_test, p4_test, learned_policy4, feature_manager=fc)
g_test4.agglomerate(0.5)
seg_test4 = g_test4.get_segmentation()

# gala allows implementation of other agglomerative algorithms, including
# the default, mean agglomeration
g_testm = agglo.Rag(ws_test, pr_test,
                    merge_priority_function=agglo.boundary_mean)
g_testm.agglomerate(0.5)
seg_testm = g_testm.get_segmentation()

# examine how well we did with either learning approach, or mean agglomeration
gt_test = imio.read_h5_stack('test-gt.lzf.h5')
import numpy as np
results = np.vstack((
    ev.split_vi(ws_test, gt_test),
    ev.split_vi(seg_testm, gt_test),
    ev.split_vi(seg_test1, gt_test),
    ev.split_vi(seg_test4, gt_test)
    ))

print(results)

# generate split-vi plots for each method
from matplotlib import pyplot as plt
g_test.agglomerate(np.inf)
g_test4.agglomerate(np.inf)
g_testm.agglomerate(np.inf)
ucms = [g.get_ucm() for g in [g_testm, g_test, g_test4]]
vis = [ev.vi_by_threshold(u, gt_test, [0], [0])[1:] for u in ucms]
colors = ['black', 'deepskyblue', 'orange']
plt.figure(figsize=(3.3, 3.3))
from gala import viz
viz.plot_split_vi(vis, colors=colors)
plt.xlim(0, 1); plt.ylim(0, 1)
plt.legend(['mean agglomeration', 'gala 1-channel', 'gala 4-channel'],
           loc=1, fontsize=8)
plt.xticks(np.arange(0, 1.01, 0.2), fontsize=10)
plt.yticks(np.arange(0, 1.01, 0.2), fontsize=10)
plt.xlabel('false merges (bits)', fontsize=10)
plt.ylabel('false splits (bits)', fontsize=10)
plt.savefig('split-vi.png', dpi=300, bbox_inches='tight')

