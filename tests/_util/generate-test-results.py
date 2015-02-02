
import numpy as np
from gala import imio, classify, features, agglo, evaluate as ev
gt_train, pr_train, p4_train, ws_train = map(imio.read_h5_stack, ['example-data/train-gt.lzf.h5', 'example-data/train-p1.lzf.h5', 'example-data/train-p4.lzf.h5', 'example-data/train-ws.lzf.h5'])
gt_test, pr_test, p4_test, ws_test = map(imio.read_h5_stack, ['example-data/test-gt.lzf.h5', 'example-data/test-p1.lzf.h5', 'example-data/test-p4.lzf.h5', 'example-data/test-ws.lzf.h5'])
fm = features.moments.Manager()
fh = features.histogram.Manager()
fc = features.base.Composite(children=[fm, fh])
g_train = agglo.Rag(ws_train, pr_train, feature_manager=fc)
np.random.RandomState(0)
(X, y, w, merges) = map(np.copy, map(np.ascontiguousarray,
                        g_train.learn_agglomerate(gt_train, fc)[0]))
print X.shape
np.savez('example-data/train-set.npz', X=X, y=y)
y = y[:, 0]
rf = classify.DefaultRandomForest()
X.shape
np.random.RandomState(0)
rf = rf.fit(X, y)
classify.save_classifier(rf, 'example-data/rf-1.joblib')
learned_policy = agglo.classifier_probability(fc, rf)
g_test = agglo.Rag(ws_test, pr_test, learned_policy, feature_manager=fc)
g_test.agglomerate(0.5)
seg_test1 = g_test.get_segmentation()
imio.write_h5_stack(seg_test1, 'example-data/test-seg1.lzf.h5', compression='lzf')
g_train4 = agglo.Rag(ws_train, p4_train, feature_manager=fc)
np.random.RandomState(0)
(X4, y4, w4, merges4) = map(np.copy, map(np.ascontiguousarray,
                            g_train4.learn_agglomerate(gt_train, fc)[0]))
print X4.shape
np.savez('example-data/train-set4.npz', X=X4, y=y4)
y4 = y4[:, 0]
rf4 = classify.DefaultRandomForest()
np.random.RandomState(0)
rf4 = rf4.fit(X4, y4)
classify.save_classifier(rf4, 'example-data/rf-4.joblib')
learned_policy4 = agglo.classifier_probability(fc, rf4)
g_test4 = agglo.Rag(ws_test, p4_test, learned_policy4, feature_manager=fc)
g_test4.agglomerate(0.5)
seg_test4 = g_test4.get_segmentation()
imio.write_h5_stack(seg_test4, 'example-data/test-seg4.lzf.h5', compression='lzf')

results = np.vstack((
    ev.split_vi(ws_test, gt_test),
    ev.split_vi(seg_test1, gt_test),
    ev.split_vi(seg_test4, gt_test)
    ))

np.save('example-data/vi-results.npy', results)

