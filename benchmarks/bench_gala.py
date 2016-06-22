import os

from gala import imio, features, agglo, classify


rundir = os.path.dirname(__file__)
dd = os.path.abspath(os.path.join(rundir, '../tests/example-data'))


em3d = features.default.paper_em()


def setup_trdata():
    wstr = imio.read_h5_stack(os.path.join(dd, 'train-ws.lzf.h5'))
    prtr = imio.read_h5_stack(os.path.join(dd, 'train-p1.lzf.h5'))
    gttr = imio.read_h5_stack(os.path.join(dd, 'train-gt.lzf.h5'))
    return wstr, prtr, gttr


def setup_tsdata():
    wsts = imio.read_h5_stack(os.path.join(dd, 'test-ws.lzf.h5'))
    prts = imio.read_h5_stack(os.path.join(dd, 'test-p1.lzf.h5'))
    gtts = imio.read_h5_stack(os.path.join(dd, 'test-gt.lzf.h5'))
    return wsts, prts, gtts


def setup_trgraph():
    ws, pr, ts = setup_trdata()
    g = agglo.Rag(ws, pr, feature_manager=em3d)
    return g


def setup_tsgraph():
    ws, pr, ts = setup_tsdata()
    g = agglo.Rag(ws, pr, feature_manager=em3d)
    return g


def setup_trexamples():
    gt = imio.read_h5_stack(os.path.join(dd, 'train-gt.lzf.h5'))
    g = setup_trgraph()
    (X, y, w, e), _ = g.learn_agglomerate(gt, em3d, min_num_epochs=5)
    y = y[:, 0]
    return X, y


def setup_classifier():
    X, y = setup_trexamples()
    rf = classify.DefaultRandomForest()
    rf.fit(X, y)
    return rf


def setup_policy():
    rf = classify.DefaultRandomForest()
    cl = agglo.classifier_probability(em3d, rf)
    return cl


def setup_tsgraph_queue():
    g = setup_tsgraph()
    cl = setup_policy()
    g.merge_priority_function = cl
    g.rebuild_merge_queue()
    return g


