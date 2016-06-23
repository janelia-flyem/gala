import os

from contextlib import contextmanager
from gala import imio, features, agglo, classify
from asv.extern.asizeof import asizeof

rundir = os.path.dirname(__file__)
## dd: the data directory
dd = os.path.abspath(os.path.join(rundir, '../tests/example-data'))


from time import process_time


@contextmanager
def timer():
    time = []
    t0 = process_time()
    yield time
    t1 = process_time()
    time.append(t1 - t0)


em = features.default.paper_em()


def trdata():
    wstr = imio.read_h5_stack(os.path.join(dd, 'train-ws.lzf.h5'))
    prtr = imio.read_h5_stack(os.path.join(dd, 'train-p1.lzf.h5'))
    gttr = imio.read_h5_stack(os.path.join(dd, 'train-gt.lzf.h5'))
    return wstr, prtr, gttr


def tsdata():
    wsts = imio.read_h5_stack(os.path.join(dd, 'test-ws.lzf.h5'))
    prts = imio.read_h5_stack(os.path.join(dd, 'test-p1.lzf.h5'))
    gtts = imio.read_h5_stack(os.path.join(dd, 'test-gt.lzf.h5'))
    return wsts, prts, gtts


def trgraph():
    ws, pr, ts = trdata()
    g = agglo.Rag(ws, pr)
    return g


def tsgraph():
    ws, pr, ts = tsdata()
    g = agglo.Rag(ws, pr, feature_manager=em)
    return g


def trexamples():
    gt = imio.read_h5_stack(os.path.join(dd, 'train-gt.lzf.h5'))
    g = trgraph()
    (X, y, w, e), _ = g.learn_agglomerate(gt, em, min_num_epochs=5)
    y = y[:, 0]
    return X, y


def classifier():
    X, y = trexamples()
    rf = classify.DefaultRandomForest()
    rf.fit(X, y)
    return rf


def policy():
    rf = classify.DefaultRandomForest()
    cl = agglo.classifier_probability(em, rf)
    return cl


def tsgraph_queue():
    g = tsgraph()
    cl = policy()
    g.merge_priority_function = cl
    g.rebuild_merge_queue()
    return g

def bench_suite():
    times = {}
    memory = {}
    wstr, prtr, gttr = trdata()
    with timer() as t_build_rag:
        g = agglo.Rag(wstr, prtr)
    times['build RAG'] = t_build_rag[0]
    memory['base RAG'] = asizeof(g)
    with timer() as t_features:
        g.set_feature_manager(em)
    times['build feature caches'] = t_features[0]
    memory['feature caches'] = asizeof(g) - memory['base RAG']
    with timer() as t_flat:
        g.learn_flat(gttr, em)
    times['learn flat'] = t_flat[0]
    return times, memory


