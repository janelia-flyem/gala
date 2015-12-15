import numpy as np
from sklearn.utils import check_random_state
import zmq
from . import agglo, features, classify, evaluate as ev


_feature_manager = features.default.snemi3d()


# constants
# labels for machine learning libs
MERGE_LABEL = 0
SEPAR_LABEL = 1


def root(tree, n):  # speed this up by adding a function to viridis
    anc = tree.ancestors(n)
    if anc == []:
        return n
    else:
        return anc[-1]


class Solver(object):
    '''
    ZMQ-based interface between proofreading clients and gala RAGs.

    Parameters
    ----------

    Attributes
    ----------
    '''
    def __init__(self, labels, image, port=5556, host='tcp://*',
                 relearn_threshold=20):
        self.labels = labels
        self.image = image
        self.policy = agglo.boundary_mean
        self.build_rag()
        self.comm = zmq.Context().socket(zmq.PAIR)
        self.addr = host + ':' + str(port)
        self.comm.bind(self.addr)
        self.history = []
        self.separate = []
        self.features = []
        self.targets = []
        self.relearn_threshold = relearn_threshold
        self.relearn_trigger = relearn_threshold

    def build_rag(self):
        self.rag = agglo.Rag(self.labels, self.image,
                             merge_priority_function=self.policy,
                             feature_manager=_feature_manager,
                             normalize_probabilities=True)
        self.original_rag = self.rag.copy()

    def send_segmentation(self):
        self.relearn()  # correct way to do it is to implement RAG splits
        self.rag.agglomerate(0.5)
        dst = [int(i) for i in self.rag.tree.get_map(0.5)]
        src = list(range(len(dst)))
        message = {'type': 'fragment-segment-lut',
                   'data': {'fragments': src, 'segments': dst}}
        self.comm.send_json(message)

    def listen(self):
        while True:
            message = self.comm.recv_json()
            command = message['type']
            data = message['data']
            if command == 'merge':
                segments = data['segments']
                self.learn_merge(segments)
            elif command == 'separate':
                segments = data['segments']
                self.learn_separation(segments)
            elif command == 'request':
                what = data['what']
                if what == 'fragment-segment-lut':
                    self.send_segmentation()
            elif command == 'stop':
                return
            else:
                print('command %s not recognized.' % command)
                return

    def learn_merge(self, segments):
        segments = iter(set(root(self.rag.tree, s) for s in  segments))
        s0 = next(segments)
        for s1 in segments:
            self.features.append(_feature_manager(self.rag, s0, s1))
            self.history.append((s0, s1))
            s0 = self.rag.merge_nodes(s0, s1)
            self.targets.append(MERGE_LABEL)

    def learn_separation(self, fragments):
        f0, f1 = fragments
        s0, s1 = self.rag.separate_fragments(f0, f1)
        # trace the segments up to the current state of the RAG
        # don't use the segments directly
        self.features.append(_feature_manager(self.rag, s0, s1))
        self.targets.append(SEPAR_LABEL)
        self.separate.append((f0, f1))

    def relearn(self):
        clf = classify.DefaultRandomForest().fit(self.features, self.targets)
        self.policy = agglo.classifier_probability(_feature_manager, clf)
        self.rag = self.original_rag.copy()
        self.rag.merge_priority_function = self.policy
        self.rag.rebuild_merge_queue()
        for i, (s0, s1) in enumerate(self.separate):
            self.rag.node[s0]['exclusions'].add(i)
            self.rag.node[s1]['exclusions'].add(i)
        self.rag.replay_merge_history(self.history)


def proofread(fragments, true_segmentation, host='tcp://localhost', port=5556,
              num_operations=10, mode='fast paint', random_state=None):
    """Simulate a proofreader by sending and receiving messages to a Solver.

    Parameters
    ----------
    fragments : array of int
        The initial segmentation to be proofread.
    true_segmentation : array of int
        The target segmentation. Should be a superset of `fragments`.
    host : string
        The host to serve ZMQ commands to.
    port : int
        Port on which to connect ZMQ.
    num_operations : int, optional
        How many proofreading operations to perform before returning.
    mode : string, optional
        The mode with which to simulate proofreading.
    random_state : None or int or numpy.RandomState instance, optional
        Fix the random state for proofreading.

    Returns
    -------
    lut : tuple of array-like of int
        A look-up table from fragments (first array) to segments
        (second array), obtained by requesting it from the Solver after
        initial proofreading simulation.
    """
    true = agglo.best_possible_segmentation(fragments, true_segmentation)
    base_graph = agglo.Rag(fragments)
    comm = zmq.Context().socket(zmq.PAIR)
    comm.connect(host + ':' + str(port))
    ctable = ev.contingency_table(fragments, true).tocsc()
    true_labels = np.unique(true)
    random = check_random_state(random_state)
    random.shuffle(true_labels)
    for _, label in zip(range(num_operations), true_labels):
        components = [int(i) for i in ctable.getcol(int(label)).indices]
        comm.send_json({'type': 'merge',
                        'data': {'segments': components}})
        for fragment in components:
            for neighbor in base_graph[fragment]:
                if neighbor not in components:
                    edge = [int(fragment), int(neighbor)]
                    comm.send_json({'type': 'separate',
                                    'data': {'segments': edge}})

    comm.send_json({'type': 'request',
                    'data': {'what': 'fragment-segment-lut'}})
    response = comm.recv_json()
    src = response['data']['fragments']
    dst = response['data']['segments']
    return src, dst
