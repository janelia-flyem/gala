from . import agglo, features, classify
import zmq
import json


_feature_manager = features.default.snemi3d()


# constants
# labels for machine learning libs
MERGE_LABEL = 0
SEPAR_LABEL = 1

def obj2jsonbytes(obj):
    '''Convert object to JSON representation using ASCII bytes, not Unicode.

    Parameters
    ----------
    obj : any object
        The input object. Must be JSON-serializable.

    Returns
    -------
    bytes_msg : bytes
        JSON encoding of ``obj`` in a Python bytes object (not a string).
    '''
    unicode_msg = json.dumps(obj)
    bytes_msg = str.encode(unicode_msg, encoding='ascii')
    return bytes_msg


def jsonbytes2obj(bytes_msg):
    '''Decode a bytes-encoded message to string before loading from JSON.


    Parameters
    ----------
    bytes_msg : bytes
        A bytes-encoded message containing a JSON object representation.

    Returns
    -------
    obj : object (typically dict)
        The object represented by the JSON encoding.
    '''
    unicode_msg = bytes.decode(bytes_msg)
    obj = json.loads(unicode_msg)


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
    def __init__(self, labels, image, port=5556, host='tcp://localhost',
                 relearn_threshold=20):
        self.labels = labels
        self.image = image
        self.policy = agglo.boundary_mean
        self.build_rag()
        self.comm = zmq.Context().socket(zmq.PAIR)
        self.comm.connect(host + ':' + str(port))
        self.features = []
        self.targets = []
        self.relearn_threshold = relearn_threshold
        self.relearn_trigger = relearn_threshold

    def build_rag(self):
        self.rag = agglo.Rag(self.labels, self.image,
                             merge_priority_function=self.policy,
                             feature_manager=_feature_manager,
                             normalize_probabilities=True)

    def send_segmentation(self):
        self.rag.agglomerate(0.5)
        dst = list(self.rag.tree.get_map(0.5))
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
            s0 = self.rag.merge_nodes(s0, s1)
            self.targets.append(MERGE_LABEL)

    def learn_separation(self, segments):
        s0, s1 = segments
        self.features.append(_feature_manager(self.rag, s0, s1))
        self.targets.append(SEPAR_LABEL)

    def relearn(self):
        clf = classify.DefaultRandomForest().fit(self.features, self.targets)
        self.policy = agglo.classifier_probability(_feature_manager, clf)
