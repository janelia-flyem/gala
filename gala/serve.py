from . import agglo, features
import zmq
import json
import ujson


_feature_manager = features.default.snemi3d()

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


class Solver(object):
    '''
    ZMQ-based interface between proofreading clients and gala RAGs.

    Parameters
    ----------

    Attributes
    ----------
    '''
    def __init__(self, labels, image, port=5556, host='tcp://localhost'):
        self.rag = agglo.Rag(labels, image,
                             feature_manager=_feature_manager,
                             normalize_probabilities=True)
        self.comm = zmq.Context().socket(zmq.PAIR)
        self.comm.connect(host + ':' + str(port))

    def send_segmentation(self):
        self.rag.agglomerate(0.5)
        dst = list(self.rag.tree.get_map(0.5))
        src = list(range(len(dst)))
        message = {'type': 'region-lut',
                   'data': {'src': src, 'dst': dst}}
        self.comm.send_json(message)

    def listen(self):
        while True:
            message = self.comm.recv_json()
            command = message['type']
            data = message['data']
            if command == 'merge':
                regions = data['regions']
                self.learn_merge(regions)
            elif command == 'separate':
                regions = data['regions']
                self.learn_separation(regions)
            elif command == 'request':
                what = data['what']
                if what == 'region-lut':
                    self.send_segmentation()
            elif command == 'stop':
                return

    def learn_merge(self, regions):
        def top_level(n):  # speed this up by adding a function to viridis
            anc = self.rag.tree.ancestors(n)
            if anc == []:
                return n
            else:
                return anc[-1]
        regions = iter(set(map(top_level, regions)))
        r0 = next(regions)
        for r1 in regions:
            self.features.append(_feature_manager(self.rag, r0, r1))
            r0 = self.rag.merge_nodes(r0, r1)
            self.targets.append(0)
