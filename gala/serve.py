import time
import numpy as np
import networkx as nx
import json
from sklearn.utils import check_random_state
import zmq
from . import agglo, agglo2, features, classify, evaluate as ev


# constants
# labels for machine learning libs
MERGE_LABEL = 0
SEPAR_LABEL = 1


class Solver(object):
    """ZMQ-based interface between proofreading clients and gala RAGs.

    This docstring is intentionally incomplete until the interface settles.

    Parameters
    ----------

    Attributes
    ----------
    """
    def __init__(self, labels, image=np.array([]),
                 feature_manager=features.default.snemi3d(),
                 address=None, relearn_threshold=20,
                 config_file=None):
        self.labels = labels
        self.image = image
        self.feature_manager = feature_manager
        self._build_rag()
        config_address, id_address = self._configure_from_file(config_file)
        self.id_service = self._connect_to_id_service(id_address)
        self._connect_to_client(address or config_address)
        self.history = []
        self.separate = []
        self.features = []
        self.targets = []
        self.relearn_threshold = relearn_threshold
        self.relearn_trigger = relearn_threshold
        self.recently_solved = True

    def _build_rag(self):
        """Build the region-adjacency graph from the label image."""
        self.rag = agglo.Rag(self.labels, self.image,
                             feature_manager=self.feature_manager,
                             normalize_probabilities=True)
        self.original_rag = self.rag.copy()

    def _configure_from_file(self, filename):
        """Get all configuration parameters from a JSON file.

        The file specification is currently in flux, but looks like:

        ```
        {'id_service_url': 'tcp://localhost:5555',
         'client_url': 'tcp://*:9001',
         'solver_url': 'tcp://localhost:9001'}
        ```

        Parameters
        ----------
        filename : str
            The input filename.

        Returns
        -------
        address : str
            The URL to bind a ZMQ socket to.
        id_address : str
            The URL to bind an ID service to
        """
        if filename is None:
            return None, None
        with open(filename, 'r') as fin:
            config = json.load(fin)
        return (config.get('client_url', None),
                config.get('id_service_url', None))

    def _connect_to_client(self, address):
        self.comm = zmq.Context().socket(zmq.PAIR)
        self.comm.bind(address)

    def _connect_to_id_service(self, url):
        if url is not None:
            service_comm = zmq.Context().socket(zmq.REQ)
            service_comm.connect(url)

            def get_ids(count):
                print('requesting %i ids...' % count)
                service_comm.send_json({'count': count})
                print('receiving %i ids...' % count)
                received = service_comm.recv_json()
                id_range = received['begin'], received['end']
                return id_range
        else:
            def get_ids(count):
                start = np.max(self.labels) + 2
                return start, start + count
        return get_ids

    def send_segmentation(self):
        """Send a segmentation to ZMQ as a fragment-to-segment lookup table.

        The format of the lookup table (LUT) is specified in the BigCat
        wiki [1]_.

        References
        ----------
        .. [1] https://github.com/saalfeldlab/bigcat/wiki/Actors,-responsibilities,-and-inter-process-communication
        """
        if len(self.targets) < self.relearn_threshold:
            print('server has insufficient data to resolve')
            return
        self.relearn()  # correct way to do it is to implement RAG splits
        self.rag.agglomerate(0.5)
        self.recently_solved = True
        dst_tree = [int(i) for i in self.rag.tree.get_map(0.5)]
        unique = set(dst_tree)
        start, end = self.id_service(len(unique))
        remap = dict(zip(unique, range(start, end)))
        dst = list(map(remap.__getitem__, dst_tree))
        src = list(range(len(dst)))
        message = {'type': 'fragment-segment-lut',
                   'data': {'fragments': src, 'segments': dst}}
        print('server sending:', message)
        try:
            self.comm.send_json(message, flags=zmq.NOBLOCK)
        except zmq.error.Again:
            return

    def listen(self, send_every=None):
        """Listen to ZMQ port for instructions and data.

        The instructions conform to the proofreading protocol defined in the
        BigCat wiki [1]_.

        Parameters
        ----------
        send_every : int or float, optional
            Send a new segmentation every `send_every` seconds.

        References
        ----------
        .. [1] https://github.com/saalfeldlab/bigcat/wiki/Actors,-responsibilities,-and-inter-process-communication
        """
        start_time = time.time()
        recv_flags = zmq.NOBLOCK
        while True:
            if send_every is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time > send_every:
                    print('server resolving')
                    self.send_segmentation()
                    start_time = time.time()
            try:
                if recv_flags == zmq.NOBLOCK:
                    print('server receiving no blocking...')
                else:
                    print('server receiving blocking...')
                message = self.comm.recv_json(flags=recv_flags)
                print('server received:', message)
                recv_flags = zmq.NOBLOCK
            except zmq.error.Again:  # no message received
                recv_flags = zmq.NULL
                print('server: no message received in time')
                if not self.recently_solved:
                    print('server resolving')
                    self.send_segmentation()
                continue
            command = message['type']
            data = message['data']
            if command == 'merge':
                segments = data['segments']
                self.learn_merge(segments)
            elif command == 'separate':
                fragment = data['fragment']
                separate_from = data['from']
                self.learn_separation(fragment, separate_from)
            elif command == 'request':
                what = data['what']
                if what == 'fragment-segment-lut':
                    self.send_segmentation()
            elif command == 'stop':
                return
            else:
                print('command %s not recognized.' % command)
                continue

    def learn_merge(self, segments):
        """Learn that a pair of segments should be merged.

        Parameters
        ----------
        segments : tuple of int
            A pair of segment identifiers.
        """
        segments = set(self.rag.tree.highest_ancestor(s) for s in segments)
        # ensure the segments are ordered such that every subsequent
        # pair shares an edge
        ordered = nx.dfs_preorder_nodes(nx.subgraph(self.rag, segments))
        s0 = next(ordered)
        for s1 in ordered:
            self.features.append(self.feature_manager(self.rag, s0, s1))
            self.history.append((s0, s1))
            s0 = self.rag.merge_nodes(s0, s1)
            self.targets.append(MERGE_LABEL)
        self.recently_solved = False

    def learn_separation(self, fragment, separate_from):
        """Learn that a pair of fragments should never be in the same segment.

        Parameters
        ----------
        fragments : tuple of int
            A pair of fragment identifiers.
        """
        f0 = fragment
        if not separate_from:
            separate_from = self.original_rag.neighbors(f0)
        s0 = self.rag.tree.highest_ancestor(f0)
        for f1 in separate_from:
            if self.rag.boundary_body in (f0, f1):
                continue
            s1 = self.rag.tree.highest_ancestor(f1)
            if self.rag.has_edge(s0, s1):
                self.features.append(self.feature_manager(self.rag, s0, s1))
                self.targets.append(SEPAR_LABEL)
            if self.original_rag.has_edge(f0, f1):
                self.features.append(self.feature_manager(self.original_rag,
                                                          f0, f1))
                self.targets.append(SEPAR_LABEL)
            self.separate.append((f0, f1))
        self.recently_solved = False

    def relearn(self):
        """Learn a new merge policy using data gathered so far.

        This resets the state of the RAG to contain only the merges and
        separations received over the course of its history.
        """
        clf = classify.DefaultRandomForest().fit(self.features, self.targets)
        self.policy = agglo.classifier_probability(self.feature_manager, clf)
        self.rag = self.original_rag.copy()
        self.rag.merge_priority_function = self.policy
        self.rag.rebuild_merge_queue()
        for i, (s0, s1) in enumerate(self.separate):
            self.rag.node[s0]['exclusions'].add(i)
            self.rag.node[s1]['exclusions'].add(i)


def proofread(fragments, true_segmentation, host='tcp://localhost', port=5556,
              num_operations=10, mode='fast paint', stop_when_finished=False,
              request_seg=True, random_state=None):
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
    stop_when_finished : bool, optional
        Send the solver a "stop" action when done proofreading. Useful
        when running tests so we don't intend to continue proofreading.
    random_state : None or int or numpy.RandomState instance, optional
        Fix the random state for proofreading.

    Returns
    -------
    lut : tuple of array-like of int
        A look-up table from fragments (first array) to segments
        (second array), obtained by requesting it from the Solver after
        initial proofreading simulation.
    """
    true = agglo2.best_segmentation(fragments, true_segmentation)
    base_graph = agglo2.fast_rag(fragments)
    comm = zmq.Context().socket(zmq.PAIR)
    comm.connect(host + ':' + str(port))
    ctable = ev.contingency_table(fragments, true).tocsc()
    true_labels = np.unique(true)
    random = check_random_state(random_state)
    random.shuffle(true_labels)
    for _, label in zip(range(num_operations), true_labels):
        time.sleep(3)
        components = [int(i) for i in ctable.getcol(int(label)).indices]
        merge_msg = {'type': 'merge', 'data': {'segments': components}}
        print('proofreader sends:', merge_msg)
        comm.send_json(merge_msg)
        for fragment in components:
            others = [int(neighbor) for neighbor in base_graph[fragment]
                      if neighbor not in components]
            if not others:
                continue
            split_msg = {'type': 'separate',
                         'data': {'fragment': int(fragment), 'from': others}}
            print('proofreader sends:', split_msg)
            comm.send_json(split_msg)
    if request_seg:  # if no request, assume server sends periodic updates
        req_msg = {'type': 'request', 'data': {'what': 'fragment-segment-lut'}}
        print('proofreader sends:', req_msg)
        comm.send_json(req_msg)
    print('proofreader receiving...')
    response = comm.recv_json()
    print('proofreader received:', response)
    src = response['data']['fragments']
    dst = response['data']['segments']
    if stop_when_finished:
        stop_msg = {'type': 'stop', 'data': {}}
        print('proofreader sends: ', stop_msg)
        comm.send_json(stop_msg)
    return src, dst
