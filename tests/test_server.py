import os
import threading

import numpy as np

import pytest
from skimage import io

from gala import serve, evaluate as ev

D = os.path.dirname(os.path.abspath(__file__))

os.chdir(os.path.join(D, 'example-data/snemi-mini'))

@pytest.fixture
def data():
    frag, gt, pr = map(io.imread, sorted(os.listdir('.')))
    return frag, gt, pr


def test_server(data):
    frag, gt, pr = data
    host, port = 'tcp://localhost', 5590
    solver = serve.Solver(frag, pr, port=port, host='tcp://*')
    thread = threading.Thread(target=solver.listen, name='solver')
    thread.start()
    _, dst = serve.proofread(frag, gt, host=host, port=port,
                             stop_when_finished=True, random_state=0)
    result = np.array(dst)[frag]
    # test: resulting segmentation should be improvement over fragments alone
    assert (ev.vi(result, gt, ignore_x=[], ignore_y=[]) <
            ev.vi(frag, gt, ignore_x=[], ignore_y=[]))
