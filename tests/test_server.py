import os
import threading

import numpy as np

import pytest
from skimage import io
from scipy import ndimage as ndi

from gala import features, agglo, serve, evaluate as ev

D = os.path.dirname(os.path.abspath(__file__))

os.chdir(os.path.join(D, 'example-data/snemi-mini'))


@pytest.fixture
def dummy_data():
    frag0 = np.arange(1, 17, dtype=int).reshape((4, 4))
    gt0 = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3] * 4, [3] * 4], dtype=int)
    frag, gt = (ndi.zoom(image, 4, order=0, mode='reflect')
                for image in [frag0, gt0])
    fman = features.base.Mock(frag, gt)
    return frag, gt, fman


@pytest.fixture
def dummy_data2(dummy_data):
    frag, gt, _ = dummy_data
    frag[7, 7:9] = 17
    frag[7:10, -1] = 18
    fman = features.base.Mock(frag, gt)
    return frag, gt, fman


def test_server(dummy_data):
    frag, gt, fman = dummy_data
    host, port = 'tcp://localhost', 5588
    solver = serve.Solver(frag, feature_manager=fman,
                          port=port, host='tcp://*')
    thread = threading.Thread(target=solver.listen, name='solver')
    thread.start()
    _, dst = serve.proofread(frag, gt, host=host, port=port, num_operations=2,
                             stop_when_finished=True, random_state=0)
    result = np.array(dst)[frag]
    # test: resulting segmentation should be improvement over fragments alone
    assert (ev.vi(result, gt, ignore_x=[], ignore_y=[]) <
            ev.vi(frag, gt, ignore_x=[], ignore_y=[]))


def test_server_imperfect_fragments(dummy_data2):
    frag, gt, fman = dummy_data2
    host, port = 'tcp://localhost', 5589
    solver = serve.Solver(frag, feature_manager=fman,
                          port=port, host='tcp://*')
    thread = threading.Thread(target=solver.listen, name='solver')
    thread.start()
    _, dst = serve.proofread(frag, gt, host=host, port=port, num_operations=2,
                             stop_when_finished=True, random_state=0)
    result = np.array(dst)[frag]
    # test: resulting segmentation should be improvement over fragments alone
    assert (ev.vi(result, gt, ignore_x=[], ignore_y=[]) <
            ev.vi(frag, gt, ignore_x=[], ignore_y=[]))


@pytest.fixture
def data():
    frag, gt, pr = map(io.imread, sorted(os.listdir('.')))
    return frag, gt, pr


@pytest.mark.skipif('GALA_TEST_FULL' not in os.environ,
                    reason=("Test takes too long; "
                            "set GALA_TEST_FULL env variable to run this."))
def test_server_long(data):
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
