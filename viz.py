from math import ceil
import numpy as np
import evaluate
import matplotlib
plt = matplotlib.pyplot

###########################
# VISUALIZATION FUNCTIONS #
###########################

def imshow_grey(im):
    return plt.imshow(im, cmap=plt.cm.gray, interpolation='nearest')

def imshow_jet(im):
    return plt.imshow(im, cmap=plt.cm.jet, interpolation='nearest')

def imshow_rand(im):
    rcmap = matplotlib.colors.ListedColormap(np.concatenate(
        (np.zeros((1,3)), np.random.rand(ceil(im.max()), 3))
    ))
    return plt.imshow(im, cmap=rcmap, interpolation='nearest')

def inspect_segs_3D(*args, **kwargs):
    """Show corresponding slices side by side in multiple segmentations."""
    z = 0
    if kwargs.has_key('z'):
        z = kwargs['z']
    axis=-1
    if kwargs.has_key('axis'):
        axis = kwargs['axis']
    numplots = 0
    im = None
    if kwargs.has_key('image'):
        im = kwargs['image']
        numplots += 1
    fignum = 1
    if kwargs.has_key('fignum'):
        fignum = kwargs['fignum']
    prob = None
    if kwargs.has_key('prob'):
        prob = kwargs['prob']
        numplots += 1
    numplots += len(args)
    plot_arrangements = []
    for i in range(1,4):
        for j in range(i,4):
            plot_arrangements.append((i*j, i,j))
    # first plot arrangement 
    plot_arrangement = [(i,j) for p,i,j in plot_arrangements
                                                    if p >= numplots][0]
    fig = plt.figure(fignum)
    current_subplot = 1
    if im is not None:
        plt.subplot(*plot_arrangement+(current_subplot,))
        imshow_grey(im.swapaxes(0,axis)[z])
        current_subplot += 1
    if prob is not None:
        plt.subplot(*plot_arrangement+(current_subplot,))
        imshow_jet(prob.swapaxes(0,axis)[z])
        current_subplot += 1
    for i, j in enumerate(range(current_subplot, numplots+1)):
        plt.subplot(*plot_arrangement+(j,))
        imshow_rand(args[i].swapaxes(0,axis)[z])
    return fig

def plot_voi(r, gt, fig=None):
    """ Plot the voi from segmentations output from agglo.agglom_segmentations(). """
    v = []
    n = []
    for i in r:
        v.append(evaluate.voi(i[0], gt))
        n.append(len(np.unique(i[0])-1))
    if fig is None:
        fig = plt.figure()
    plt.plot(n, v, figure = fig)
    plt.xlabel('Number of segments', figure = fig)
    plt.ylabel('VOI', figure = fig)

def plot_num_segments(r, gt, num, fig=None):
    """ Plot a segmentation with a specific number of semgents.
    
        The first input should be the output from agglo.agglom_segmentations(). 
    
    """
    i = len(r)-1
    n = len(np.unique(r[i])-1)
    while n > num:
        i -= 1
        n = len(np.unique(r[i])-1)
    v = evaluate.voi(r[i][0], gt)
    if fig is None:
        fig = plt.figure()
    plt.imshow(r[i][0], figure = fig)
    plt.title('VOI: ' + str(v) + '\t Number of segments: ' + str(n), figure = fig)
    
