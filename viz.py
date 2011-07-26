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

def plot_voi(a, history, gt, fig=None):
    """ Plot the voi from segmentations based on a Rag and a sequence of merges. """
    v = []
    n = []
    seg = a.get_segmentation()
    for i in history:
        seg[seg==i[1]] = i[0]
        v.append(evaluate.voi(seg, gt))
        n.append(len(np.unique(seg)-1))
    if fig is None:
        fig = plt.figure()
    plt.plot(n, v, figure = fig)
    plt.xlabel('Number of segments', figure = fig)
    plt.ylabel('VOI', figure = fig)

def plot_voi_parts(seg, gt, ignore_seg_labels=[], ignore_gt_labels=[]):
    """Given a segmentation and ground truth, plot the size of segments versus the conditional entropy."""
    pxy,px,py,hxgy,hygx,lpygx,lpxgy = evaluate.voi_tables(seg,gt,
	ignore_seg_labels=ignore_seg_labels,ignore_gt_labels=ignore_gt_labels)
    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(px, -lpygx, c=-px*lpygx)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel('p(seg)')
    plt.ylabel('H(GT|SEG=seg)')
    plt.title('Undersegmentation')
    plt.subplot(1,2,2)
    plt.scatter(py,-lpxgy, c=-py*lpxgy)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel('p(gt)')
    plt.ylabel('H(SEG|GT=gt)')
    plt.title('Oversegmentation')
