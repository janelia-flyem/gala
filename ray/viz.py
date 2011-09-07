from annotefinder import AnnoteFinder
from math import ceil
import numpy as np
import scipy
import evaluate
import matplotlib
plt = matplotlib.pyplot

label=scipy.ndimage.measurements.label

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
    """Plot the voi from segmentations based on Rag and sequence of merges."""
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

def plot_voi_breakdown_panel(px, h, title, xlab, ylab, hlines, **kwargs):
    x = scipy.arange(max(min(px),1e-10), max(px), (max(px)-min(px))/100.0)
    for val in hlines:
        plt.plot(x, val/x, 'g:') 
    plt.scatter(px, h, c=kwargs['c'], label=title)
    af = AnnoteFinder(px, h, [str(i) for i in range(len(px))], 
        xtol=10, ytol=10, xmin=-0.05*max(px), ymin=-0.05*max(px), 
        xmax = 1.05*max(px), ymax=1.05*max(h))
    plt.connect('button_press_event', af)
    plt.xlim(xmin=-0.05*max(px), xmax=1.05*max(px))
    plt.ylim(ymin=-0.05*max(h), ymax=1.05*max(h))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)

def plot_voi_breakdown(seg, gt, ignore_seg=[], ignore_gt=[], 
                                                    hlines=None, subplot=False):
    """Plot conditional entropy H(Y|X) vs P(X) for both seg|gt and gt|seg."""
    plt.ion()
    pxy,px,py,hxgy,hygx,lpygx,lpxgy = evaluate.voi_tables(seg,gt,
            ignore_seg_labels=ignore_seg, ignore_gt_labels=ignore_gt)
    cu = -px*lpygx
    co = -py*lpxgy
    if hlines is None:
        hlines = []
    elif hlines == True:
        hlines = 10
    if type(hlines) == int:
        minc = min(cu[cu!=0].min(), co[co!=0].min())
        maxc = max(cu[cu!=0].max(), co[co!=0].max())
        hlines = np.arange(maxc/hlines, maxc, maxc/hlines)
    plt.figure()
    if subplot: plt.subplot(1,2,1)
    plot_voi_breakdown_panel(px, -lpygx, 
        'Undersegmentation', 'p(S=seg)', 'H(G|S=seg)', 
        hlines, c='r')
    if subplot: plt.subplot(1,2,2)
    plot_voi_breakdown_panel(py, -lpxgy, 
        'Oversegmentation', 'p(G=gt)', 'H(S|G=gt)', 
        hlines, c='b')
    if not subplot:
        plt.title('VOI contributions by body.')
        plt.legend(loc='lower right')
        plt.xlabel('$p_x$')
        plt.ylabel('H(Y|X=x)')

def plot_voi_parts(*args, **kwargs):
    kwargs['subplot'] = True
    plot_voi_breakdown(*args, **kwargs)

def plot_split_voi(ucm, gt, ignore_seg=[], ignore_gt=[], 
                                                linespec='k-', npoints=100):
    ts = np.unique(ucm)[1:]
    if len(ts) > 2*npoints:
        ts = ts[np.arange(1, len(ts), len(ts)/npoints)]
    result = np.zeros((2,len(ts)))
    for i, t in enumerate(ts):
        seg = label(ucm<t)[0]
        result[:,i] = evaluate.split_voi(seg, gt, None, ignore_seg, ignore_gt)
    plt.plot(result[0], result[1], linespec)
    return np.concatenate((ts[np.newaxis, :], result), 0)
