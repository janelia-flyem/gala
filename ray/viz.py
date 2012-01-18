from annotefinder import AnnoteFinder
from math import ceil
import numpy as np
import scipy
import evaluate
import morpho
import matplotlib
plt = matplotlib.pyplot
cm = plt.cm
from itertools import cycle

label=scipy.ndimage.measurements.label
center_of_mass=scipy.ndimage.measurements.center_of_mass

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

def plot_vi(a, history, gt, fig=None):
    """Plot the VI from segmentations based on Rag and sequence of merges."""
    v = []
    n = []
    seg = a.get_segmentation()
    for i in history:
        seg[seg==i[1]] = i[0]
        v.append(evaluate.vi(seg, gt))
        n.append(len(np.unique(seg)-1))
    if fig is None:
        fig = plt.figure()
    plt.plot(n, v, figure = fig)
    plt.xlabel('Number of segments', figure = fig)
    plt.ylabel('vi', figure = fig)

def plot_vi_breakdown_panel(px, h, title, xlab, ylab, hlines, **kwargs):
    x = scipy.arange(max(min(px),1e-10), max(px), (max(px)-min(px))/100.0)
    for val in hlines:
        plt.plot(x, val/x, c='gray', ls=':') 
    plt.scatter(px, h, label=title, **kwargs)
    af = AnnoteFinder(px, h, [str(i) for i in range(len(px))], 
        xtol=0.005, ytol=0.005, xmin=-0.05*max(px), ymin=-0.05*max(px), 
        xmax = 1.05*max(px), ymax=1.05*max(h))
    plt.connect('button_press_event', af)
    plt.xlim(xmin=-0.05*max(px), xmax=1.05*max(px))
    plt.ylim(ymin=-0.05*max(h), ymax=1.05*max(h))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)

def plot_vi_breakdown(seg, gt, ignore_seg=[], ignore_gt=[], 
                                        hlines=None, subplot=False, **kwargs):
    """Plot conditional entropy H(Y|X) vs P(X) for both seg|gt and gt|seg."""
    plt.ion()
    pxy,px,py,hxgy,hygx,lpygx,lpxgy = evaluate.vi_tables(seg,gt,
            ignore_seg=ignore_seg, ignore_gt=ignore_gt)
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
    plot_vi_breakdown_panel(px, -lpygx, 
        'False merges', 'p(S=seg)', 'H(G|S=seg)', 
        hlines, c='blue', **kwargs)
    if subplot: plt.subplot(1,2,2)
    plot_vi_breakdown_panel(py, -lpxgy, 
        'False splits', 'p(G=gt)', 'H(S|G=gt)', 
        hlines, c='orange', **kwargs)
    if not subplot:
        plt.title('vi contributions by body.')
        plt.legend(loc='upper right', scatterpoints=1)
        plt.xlabel('Segment size (fraction of volume)', fontsize='large')
        plt.ylabel('Conditional entropy (bits)', fontsize='large')
        xmax = max(px.max(), py.max())
        plt.xlim(-0.05*xmax, 1.05*xmax)
        ymax = max(-lpygx.min(), -lpxgy.min())
        plt.ylim(-0.05*ymax, 1.05*ymax)

def plot_vi_parts(*args, **kwargs):
    kwargs['subplot'] = True
    plot_vi_breakdown(*args, **kwargs)

def add_opts_to_plot(ars, colors='k', markers='^', **kwargs):
    if type(colors) not in [list, tuple]:
        colors = [colors]
    if len(colors) < len(ars):
        colors = cycle(colors)
    if type(markers) not in [list, tuple]:
        markers = [markers]
    if len(markers) < len(ars):
        markers = cycle(markers)
    points = []
    for ar, c, m in zip(ars, colors, markers):
        opt = ar[:,ar.sum(axis=0).argmin()]
        points.append(plt.scatter(opt[0], opt[1], c=c, marker=m, **kwargs))
    return points

def add_nats_to_plot(ars, tss, stops=0.5, colors='k', markers='o', **kwargs):
    if type(colors) not in [list, tuple]: colors = [colors]
    if len(colors) < len(ars): colors = cycle(colors)
    if type(markers) not in [list, tuple]: markers = [markers]
    if len(markers) < len(ars): markers = cycle(markers)
    if type(stops) not in [list, tuple]: stops = [stops]
    if len(stops) < len(ars): stops = cycle(stops)
    points = []
    for ar, ts, stop, c, m in zip(ars, tss, stops, colors, markers):
        nat = ar[:,np.flatnonzero(ts<stop)[-1]]
        points.append(plt.scatter(nat[0], nat[1], c=c, marker=m, **kwargs))
    return points

def plot_split_vi(ars, best=None, colors='k', linespecs='-', 
                                        addopt=None, addnat=None, **kwargs):
    if type(ars) not in [list, tuple]: ars = [ars]
    if type(colors) not in [list, tuple]: colors = [colors]
    if len(colors) < len(ars): colors = cycle(colors)
    if type(linespecs) not in [list, tuple]: linespecs = [linespecs]
    if len(linespecs) < len(ars): linespecs = cycle(linespecs)
    lines = []
    for ar, color, linespec in zip(ars, colors, linespecs):
        lines.append(plt.plot(ar[0], ar[1], c=color, ls=linespec, **kwargs))
    if best is not None:
        lines.append(plt.scatter(
            best[0], best[1], 
            c=kwargs.get('best-color', 'k'), marker=(5,3,0), **kwargs)
        )
    return lines

def jet_transparent(im, alpha=0.5):
    im = cm.jet(im.astype(np.double)/im.max(), alpha=alpha)
    im[(im[...,:3]==np.array([0,0,0.5])).all(axis=-1)] = np.array([0,0,0.5,0])
    return im

def show_merge(im, bdr, z, ax=0, alpha=0.5, **kwargs):
    plt.figure(figsize=kwargs.get('figsize', (3.25,3.25)))
    im = im.swapaxes(0,ax)[z]
    bdr = bdr.swapaxes(0,ax)[z]
    bdr = jet_transparent(bdr, alpha)
    imshow_grey(im)
    plt.imshow(bdr)
    plt.xticks([])
    plt.yticks([])

def show_merge_3D(g, n1, n2, **kwargs):
    im = kwargs.get('image', None)
    alpha = kwargs.get('alpha', 0.5)
    fignum = kwargs.get('fignum', 10)
    bdr = np.zeros(g.segmentation.shape, np.uint8)
    bdri = list(g[n1][n2]['boundary'])
    bdr.ravel()[bdri] = 3
    bdr.ravel()[list(g.node[n1]['extent'])] = 1
    bdr.ravel()[list(g.node[n2]['extent'])] = 2
    bdr = morpho.juicy_center(bdr, g.pad_thickness)
    x, y, z = np.array(center_of_mass(bdr==3)).round().astype(np.uint32)
    fig = plt.figure(fignum)
    bdr_cmapped = jet_transparent(bdr, alpha)
    plt.subplot(221); imshow_grey(im[:,:,z]); \
                     plt.imshow(bdr_cmapped[:,:,z], interpolation='nearest')
    plt.subplot(222); imshow_grey(im[:,y,:]); \
                     plt.imshow(bdr_cmapped[:,y,:], interpolation='nearest')
    plt.subplot(223); imshow_grey(im[x,:,:]); \
                     plt.imshow(bdr_cmapped[x,:,:], interpolation='nearest')
    plt.subplot(224); _ = plt.hist(g.probabilities_r[bdri][:,0], bins=25)
    return bdr, (x,y,z)
