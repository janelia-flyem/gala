import numpy
import agglo
from scipy.ndimage.measurements import label
from scipy.misc import comb as nchoosek

def pixel_wise_boundary_precision_recall(aseg, gt):
    gt = (1-gt.astype(numpy.bool)).astype(numpy.uint8)
    aseg = (1-aseg.astype(numpy.bool)).astype(numpy.uint8)
    tp = float((gt * aseg).sum())
    fp = (aseg * (1-gt)).sum()
    fn = (gt * (1-aseg)).sum()
    return tp/(tp+fp), tp/(tp+fn)

def edit_distance(aseg, gt, ws):
    return edit_distance_to_bps(aseg, agglo.best_possible_segmentation(ws, gt))

def edit_distance_to_bps(aseg, bps):
    r = agglo.Rug(aseg, bps)
    r.overlaps = r.overlaps.astype(numpy.bool)
    false_splits = (r.overlaps.sum(axis=0)-1)[1:].sum()
    false_merges = (r.overlaps.sum(axis=1)-1)[1:].sum()
    return (false_merges, false_splits)


def contingency_table(seg, gt):
    """Return the contingency table for all regions in matched segmentations."""
    # Re-number the segmentations in so they have sequentially-numbered segments
    sel = numpy.zeros((3,3))
    sel[:,1] = 1
    sel[1,:] = 1
    gt,n = label(gt, sel)
    seg,n = label(seg, sel)

    gtr = numpy.ravel(gt)
    segr = numpy.ravel(seg)

    # Get contingency table
    cont = numpy.zeros((numpy.max(gtr)+1, numpy.max(segr)+1))
    for i in range(numpy.size(gt)):
        cont[gtr[i], segr[i]] = cont[gtr[i], segr[i]] + 1

    return cont
    
def voi(X, Y, cont=None):
    """Return the variation of information metric."""
    return numpy.sum(split_voi(X,Y,cont))

def split_voi(X, Y, cont=None):
    """Return the symmetric conditional entropies associated with the VOI.
    
    The variation of information is defined as VI(X,Y) = H(X|Y) + H(Y|X).
    If Y is the ground-truth segmentation, then H(Y|X) can be interpreted
    as the amount of under-segmentation of Y and H(X|Y) is then the amount
    of over-segmentation.  In other words, a perfect over-segmentation
    will have H(Y|X)=0 and a perfect under-segmentation will have H(X|Y)=0.
    """
    if cont is None:
        cont = contingency_table(X, Y)

    n = numpy.sum(cont)
    s1,s2 = numpy.shape(cont)

    # Calculate probabilities
    pxy = cont/float(n)
    px = numpy.sum(pxy,0)
    py = numpy.sum(pxy,1)

    # Calculate log conditional probabilities
    lpygx = numpy.divide(pxy, numpy.tile(px, (s1,1))) # log P(Y|X)
    r,c = numpy.nonzero(pxy)
    lpygx[r,c] = numpy.log2(lpygx[r,c])

    lpxgy = numpy.divide(pxy, numpy.tile(py, (s2,1)).transpose()) # log P(X|Y)
    r,c = numpy.nonzero(pxy)
    lpxgy[r,c] = numpy.log2(lpxgy[r,c])

    # Calculate conditional entropies
    hygx = -numpy.sum(numpy.multiply(pxy, lpygx))
    hxgy = -numpy.sum(numpy.multiply(pxy, lpxgy))
    return hygx, hxgy

def rand_index(seg, gt, cont=None):
    """ Return the unadjusted Rand index. """
    if cont is None:
        cont = contingency_table(seg, gt)

    # Calculate values for rand indices
    n = numpy.sum(cont)
    sum1 = numpy.sum(numpy.power(cont,2))
    sum2 = numpy.sum(numpy.power(numpy.sum(cont,1),2))
    sum3 = numpy.sum(numpy.power(numpy.sum(cont,0),2))
    a = (sum1 - n)/2.0;
    b = (sum2 - sum1)/2
    c = (sum3 - sum1)/2
    d = (sum1 + n**2 - sum2 - sum3)/2

    return (a+d)/(a+b+c+d)
    
def adj_rand_index(seg, gt, cont=None):
    """ Return the adjusted Rand index. """
    if cont is None:
        cont = contingency_table(seg, gt)

    # Calculate values for rand indices
    n = numpy.sum(cont)
    sum1 = numpy.sum(numpy.power(cont,2))
    sum2 = numpy.sum(numpy.power(numpy.sum(cont,1),2))
    sum3 = numpy.sum(numpy.power(numpy.sum(cont,0),2))
    a = (sum1 - n)/2.0;
    b = (sum2 - sum1)/2
    c = (sum3 - sum1)/2
    d = (sum1 + n**2 - sum2 - sum3)/2

    return (nchoosek(n,2)*(a+d) - ((a+b)*(a+c) + (c+d)*(b+d)))/(
        nchoosek(n,2)**2 - ((a+b)*(a+c) + (c+d)*(b+d)))
        
def fm_index(seg, gt, cont=None):
    """ Return the Fowlkes-Mallows index. """
    if cont is None:
        cont = contingency_table(seg, gt)

    # Calculate values for rand indices
    n = numpy.sum(cont)
    sum1 = numpy.sum(numpy.power(cont,2))
    sum2 = numpy.sum(numpy.power(numpy.sum(cont,1),2))
    sum3 = numpy.sum(numpy.power(numpy.sum(cont,0),2))
    a = (sum1 - n)/2.0;
    b = (sum2 - sum1)/2
    c = (sum3 - sum1)/2
    d = (sum1 + n**2 - sum2 - sum3)/2

    return a/(numpy.sqrt((a+b)*(a+c)))

