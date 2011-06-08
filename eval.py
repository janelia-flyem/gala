from numpy import bool, uint8
import agglo

def pixel_wise_boundary_precision_recall(aseg, gt):
    gt = (1-gt.astype(bool)).astype(uint8)
    aseg = (1-aseg.astype(bool)).astype(uint8)
    tp = float((gt * aseg).sum())
    fp = (aseg * (1-gt)).sum()
    fn = (gt * (1-aseg)).sum()
    return tp/(tp+fp), tp/(tp+fn)

def edit_distance(aseg, gt, ws):
    return edit_distance_to_bps(aseg, agglo.best_possible_segmentation(ws, gt))

def edit_distance_to_bps(aseg, bps):
    r = agglo.Rug(aseg, bps)
    r.overlaps = r.overlaps.astype(bool)
    false_splits = (r.overlaps.sum(axis=0)-1)[1:].sum()
    false_merges = (r.overlaps.sum(axis=1)-1)[1:].sum()
    return (false_merges, false_splits)

def body_rand(seg, gt):
	"""Calculates the rand index of a segmentation versus ground truth.
		Returns the rand index, adjusted rand index, and Fowlkes-Mallows index
		
		Equations based on the paper 
			"On the Use of the Adjusted Rand Index as a Metric for 
			Evaluating Supervised Classication", Santos and Embrechts
	"""
	# Initialize variables
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	gtr = gt.ravel()
	segr = seg.ravel()
	
	# Re-number the segmentations in so they have sequentially-numbered segments
	ctr = 0
	for i in numpy.unique(gtr):
		gtr[numpy.nonzero(gtr==i)] = ctr
		ctr = ctr + 1
	ctr = 0
	for i in numpy.unique(segr):
		segr[numpy.nonzero(segr==i)] = ctr
		ctr = ctr + 1
	
	# Get contingency table
	cont = numpy.zeros((numpy.max(gtr)+1, numpy.max(segr)+1))
	for i in range(numpy.size(gt)):
		cont[gtr[i], segr[i]] = cont[gtr[i], segr[i]] + 1
		
	# Calculate values
	n = numpy.sum(cont)
	a = (numpy.sum(numpy.power(cont,2)) - n)/2.0;
	b = (numpy.sum(numpy.power(numpy.sum(cont,1),2)) - 
		numpy.sum(numpy.power(cont,2)))/2
	c = (numpy.sum(numpy.power(numpy.sum(cont,0),2)) - 
		numpy.sum(numpy.power(cont,2)))/2
	d = (numpy.sum(numpy.power(cont,2)) + n**2 - 
		numpy.sum(numpy.power(numpy.sum(cont,1),2)) - 
		numpy.sum(numpy.power(numpy.sum(cont,0),2)))/2
		
	# Get indices
	RI = (a+d)/(a+b+c+d)
	ARI = (nchoosek(n,2)*(a+d) - ((a+b)*(a+c) + (c+d)*(b+d)))/(
		nchoosek(n,2)**2 - ((a+b)*(a+c) + (c+d)*(b+d)))
	FM = a/(numpy.sqrt((a+b)*(a+c)))
		
	return (RI, ARI, FM)
    
