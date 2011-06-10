

def compute_W(a, agglom_function, sigma=255.0*20):
	""" Computes the weight matrix of an agglo object """
	nodes = a.nodes()
	n = len(nodes)
	W = scipy.sparse.lil_matrix((n,n))
	for edge in a.edges_iter():
		e1 = edge[0]
		e2 = edge[1]
		val = agglom_function(a,e1,e2)
		ind1 = numpy.nonzero(nodes==e1)[0][0]
		ind2 = numpy.nonzero(nodes==e2)[0][0]
		W[ind1, ind2] = numpy.exp(-val**2/sigma)
		W[ind2, ind1] = W[ind1, ind2]
	return W

def labels_to_segmentation(a, labels):
	""" Given an agglomeration and a set of superpixel labels,
		returns the associated segmentation.
	"""
	ws = a.watershed
	nodes = a.nodes()
	aseg = a.get_segmentation()
	n,m = numpy.shape(ws)

	SegLabel = numpy.zeros((n,m))
	for i in numpy.unique(labels):
		# Find which superpixels are to be given this label
		superpixels = numpy.nonzero(labels==i)[0]
		for sp in superpixels:
			# Label the actualy image pixels with the label
			row,col = numpy.nonzero(aseg==nodes[sp])
			SegLabel[row,col] = i
	# Label the pixels on the boundary between superpixels
	SegLabel = morpho.pad(SegLabel, [0, 0])
	SegLabel = SegLabel.ravel()
	for edge in a.edges_iter():
		e1 = edge[0]
		e2 = edge[1]
		ind1 = numpy.nonzero(nodes==e1)[0][0]
		ind2 = numpy.nonzero(nodes==e2)[0][0]
		if labels[ind1]==labels[ind2]:
			# Adjacent superpixels have the same label
			SegLabel[list(a[e1][e2]['boundary'])] = labels[ind1]
	SegLabel = numpy.reshape(temp, (n+4,m+4))
	SegLabel = SegLabel[2:-2,2:-2]

	return SegLabel

def ncut(W, num_eigs=10, offset = 1e-5, kmeans_iters=10):
	""" Runs the normalized cut algorithm, as implemented in (Ng, Jordan, and Weiss, 2002)

		Returns labels, eigenvectors, and eigenvalues
	"""

	n,m = numpy.shape(W)
	# Add an offset in case some rows are zero
	W = W + scipy.sparse.spdiags(offset*numpy.ones(n),0,n,n)

	# Calculate matrix to take eigenvectors of
	rows, cols = W.nonzero()
	d = W.sum(1).transpose()
	Dinv2 = scipy.sparse.spdiags(1./numpy.sqrt(d), 0, n, n)
	P = Dinv2*W*Dinv2;

	# Get the eigenvectors and sort by eigenvalue
	eval,U = scipy.sparse.linalg.eigs(P, num_eigs, which='LR')
	ind = numpy.argsort(eval)[::-1]
	eval = eval[ind]
	U = U[:,ind]

	# Normalize
	for i in range(n):
		u = U[i,:]
		U[i,:] = U[i,:]/scipy.linalg.norm(u)

	# Cluster them into labels, running k-means multiple times
	labels_list = []
	distortion_list = []
	for iter in range(kmeans_iters):
		# Cluster
		centroid, labels = scipy.cluster.vq.kmeans2(U, num, minit='points')
		# Calculate distortion
		distortion = 0
		for j in range(num):
			numvals = (sum(labels==j))
			if numvals==0: continue
			distortion = distortion + \
				1.0/float(numvals)* \
				sum([norm(v - centroid[j])**2 for (i,v) in enumerate(U) if labels[i]==j])
		# Save values
		labels_list.append(labels)
		distortion_list.append(distortion)
	# Use lowest distortion
	ind = numpy.argmin(distortions)
	labels = labelss[ind]

	return label, U, eval