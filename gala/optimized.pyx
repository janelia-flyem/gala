import numpy as np
cimport numpy as np

def despeckle_watershed(ws, in_place=True):
    """ Function to clean up dots in an initial oversegmentation. 

    If all instances of a label are surrounded by one second label, then we
    change those pixels to the second label. Useful for dealing with tiny dots
    when running watershed on un-thresholded probability maps. Assumes all
    labels appear only on one contiguous blob.

    Parameters
    ----------
    ws : ndarray of 2 or 3 dimensions
        an image or stack of images to be cleaned up. If it is a stack, each
        plane is considered separately. 
    in_place : boolean, optional
        whether to modify the original images or create a copy. 
        defaults to True.

    Returns
    -------
    out : ndarray, same shape and type as `ws`
        the original image with all holes filled in their surrounding label
    """
    cdef int ii
    if not in_place: ws = ws.copy()
    if ws.ndim == 3:
        for ii in range(ws.shape[0]):
            ws[ii,:,:] = _despeckle_2d_watershed(ws[ii,:,:])
        return ws
    else:
        return _despeckle_2d_watershed(ws)


cdef _despeckle_2d_watershed(long[:,:] ws):
    """ workhorse function for despeckle_watershed, see its documentation """
    cdef int ii, jj, i_offset, j_offset, label
    neighborhoods = {}
    replacements = {}
    for ii in range(ws.shape[0]):
        for jj in range(ws.shape[1]):
            if ws[ii,jj] not in neighborhoods:
                neighborhoods[ws[ii,jj]] = []
            for i_offset in [-1, 0, 1]:
                if i_offset+ii < 0 or i_offset+ii >= ws.shape[0]:
                    continue
                for j_offset in [-1, 0, 1]:
                    if j_offset+jj < 0 or j_offset+jj >= ws.shape[1]:
                        continue
                    label = ws[ii+i_offset, jj+j_offset]
                    if label == ws[ii,jj] or label in neighborhoods[ws[ii,jj]]:
                        continue
                    neighborhoods[ws[ii,jj]].append(label)
    for label, neighbors in neighborhoods.iteritems():
        if len(neighbors) == 1:
            replacements[label] = neighbors[0]
        else:
            replacements[label] = label
    for ii in range(ws.shape[0]):
        for jj in range(ws.shape[1]):
            ws[ii,jj] = replacements[ws[ii,jj]]
    return ws

def flood_fill(im, start, acceptable, limits=None, raveled=False):
    """ Find all connected points in a 3D volume.

    Starting from a given point, this function floods into all adjacent
    points that have an acceptable label.

    Parameters
    ----------
    im : 3D ndarray of longs
        This is the volume in which the flood fill will fill. Each voxel's
        value is its label and its indices are its position.
    start : 1D ndarray of longs
        This gives the position of the first point from which the flood fill 
        will begin. must be length 3.
    acceptable : 1D ndarray of longs
        As the flood fills, each pixel is checked to see if its value is in
        this list. the flood fill continues into that pixel iff it is.
    limits : 2D ndarray of longs, optional
        Allows the calling function specify an outer bound for the flood fill.
        Given as a matrix where each row represents a dimension in im and the
        first column is the lower bound and the second column is the upper.
        Defaults to None, which uses the entire volume.
    raveled : boolean, optional
        Specifies whether to return the flooded pixels as coordinates or as
        raveled indices.

    Returns
    -------
    matches : ndarray
        either a 1D array of raveled indices of pixels in im, or a 
        2D ndarray where each row is the coordinates of a pixel.
    """
    cdef np.ndarray[np.int_t, ndim=2] matches
    if im.ndim == 3:
        a = np.array(acceptable)
        s = np.array(start)
        if limits == None:
            limits = np.array([[0,im.shape[0]-1],[0,im.shape[1]-1],[0,im.shape[2]-1]])
        matches = _flood_fill_3d(im, s, a, limits)
        if _list_match(a, im[s[0], s[1], s[2]]) == -1:
            return np.array([])
        if raveled:
            formatted = matches.T
            return np.ravel_multi_index(formatted, im.shape)
        else:
            return matches
    else: raise ValueError("flood fill volume must be 3d!")

cdef inline _row_match(long[:,:] rows, long[:] query, long limit):
    """ Fast check if the `query` array is a row in `rows`
    
    Parameters
    ----------
    rows : ndarray, 2 dimensions, type long
        Each row represents a unique potential match.
    query : ndarray, 1 dimension, type long
        The group of numbers we wish to know if appears in `rows`
    limit : long
        Only searches the first `limit` many rows.

    Returns
    -------
    rr : long
        The row of the first occurence of `query`, or -1 if not found.
    """
    cdef int rr, jj, match
    for rr in range(limit):
        match = 1
        for jj in range(rows.shape[1]):
            if query[jj] != rows[rr,jj]:
                match = 0
                break
        if match == 1: return rr
    return -1

cdef inline _list_match(long[:] vals, long query):
    """ Fast check if a long is in a list of longs
    
    Parameters
    ----------
    vals : ndarray, 1 dimension, type long
        The list of acceptable values.
    query : long
        The value we wish to know whether is in vals.
    
    Returns
    -------
    jj : long
        The index of the first occurence of query in vals,
        or -1 if not found.
    """
    cdef int jj
    for jj in range(vals.shape[0]):
        if vals[jj] == query: return jj
    return -1

cdef _flood_fill_3d(long[:,:,:] im, long[:] start, long[:] acceptable, long[:,:] limits):
    """ Workhorse function for flood_fill, see its documentation.
    Assumes im[start] is in acceptable and start is within limits. """
    cdef int frontier_size = 1
    cdef int matches_size = 1
    cdef int starting_size = 5000
    cdef int ndim = im.ndim
    cdef int in_matches, in_acceptable, base_point_ii, p_ii, new_frontier_size, jj
    cdef np.ndarray[np.int_t, ndim=2] adjacent
    cdef np.ndarray[np.int_t, ndim=2] matches =      np.zeros([starting_size, ndim], dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=2] frontier =     np.empty([starting_size, ndim], dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=2] new_frontier = np.empty([starting_size, ndim], dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] base_point, p

    for jj in range(ndim):
        frontier[0,jj] = start[jj]
        matches[0,jj] = start[jj]
    while frontier_size > 0:
        new_frontier_size = 0
        for base_point_ii in range(frontier_size):
            base_point = frontier[base_point_ii]
            adjacent = _adjacent_points(base_point, limits)
            for p_ii in range(adjacent.shape[0]):
                p = adjacent[p_ii]
                if _list_match(p, -1) != -1:
                    continue
                if _row_match(matches, p, matches_size) > -1:
                    continue
                if _list_match(acceptable, im[p[0],p[1],p[2]]) == -1:
                    continue
                new_frontier[new_frontier_size] = p
                matches[matches_size] = p
                matches_size += 1
                new_frontier_size += 1
                if matches_size >= matches.shape[0]:
                    matches = _expand_2darray(matches)
                if new_frontier_size >= new_frontier.shape[0]:
                    new_frontier = _expand_2darray(new_frontier)
                    frontier = _expand_2darray(frontier)
        for p_ii in range(new_frontier_size):
            for jj in range(new_frontier.shape[1]):
                frontier[p_ii,jj] = new_frontier[p_ii, jj]
        frontier_size = new_frontier_size
    return matches[0:matches_size, :]

 
cdef np.ndarray[np.int_t, ndim=2] _expand_2darray(long[:,:] a):
    """ Double the number of rows in a matrix and copy over the existing values
    """
    cdef int ii,jj
    cdef np.ndarray[np.int_t, ndim=2] expanded = np.zeros([a.shape[0] * 2, a.shape[1]], dtype=np.int)
    for ii in range(a.shape[0]):
        for jj in range(a.shape[1]):
            expanded[ii,jj] = a[ii,jj]
    return expanded


cdef np.ndarray[np.int_t, ndim=2] _adjacent_points(long[:] point, long[:,:] limits):
    """ Get all adjacent points to point that fall within limits. 
        
    Parameters
    ----------
    point : ndarray, 1 dimension, type long
        the coordinates of an n-dimensional point    
    limits : ndarray, 2 dimensions, type long
        point.shape[0] rows, 2 columns, each row of this array represents the
        lower and upper (inclusive) bounds of acceptable values for
        points that are returned.

    Returns
    -------
    adjacent : ndarray, 2 dimensions, type long
        2 rows for each "dimension" of point, one column for each dimension
        Each row represents an adjacent points to `point`. Invalid points are
        left as -1 across all columns.
    """
    cdef int dimensions = point.shape[0]
    cdef int variants = dimensions * 2
    cdef int v = -1
    cdef int d, s, new,ii
    cdef int[2] shifts
    cdef np.ndarray[np.int_t, ndim=2] adjacent = np.ones([variants, dimensions], dtype=np.int)*-1
    shifts[0] = -1
    shifts[1] = 1
    for d in range(dimensions):
        for s in shifts:
            v += 1
            new = point[d] + s
            if new < limits[d,0] or new > limits[d,1]:
                continue
            for ii in range(dimensions):
                adjacent[v,ii] = point[ii]
            adjacent[v, d] = new
    return adjacent
