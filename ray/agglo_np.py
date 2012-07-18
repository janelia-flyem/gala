# built-ins
import libNeuroProofRag as neuroproof
import morpho

from numpy import zeros_like, array, double, zeros
import numpy

class Stack:
    """Region adjacency graph for segmentation of nD volumes."""

    def __init__(self, watershed=numpy.array([]), probabilities=numpy.array([])) : 
        """Create a graph from a watershed volume and image volume.
        
        """

        self.depth, self.height, self.width = watershed.shape


        self.watershed = morpho.pad(watershed, 0)
        self.watershed = self.watershed.astype(numpy.double)    

        probs = probabilities.astype(numpy.double)
        self.probabilities = morpho.pad(probs, 0)
    
        self.stack = neuroproof.build_stack(self.watershed, self.probabilities)
        self.stack.build_rag()

    def number_of_nodes(self):
        return self.stack.get_num_bodies()

    def __copy__(self):
        raise Exception("Not implemented yet")

    def copy(self):
        raise Exception("Not implemented yet")

    def agglomerate(self, threshold=0.5):
        self.stack.agglomerate_rag(threshold) 

    # already complete when the internal values are set to watershed and 1 
    def get_segmentation(self):
        seg_buffer = numpy.zeros((self.depth, self.height, self.width), numpy.uint32)
        neuroproof.write_volume_to_buffer(self.stack, seg_buffer)
        return seg_buffer

    def remove_inclusions(self):
        raise Exception("Not implemented yet")

    # just a simple rag export -- need feature for max size in a specified dimension
    def write_plaza_json(self, fout):
        raise Exception("Not implemented yet")

    def learn_agglomerate(self, gts, feature_map, min_num_samples=1,
                                *args, **kwargs):
        raise Exception("Not implemented yet")



