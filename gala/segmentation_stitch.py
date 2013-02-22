import sys
import os
import argparse
import h5py
import numpy
import shutil
import logging
import json
from skimage import morphology as skmorph
from scipy.ndimage import label
import traceback

from . import imio, morpho, classify, evaluate, app_logger, session_manager, pixel, features, stack_np

# Group where we store predictions in HDF5 file
PREDICTIONS_HDF5_GROUP = '/volume/predictions'


def grab_pred_seg(pred_name, seg_name, border_size):
    prediction = imio.read_image_stack(pred_name,
        group=PREDICTIONS_HDF5_GROUP, single_channel=False)
    segmentation = imio.read_segmentation(seg_name)
    segmentation = segmentation.transpose((2,1,0)) 
    prediction = b1_prediction[border_size:(-1*border_size), border_size:(-1*border_size), border_size:(-1*border_size)]
    segmentation = b1_seg[border_size:(-1*border_size), border_size:(-1*border_size), border_size:(-1*border_size)]
    return prediction, segmentation



def examine_boundary(axis, b1_prediction, b1_seg, b2_prediction, b2_seg,
        b1pt, b2pt, b1pt2, b2pt2, block1, block2, agglom_stack, border_size):
    overlap = False

    dimmin = []
    dimmax = []

    if b1pt[axis] == (b2pt[axis] + 1) or (b1pt[axis] + 1) == b2pt[axis]:
        overlap = True
        for axis2 in range(0,3):
            if axis2 == axis:
                continue
            b1loc1 = b1pt[axis2]
            b1loc2 = b1pt[axis2]
            if b1loc1 > b1loc2:
                b1loc1, b1loc2 = b1loc2, b1loc1

            b2loc1 = b2pt[axis2]
            b2loc2 = b2pt[axis2]
            if b2loc1 > b2loc2:
                b2loc1, b2loc2 = b2loc2, b2loc1
            
            if b1loc1 > b2loc2 or b1loc2 < b2loc1:
                overlap = False
            else:
                dimmin.append(min(b1loc1, b2loc1))
                dimmax.append(min(b1loc2, b2loc2))
                     

    # if face overlaps make 2 part image
    # build RAG from 2 think image (special status for edges)
    if overlap:
        if b1_prediction is None: 
            b1_prediction, b1_seg = grab_pred_seg(block1["prediction-file"], block1["segmentation-file"], border_size)

        if b2_prediction is None: 
            b2_prediction, b2_seg = grab_pred_seg(block2["prediction-file"], block2["segmentation-file"], border_size)

        prediction1 = numpy.zeros((dimmax[0] - dimmin[0] + 1, dimmax[1] - dimmin[1] + 1), 1,
                dtype=b1_prediction.dtype)

        prediction2 = numpy.zeros((dimmax[0] - dimmin[0] + 1, dimmax[1] - dimmin[1] + 1), 1,
                dtype=b1_prediction.dtype)
        supervoxels1 = numpy.zeros((dimmax[0] - dimmin[0] + 1, dimmax[1] - dimmin[1] + 1), 1,
                dtype=b1_seg.dtype)
        supervoxels2 = numpy.zeros((dimmax[0] - dimmin[0] + 1, dimmax[1] - dimmin[1] + 1), 1,
                dtype=b1_seg.dtype)

        # load prediction and supervoxel slices into image
        lower = []
        upper = []
        lowerb = []
        upperb = []
        pos = 0
        for axis2 in range(0,3):
            loc1 = b1pt[axis2]
            loc2 = b1pt2[axis2]
            if loc1 > loc2:
                loc1, loc2 = loc2, loc1
            if axis == axis2:
               lower.append(b1pt[axis] - loc1)
               upper.append(b1pt[axis] - loc1 + 1)
            else: 
               lower.append(0)
               upper.append(loc2 - loc1 + 1)
               lowerb.append(loc1 - dimmin[pos])
               upperb.append(loc2 - dimmin[pos] + 1)
               pos += 1
        prediction1[lowerb[0]:upperb[0],lowerb[1]:upperb[1]] = b1_prediction[lower[0]:upper[0],lower[1]:upper[1],lower[2]:upper[2]]
        supervoxels1[lowerb[0]:upperb[0],lowerb[1]:upperb[1]] = b1_seg[lower[0]:upper[0],lower[1]:upper[1],lower[2]:upper[2]]

        lower = []
        upper = []
        lowerb = []
        upperb = []
        pos = 0
        for axis2 in range(0,3):
            loc1 = b2pt[axis2]
            loc2 = b2pt2[axis2]
            if loc1 > loc2:
                loc1, loc2 = loc2, loc1
            if axis == axis2:
               lower.append(b2pt[axis] - loc1)
               upper.append(b2pt[axis] - loc1 + 1)
            else: 
               lower.append(0)
               upper.append(loc2 - loc1 + 1)
               lowerb.append(loc1 - dimmin[pos])
               upperb.append(loc2 - dimmin[pos] + 1)
               pos += 1
        prediction2[lowerb[0]:upperb[0],lowerb[1]:upperb[1]] = b2_prediction[lower[0]:upper[0],lower[1]:upper[1],lower[2]:upper[2]]
        supervoxels2[lowerb[0]:upperb[0],lowerb[1]:upperb[1]] = b2_seg[lower[0]:upper[0],lower[1]:upper[1],lower[2]:upper[2]]

        # special build mode
        agglom_stack.build_border(supervoxels1, prediction1, supervoxels2, prediction2)

    return overlap, b1_prediction, b1_seg, b2_prediction, b2_seg


def run_stitching(session_location, options, master_logger): 
    # Assumptions
    # 1.  assume global id space (unique ids between partitions)
    # 2.  segmentations will be label volumes and mappings
    # 3.  segmentations are z,y,x and predictions are x,y,z (must transpose)
    # 4.  0,0 is the lower-left corner of the image
    # 5.  assume coordinates in json is x,y,z
        
    
    cl = classify.load_classifier(options.classifier)
    fm_info = json.loads(str(cl.feature_description))

    agglom_stack = stack_np.Stack(None, None, single_channel=False, classifier=cl, feature_info=fm_info,
            synapse_file=None, master_logger=master_logger)


    subvolumes1_json = json.loads(open(options.subvolumes1))
    subvolumes1 = subvolumes_json["subvolumes"]
    
    subvolumes2_json = json.loads(open(options.subvolumes2))
    subvolumes2 = subvolumes_json["subvolumes"]

    for block1 in subvolumes1:
        b1pt1 = block1["near-lower-left"]
        b1pt2 = block1["far-upper-right"]
        b1_prediction = None
        b1_seg = None

        faces = set()

        for block2 in subvolumes2:
            b2pt1 = block2["near-lower-left"]
            b2pt2 = block2["far-upper-right"]

            b2_prediction = None
            b2_seg = None
           
            if "faces" not in block2:
                block2["faces"] = set()

            overlap, b1_prediction, b1_seg, b2_prediction, b2_seg = examine_boundary(0,
                    b1_prediction, b1_seg, b2_prediction, b2_seg, b1pt1, b2pt2, b1pt2,
                    b2pt1, block1, block2, agglom_stack, options.border_size)
            if overlap:
                faces.insert("yz1")

            overlap, b1_prediction, b1_seg, b2_prediction, b2_seg = examine_boundary(1,
                    b1_prediction, b1_seg, b2_prediction, b2_seg, b1pt1, b2pt2,
                    b1pt2, b2pt1, block1, block2, agglom_stack, options.border_size)
            if overlap:
                faces.insert("xz1")

            overlap, b1_prediction, b1_seg, b2_prediction, b2_seg = examine_boundary(2,
                    b1_prediction, b1_seg, b2_prediction, b2_seg, b1pt1, b2pt2,
                    b1pt2, b2pt1, block1, block2, agglom_stack, options.border_size)
            if overlap:
                faces.insert("xy1")

            overlap, b1_prediction, b1_seg, b2_prediction, b2_seg = examine_boundary(0,
                    b1_prediction, b1_seg, b2_prediction, b2_seg, b1pt2, b2pt1,
                    b1pt1, b2pt2, block1, block2, agglom_stack, options.border_size)
            if overlap:
                faces.insert("yz2")

            overlap, b1_prediction, b1_seg, b2_prediction, b2_seg = examine_boundary(1,
                    b1_prediction, b1_seg, b2_prediction, b2_seg, b1pt2, b2pt1,
                    b1pt1, b2pt2, block1, block2, agglom_stack, options.border_size)
            if overlap:
                faces.insert("xz2")

            overlap, b1_prediction, b1_seg, b2_prediction, b2_seg = examine_boundary(2,
                    b1_prediction, b1_seg, b2_prediction, b2_seg, b1pt2, b2pt1,
                    b1pt1, b2pt2, block1, block2, agglom_stack, options.border_size)
            if overlap:
                faces.insert("xy2")
        
        block1["faces"] = faces


    subvolumes = subvolumes1.extend(subvolumes2)

    for subvolume in subvolumes:
        faces = subvolume["faces"]
        pred_master, seg_master = grab_pred_seg(subvolume["prediction-file"], subvolume["segmentation-file"], options.border_size)
   
        if len(faces) > 0:
            if "xy1" in faces:
                pred = pred_master[:,:,0:options.buffer_width] 
                seg = seg_master[:,:,0:options.buffer_width]
                agglom_stack.build_partial(seg, pred)

            if "xy2" in faces:
                pred = pred_master[:,:,(-1*options.buffer_width):] 
                seg = seg_master[:,:,(-1*options.buffer_width):]
                agglom_stack.build_partial(seg, pred)

            if "xz1" in faces:
                pred = pred_master[:,0:options.buffer_width,:] 
                seg = seg_master[:,0:options.buffer_width,:]
                agglom_stack.build_partial(seg, pred)
            
            if "xz2" in faces:
                pred = pred_master[:,(-1*options.buffer_width):,:] 
                seg = seg_master[:,(-1*options.buffer_width):,:]
                agglom_stack.build_partial(seg, pred)

            if "yz1" in faces:
                pred = pred_master[0:options.buffer_width,:,:] 
                seg = seg_master[0:options.buffer_width,:,:]
                agglom_stack.build_partial(seg, pred)
            
            if "yz2" in faces:
                pred = pred_master[(-1*options.buffer_width):,:,:] 
                seg = seg_master[(-1*options.buffer_width):,:,:]
                agglom_stack.build_partial(seg, pred)


    # special merge mode that preserves special nature of border edges and returns all tranformations
    transaction_dict = agglom_stack.agglomerate_border(0.1) 

    # show some stats
    master_logger.info("Number of merges: " + str(len(transactions)))
    
    master_logger.info("Writing graph.json")
    agglom_stack.write_plaza_json(session_location + "/" + graph.json, None)

    # copy volumes to new version
    for subvolume in subvolumes: 
        seg_file = subvolume["segmentation-file"]
        new_seg_file = seg_file.rstrip('.h5')
        new_seg_file = new_seg_file + "_processed.h5"
        subvolume["segmentation-file"] = new_seg_file

        hfile = h5py.File(seg_file, 'r')
        trans = numpy.array(hfile["transforms"])
        stack = numpy.array(hfile["stack"])

        for mapping in trans:
            if mapping[0] in transaction_dict:
                mapping[1] = transaction_dict[mapping[0]]

        hfile_write = h5py.File(new_seg_file, 'w')
        hfile_write.create_dataset("transforms", data=trans)
        hfile_write.create_dataset("stack", data=stack)

    # dump json pointing to new h5's combining both partitions and a copy of everything else
    json_data = {}
    json_data["subvolumes"] = subvolumes
    json_file = open("stitched_volume_0.1.json", 'w')
    json_file.write(json.dumps(subvolumes))

       
def subvolumes_file1_verify(options_parser, options, master_logger):
    if options.subvolumes1:
        if not os.path.exists(options.subvolumes1):
            raise Exception("Synapse file " + options.subvolumes1 + " not found")
        if not options.subvolumes1.endswith('.json'):
            raise Exception("Synapse file " + options.subvolumes1 + " does not end with .json")

def subvolumes_file2_verify(options_parser, options, master_logger):
    if options.subvolumes2:
        if not os.path.exists(options.subvolumes2):
            raise Exception("Synapse file " + options.subvolumes2 + " not found")
        if not options.subvolumes2.endswith('.json'):
            raise Exception("Synapse file " + options.subvolumes2 + " does not end with .json")

def classifier_verify(options_parser, options, master_logger):
    if options.classifier is not None:
        if not os.path.exists(options.classifier):
            raise Exception("Classifier " + options.classifier + " not found")
    # Note -- Classifier could be a variety of extensions (.h5, .joblib, etc) depending
    #  on whether classifier is sklearn or vigra.


def create_stitching_options(options_parser):
    options_parser.create_option("subvolumes1", "JSON file containing an array for each subvolume in the first partition providing info on segmentation, boundary prediction, x-y-z lower-left location, x-y-z upper right",
            default_val=None, required=True, dtype=str, verify_fn=subvolumes_file1_verify, num_args=None, shortcut='s')
    
    options_parser.create_option("subvolumes2", "JSON file containing an array for each subvolume in the second partition providing info on segmentation, boundary prediction, x-y-z lower-left location, x-y-z upper right",
            default_val=None, required=True, dtype=str, verify_fn=subvolumes_file2_verify, num_args=None, shortcut='s')

    options_parser.create_option("classifier", "H5 file containing RF (specific for border or just used in one of the substacks)", 
        default_val=None, required=False, dtype=str, verify_fn=classifier_verify, num_args=None,
        shortcut='k', warning=False, hidden=False) 

    options_parser.create_option("buffer-width", "Width of the stitching region", 
        default_val=50, required=False, dtype=int, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 


    options_parser.create_option("border-size", "Size of the border in pixels of the denormalized cubes", 
        default_val=10, required=False, dtype=int, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 


    options_parser.create_option("segmentation-thresholds", "Segmentation thresholds", 
        default_val=[0.1], required=False, dtype=float, verify_fn=None, num_args='+',
        shortcut='ST', warning=False, hidden=False) 



def entrypoint(argv):
    applogger = app_logger.AppLogger(False, 'seg-stitch')
    master_logger = applogger.get_logger()

    try: 
        session = session_manager.Session("seg-stitch", 
            "Stitches two subvolumes by recomputing the RAG along the border and reports changes to the RAG and mappings of each subvolume -- the subvolumes are not actually concatenated in Raveler space", 
            master_logger, applogger, create_stitching_options)    
        master_logger.info("Session location: " + session.session_location)
        run_stitching(session.session_location, session.options, master_logger) 
    except Exception, e:
        master_logger.error(str(traceback.format_exc()))
    except KeyboardInterrupt, err:
        master_logger.error(str(traceback.format_exc()))
 
   
if __name__ == "__main__":
    sys.exit(main(sys.argv))
