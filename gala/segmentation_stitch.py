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
import hashlib
import math
import re
import datetime

from . import imio, morpho, classify, evaluate, app_logger, session_manager, pixel, features, stack_np

# Group where we store predictions in HDF5 file
PREDICTIONS_HDF5_GROUP = '/volume/predictions'

def grab_extant(blocks, border):
    smallestx = 9999999
    smallesty = 9999999
    smallestz = 9999999
    largestx = 0
    largesty = 0
    largestz = 0

    for block in blocks: 
        pt1 = block["near-lower-left"]
        if pt1[0] < smallestx:
            smallestx = pt1[0]
        if pt1[1] < smallesty:
            smallesty = pt1[1]
        if pt1[2] < smallestz:
            smallestz = pt1[2]

        pt2 = block["far-upper-right"]
        if pt2[0] > largestx:
            largestx = pt2[0]
        if pt2[1] > largesty:
            largesty = pt2[1]
        if pt2[2] > largestz:
            largestz = pt2[2]

    largestx += border
    largesty += border
    largestz += border
    smallestx -= border
    smallesty -= border
    smallestz -= border
    # determine image dimensions
    xsize = largestx - smallestx + 1
    ysize = largesty - smallesty + 1
    zsize = largestz - smallestz + 1

    return largestx, largesty, largestz, smallestx, smallesty, smallestz, xsize, ysize, zsize


def find_close_tbars(subvolumes1, subvolumes2, proximity, border):
    json_data = None
    tbar_hash = set()
    
    subvolumes = list(subvolumes1)
    subvolumes.extend(subvolumes2)
    largestx, largesty, largestz, smallestx, smallesty, smallestz, xsize, ysize, zsize = grab_extant(subvolumes, border) 

    for subvolume in subvolumes1:
        pt1 = subvolume["near-lower-left"]
        startx = pt1[0] - smallestx - border
        starty = pt1[1] - smallesty - border
        startz = pt1[2] - border

        f = h5py.File(subvolume['segmentation-file'], 'r')
        if 'synapse-annotations' in f:
            j_str = f['synapse-annotations'][0]
            block_synapse_data = json.loads(j_str)
            synapse_list = block_synapse_data['data']
            
            if json_data is None:
                json_data = block_synapse_data 
                meta = json_data['metadata']
                meta['session path'] = ''
                meta['date'] = str(datetime.datetime.utcnow())
                meta['computer'] = ''
                json_data['data'] = []
            
            for synapse in synapse_list:
                loc = synapse["T-bar"]["location"]
                x = loc[0] + startx
                y = loc[1] + starty
                z = loc[2] + startz
                tbar_hash.add((x,y,z))

    for subvolume in subvolumes2:
        pt1 = subvolume["near-lower-left"]
        startx = pt1[0] - smallestx - border
        starty = pt1[1] - smallesty - border
        startz = pt1[2] - border

        f = h5py.File(subvolume['segmentation-file'], 'r')
        if 'synapse-annotations' in f:
            j_str = f['synapse-annotations'][0]
            block_synapse_data = json.loads(j_str)
            synapse_list = block_synapse_data['data']
            
            if json_data is None:
                json_data = block_synapse_data 
                meta = json_data['metadata']
                meta['session path'] = ''
                meta['date'] = str(datetime.datetime.utcnow())
                meta['computer'] = ''
                json_data['data'] = []
            
            for synapse in synapse_list:
                loc = synapse["T-bar"]["location"]
                x = loc[0] + startx
                y = loc[1] + starty
                z = loc[2] + startz

                for point in tbar_hash:
                    x2, y2, z2 = point
                    dist = math.sqrt((x2-x)**2 + (y2-y)**2 + (z2-z)**2)
                    if dist <= proximity:
                        synapse["T-bar"]["location"][0] = x
                        synapse["T-bar"]["location"][1] = y
                        synapse["T-bar"]["location"][2] = z

                        for partner in synapse['partners']:
                            loc = partner['location']
                            partner['location'][0] = loc[0] + startx
                            partner['location'][1] = loc[1] + starty
                            partner['location'][2] = loc[2] + startz

                        json_data['data'].append(synapse)          
                        break

    return json_data

def update_filename(name, md5):
    nums = re.findall(r'\d+\.', name)
    val = int(nums[-1].rstrip('.'))
    temp_name = re.sub(r'v\d+\.', str("v" + str(val + 1) + "."), name)
    hashes = re.findall(r'-[0-9a-f]+-',name)
    match_hash = hashes[-1]
    temp_name = re.sub(match_hash, str("-" + md5 + "-"), temp_name)
    return temp_name


def grab_pred_seg(pred_name, seg_name, border_size):
    prediction = imio.read_image_stack(pred_name,
        group=PREDICTIONS_HDF5_GROUP)
    segmentation = imio.read_mapped_segmentation(seg_name)
    segmentation = segmentation.transpose((2,1,0)) 
    prediction = prediction[border_size:(-1*border_size), border_size:(-1*border_size), border_size:(-1*border_size)]
    segmentation = segmentation[border_size:(-1*border_size), border_size:(-1*border_size), border_size:(-1*border_size)]
    return prediction, segmentation



def examine_boundary(axis, b1_prediction, b1_seg, b2_prediction, b2_seg,
        b1pt, b2pt, b1pt2, b2pt2, block1, block2, agglom_stack, border_size, master_logger):
    overlap = False

    dimmin = []
    dimmax = []

    if b1pt[axis] == (b2pt[axis] + 1) or (b1pt[axis] + 1) == b2pt[axis]:
        overlap = True
        for axis2 in range(0,3):
            if axis2 == axis:
                continue
            b1loc1 = b1pt[axis2]
            b1loc2 = b1pt2[axis2]
            if b1loc1 > b1loc2:
                b1loc1, b1loc2 = b1loc2, b1loc1

            b2loc1 = b2pt[axis2]
            b2loc2 = b2pt2[axis2]
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

        (xt, yt, zt, ch) = b1_prediction.shape

        prediction1 = numpy.zeros((dimmax[0] - dimmin[0] + 1, dimmax[1] - dimmin[1] + 1, 1, ch),
                dtype=b1_prediction.dtype)

        prediction2 = numpy.zeros((dimmax[0] - dimmin[0] + 1, dimmax[1] - dimmin[1] + 1, 1, ch),
                dtype=b1_prediction.dtype)
        supervoxels1 = numpy.zeros((dimmax[0] - dimmin[0] + 1, dimmax[1] - dimmin[1] + 1, 1),
                dtype=b1_seg.dtype)
        supervoxels2 = numpy.zeros((dimmax[0] - dimmin[0] + 1, dimmax[1] - dimmin[1] + 1, 1),
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

        master_logger.info("Examining border between " + block1["segmentation-file"] + " and " + block2["segmentation-file"])
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
    
    # prevent stitch if hashes are different
    hashes = re.findall(r'-[0-9a-f]+-',options.subvolumes1)
    match_hash1 = hashes[-1]

    hashes = re.findall(r'-[0-9a-f]+-',options.subvolumes2)
    match_hash2 = hashes[-1]

    if match_hash1 != match_hash2:
        raise Exception("Incompatible segmentations: hashes do not match")

    md5_str = hashlib.md5(' '.join(sys.argv)).hexdigest()

    cl = classify.load_classifier(options.classifier)
    fm_info = json.loads(str(cl.feature_description))


    subvolumes1_json = json.load(open(options.subvolumes1))
    subvolumes1_temp = subvolumes1_json["subvolumes"]
    
    subvolumes2_json = json.load(open(options.subvolumes2))
    subvolumes2_temp = subvolumes2_json["subvolumes"]

    subvolumes1 = []
    subvolumes2 = []
    for seg in subvolumes1_temp:
        if 'config-file' in seg:
            config_file = seg['config-file']
            config_data = json.load(open(config_file))
            seg = config_data["subvolumes"][0]
        subvolumes1.append(seg)
    for seg in subvolumes2_temp:
        if 'config-file' in seg:
            config_file = seg['config-file']
            config_data = json.load(open(config_file))
            seg = config_data["subvolumes"][0]
        subvolumes2.append(seg)

    pred_probe = subvolumes1[0]['prediction-file']
   
    num_channels = 1
    if True: 
        prediction = imio.read_image_stack(pred_probe,
            group=PREDICTIONS_HDF5_GROUP)
        num_channels = prediction.shape[prediction.ndim-1]
    
    master_logger.info("Number of prediction channels: " + str(num_channels))

    agglom_stack = stack_np.Stack(None, None, single_channel=False, classifier=cl, feature_info=fm_info,
                synapse_file=None, master_logger=master_logger, num_channels=num_channels, overlap=True)


    master_logger.info("Examining sub-blocks")
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
                    b2pt1, block1, block2, agglom_stack, options.border_size, master_logger)
            if overlap:
                faces.add("yz1")
                block2["faces"].add("yz2")

            overlap, b1_prediction, b1_seg, b2_prediction, b2_seg = examine_boundary(1,
                    b1_prediction, b1_seg, b2_prediction, b2_seg, b1pt1, b2pt2,
                    b1pt2, b2pt1, block1, block2, agglom_stack, options.border_size, master_logger)
            if overlap:
                faces.add("xz1")
                block2["faces"].add("xz2")

            overlap, b1_prediction, b1_seg, b2_prediction, b2_seg = examine_boundary(2,
                    b1_prediction, b1_seg, b2_prediction, b2_seg, b1pt1, b2pt2,
                    b1pt2, b2pt1, block1, block2, agglom_stack, options.border_size, master_logger)
            if overlap:
                faces.add("xy1")
                block2["faces"].add("xy2")

            overlap, b1_prediction, b1_seg, b2_prediction, b2_seg = examine_boundary(0,
                    b1_prediction, b1_seg, b2_prediction, b2_seg, b1pt2, b2pt1,
                    b1pt1, b2pt2, block1, block2, agglom_stack, options.border_size, master_logger)
            if overlap:
                faces.add("yz2")
                block2["faces"].add("yz1")

            overlap, b1_prediction, b1_seg, b2_prediction, b2_seg = examine_boundary(1,
                    b1_prediction, b1_seg, b2_prediction, b2_seg, b1pt2, b2pt1,
                    b1pt1, b2pt2, block1, block2, agglom_stack, options.border_size, master_logger)
            if overlap:
                faces.add("xz2")
                block2["faces"].add("xz1")

            overlap, b1_prediction, b1_seg, b2_prediction, b2_seg = examine_boundary(2,
                    b1_prediction, b1_seg, b2_prediction, b2_seg, b1pt2, b2pt1,
                    b1pt1, b2pt2, block1, block2, agglom_stack, options.border_size, master_logger)
            if overlap:
                faces.add("xy2")
                block2["faces"].add("xy1")

        block1["faces"] = faces

    # find all synapses that conflict by mapping to a global space and returning json data
    tbar_json = find_close_tbars(subvolumes1, subvolumes2, options.tbar_proximity, options.border_size)
    
    subvolumes1.extend(subvolumes2)
    subvolumes = subvolumes1

    for subvolume in subvolumes:
        faces = subvolume["faces"]
   
        if len(faces) > 0 and options.buffer_width > 1:
            master_logger.info("Examining buffer area: " + subvolume["segmentation-file"])
            pred_master, seg_master = grab_pred_seg(subvolume["prediction-file"], subvolume["segmentation-file"], options.border_size)
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
    master_logger.info("Starting agglomeration to threshold " + str(options.segmentation_threshold)
        + " with " + str(agglom_stack.number_of_nodes()))
    transaction_dict = agglom_stack.agglomerate_border(options.segmentation_threshold) 
    master_logger.info("Finished agglomeration to threshold " + str(options.segmentation_threshold)
        + " with " + str(agglom_stack.number_of_nodes()))


    # output stitched segmentation and update subvolumes accordingly

    if not os.path.exists(session_location+"/seg_data"):
        os.makedirs(session_location+"/seg_data")

    file_base = os.path.abspath(session_location)+"/seg_data/seg-"+str(options.segmentation_threshold) + "-" + md5_str + "-"
    # version is maintained relative to the hash    
    graph_loc = file_base+"graphv1.json"
    tbar_debug_loc = file_base+"synapse-verify.json"

    master_logger.info("Writing graph.json")

    # set threshold value for outputing as appropriate
    agglom_stack.set_overlap_cutoff(5)
    agglom_stack.write_plaza_json(graph_loc, None)

    # write tbar debug file
    jw = open(tbar_debug_loc, 'w')
    jw.write(json.dumps(tbar_json, indent=4))

    json_data = {}
    json_data['graph'] = graph_loc
    json_data['tbar-debug'] = tbar_debug_loc
    json_data['border'] = options.border_size  
    subvolume_configs = []
    subvolume_configs_orig = []


    # load config files into subvolumes
    noconfig = True
    for seg in subvolumes1_temp:
        if 'config-file' in seg:
            subvolume_configs_orig.append(seg['config-file'])
            subvolume_configs.append({'config-file': update_filename(seg['config-file'], md5_str)}) 
            noconfig = False
    if noconfig:
        subvolume_configs_orig.append(options.subvolumes1)
        subvolume_configs.append({'config-file': update_filename(options.subvolumes1, md5_str)})
    
    noconfig = True
    for seg in subvolumes2_temp:
        if 'config-file' in seg:
            subvolume_configs_orig.append(seg['config-file'])
            subvolume_configs.append({'config-file': update_filename(seg['config-file'], md5_str)}) 
            noconfig = False
    if noconfig:
        subvolume_configs_orig.append(options.subvolumes2)
        subvolume_configs.append({'config-file': update_filename(options.subvolumes2, md5_str)})
    json_data['subvolumes'] = subvolume_configs
    
    # write out json file
    json_str = json.dumps(json_data, indent=4)
    json_file = session_location + "/seg-" + str(options.segmentation_threshold) + "-" + md5_str + "-v1.json"
    jw = open(json_file, 'w')
    jw.write(json_str)

    # copy volumes to new version
    for subvolume_file in subvolume_configs_orig:
        config_data = json.load(open(subvolume_file))
        subvolume = config_data["subvolumes"][0] 

        seg_file = subvolume["segmentation-file"]
        new_seg_file = update_filename(seg_file, md5_str)
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
       
        jw = open(update_filename(subvolume_file, md5_str), 'w') 
        jw.write(json.dumps(config_data, indent=4))

       
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
            default_val=None, required=True, dtype=str, verify_fn=subvolumes_file1_verify, num_args=None, shortcut='s1')
    
    options_parser.create_option("subvolumes2", "JSON file containing an array for each subvolume in the second partition providing info on segmentation, boundary prediction, x-y-z lower-left location, x-y-z upper right",
            default_val=None, required=True, dtype=str, verify_fn=subvolumes_file2_verify, num_args=None, shortcut='s2')

    options_parser.create_option("classifier", "H5 file containing RF (specific for border or just used in one of the substacks)", 
        default_val=None, required=False, dtype=str, verify_fn=classifier_verify, num_args=None,
        shortcut='k', warning=False, hidden=False) 

    options_parser.create_option("buffer-width", "Width of the stitching region", 
        default_val=0, required=False, dtype=int, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 


    options_parser.create_option("border-size", "Size of the border in pixels of the denormalized cubes", 
        default_val=10, required=False, dtype=int, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=False) 

    options_parser.create_option("tbar-proximity", "Minimum pixel separation between different tbars in a border region beyond which the tbars get flagged", 
        default_val=10, required=False, dtype=int, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("segmentation-threshold", "Segmentation threshold", 
        default_val=0.1, required=False, dtype=float, verify_fn=None, num_args=None,
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
