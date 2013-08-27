import sys
import os
import argparse
import h5py
import numpy
from numpy import array, uint8, uint16, uint32, uint64, zeros, \
    zeros_like, squeeze, fromstring, ndim, concatenate, newaxis, swapaxes, \
    savetxt, unique, double, cumsum, ndarray
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


def find_close_tbars(regions, proximity, border):
    json_data = None
    tbar_hash = set()
    
    blocks = None
    for region in regions:
        if blocks is None:
            blocks = list(region)
        else:
            blocks.extend(region)

    largestx, largesty, largestz, smallestx, smallesty, smallestz, xsize, ysize, zsize = grab_extant(blocks, border) 

    for region in regions:
        tbar_hash_temp = set()
        for block in region:
            pt1 = block["near-lower-left"]
            pt2 = block["far-upper-right"]
            startx = pt1[0] - smallestx - border
            starty = largesty - pt2[1] - border
            startz = pt1[2] - border

            f = h5py.File(block['segmentation-file'], 'r')
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
                    tbar_hash_temp.add((x,y,z))

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

        tbar_hash = tbar_hash.union(tbar_hash_temp)

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
    if border_size > 0: 
        prediction = prediction[border_size:(-1*border_size), border_size:(-1*border_size), border_size:(-1*border_size)]
        segmentation = segmentation[border_size:(-1*border_size), border_size:(-1*border_size), border_size:(-1*border_size)]
    return prediction, segmentation



def examine_boundary(axis, b1_prediction, b1_seg, b2_prediction, b2_seg,
        b1pt, b2pt, b1pt2, b2pt2, block1, block2, agglom_stack, border_size, master_logger, options,
        all_bodies, disjoint_face_bodies, already_examined1, already_examined2):
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
                dimmax.append(max(b1loc2, b2loc2))
                     

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

        prediction_vol = numpy.zeros((dimmax[0] - dimmin[0] + 1,
                dimmax[1] - dimmin[1] + 1, 50+50, ch), dtype=b1_prediction.dtype)

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
        firstblock_first = False
        for axis2 in range(0,3):
            loc1 = b1pt[axis2]
            loc2 = b1pt2[axis2]
            if loc1 > loc2:
                firstblock_first = True
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
       
        b1_prediction_temp = b1_prediction[lower[0]:upper[0],lower[1]:upper[1],lower[2]:upper[2]]
        b1_seg_temp = b1_seg[lower[0]:upper[0],lower[1]:upper[1],lower[2]:upper[2]]

        prediction_vol_temp = None

        if axis == 0:
            b1_prediction_temp = b1_prediction_temp.transpose((1,2,0,3))
            b1_seg_temp = b1_seg_temp.transpose((1,2,0))
            if lower[0] == 0:
                prediction_vol_temp = b1_prediction[lower[0]:50,lower[1]:upper[1],lower[2]:upper[2]]
            else:
                prediction_vol_temp = b1_prediction[upper[0]-50:upper[0],lower[1]:upper[1],lower[2]:upper[2]]
            prediction_vol_temp = prediction_vol_temp.transpose((1,2,0,3))
        if axis == 1:
            b1_prediction_temp = b1_prediction_temp.transpose((0,2,1,3))
            b1_seg_temp = b1_seg_temp.transpose((0,2,1))
            if lower[1] == 0:
                prediction_vol_temp = b1_prediction[lower[0]:upper[0],lower[1]:50,lower[2]:upper[2]]
            else:
                prediction_vol_temp = b1_prediction[lower[0]:upper[0],upper[1]-50:upper[1],lower[2]:upper[2]]
            prediction_vol_temp = prediction_vol_temp.transpose((0,2,1,3))
        if axis == 2:
            if lower[2] == 0:
                prediction_vol_temp = b1_prediction[lower[0]:upper[0],lower[1]:upper[1],lower[2]:50]
            else:
                prediction_vol_temp = b1_prediction[lower[0]:upper[0],lower[1]:upper[1],upper[2]-50:upper[2]]

        if firstblock_first:
            prediction_vol[lowerb[0]:upperb[0],lowerb[1]:upperb[1],0:50] = prediction_vol_temp
        else:
            prediction_vol[lowerb[0]:upperb[0],lowerb[1]:upperb[1],50:100] = prediction_vol_temp

        prediction1[lowerb[0]:upperb[0],lowerb[1]:upperb[1]] = b1_prediction_temp 
        supervoxels1[lowerb[0]:upperb[0],lowerb[1]:upperb[1]] = b1_seg_temp 

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
        
        b2_prediction_temp = b2_prediction[lower[0]:upper[0],lower[1]:upper[1],lower[2]:upper[2]]
        b2_seg_temp = b2_seg[lower[0]:upper[0],lower[1]:upper[1],lower[2]:upper[2]]

        if axis == 0:
            b2_prediction_temp = b2_prediction_temp.transpose((1,2,0,3))
            b2_seg_temp = b2_seg_temp.transpose((1,2,0))
            if lower[0] == 0:
                prediction_vol_temp = b2_prediction[lower[0]:50,lower[1]:upper[1],lower[2]:upper[2]]
            else:
                prediction_vol_temp = b2_prediction[upper[0]-50:upper[0],lower[1]:upper[1],lower[2]:upper[2]]
            prediction_vol_temp = prediction_vol_temp.transpose((1,2,0,3))
        if axis == 1:
            b2_prediction_temp = b2_prediction_temp.transpose((0,2,1,3))
            b2_seg_temp = b2_seg_temp.transpose((0,2,1))
            if lower[1] == 0:
                prediction_vol_temp = b2_prediction[lower[0]:upper[0],lower[1]:50,lower[2]:upper[2]]
            else:
                prediction_vol_temp = b2_prediction[lower[0]:upper[0],upper[1]-50:upper[1],lower[2]:upper[2]]
            prediction_vol_temp = prediction_vol_temp.transpose((0,2,1,3))
        if axis == 2:
            if lower[2] == 0:
                prediction_vol_temp = b2_prediction[lower[0]:upper[0],lower[1]:upper[1],lower[2]:50]
            else:
                prediction_vol_temp = b2_prediction[lower[0]:upper[0],lower[1]:upper[1],upper[2]-50:upper[2]]
        
        if not firstblock_first:
            prediction_vol[lowerb[0]:upperb[0],lowerb[1]:upperb[1],0:50] = prediction_vol_temp
        else:
            prediction_vol[lowerb[0]:upperb[0],lowerb[1]:upperb[1],50:100] = prediction_vol_temp
        
        prediction2[lowerb[0]:upperb[0],lowerb[1]:upperb[1]] = b2_prediction_temp 
        supervoxels2[lowerb[0]:upperb[0],lowerb[1]:upperb[1]] = b2_seg_temp 

        master_logger.info("Examining border between " + block1["segmentation-file"] + " and " + block2["segmentation-file"])
       
        mask1 = None
        mask2 = None

        if options.run_watershed:
            # generate watershed as in gala main flow over thick boundary stuff
            master_logger.info("Generating watershed in boundary region")
            boundary_vol = prediction_vol[...,0]
            seeds = label(boundary_vol==0)[0]
            seeds = morpho.remove_small_connected_components(seeds, 5)
            supervoxels = skmorph.watershed(boundary_vol, seeds)
            master_logger.info("Finished generating watershed in boundary region")
            
            # generate thick boundary in neuroproof and return volume
            supervoxels = agglom_stack.dilate_edges(supervoxels) 

            # grab inner 2 slices and mask supervoxels (ignore if one 0) 
            # OR increase edge size to 0 when one is 0
            mask1 = supervoxels[:,:,49:50]
            mask2 = supervoxels[:,:,50:51]

            if not firstblock_first:
                mask2 = supervoxels[:,:,49:50]
                mask1 = supervoxels[:,:,50:51]

        # special build mode
        agglom_stack.build_border(supervoxels1, prediction1, supervoxels2,
                prediction2, mask1, mask2, not options.vertical_mode)

        # load disjoint block face and disjoint bodies
        body_list1 = []
        body_list2 = []
   
        if not already_examined1: 
            body_list1 = unique(supervoxels1)
        if not already_examined2: 
            body_list2 = unique(supervoxels2)
        body_list = numpy.append(body_list1, body_list2) 

        master_logger.info("Finding disjoint bodies on one face")
        if not already_examined1: 
            supervoxels1 = agglom_stack.dilate_edges(supervoxels1) 
        if not already_examined2: 
            supervoxels2 = agglom_stack.dilate_edges(supervoxels2) 

        # run cc on supervoxels
        def load_disjoint_bodies(supervoxels0s, disjoint_bodies):
            supervoxels_sep, num_ccs = label(supervoxels0s)
            bodies_found = set()

            # find one location for each cc and add to bodies found
            for cc_id in range(1,num_ccs+1):
                loc1, loc2, dummy = numpy.where(supervoxels_sep == cc_id)
                # hack to deal with small disjoint bodies created
                # by inadvertently pinching off small non-disjoint pieces
                if len(loc1) < 5:
                    continue
                loc1 = loc1[0]
                loc2 = loc2[0]
                corresponding_body = supervoxels0s[loc1, loc2, 0]
                if corresponding_body in bodies_found:
                    disjoint_bodies.add(corresponding_body)
                else:
                    bodies_found.add(corresponding_body)

        if not already_examined1: 
            load_disjoint_bodies(supervoxels1, disjoint_face_bodies)
        if not already_examined2: 
            load_disjoint_bodies(supervoxels2, disjoint_face_bodies)

        master_logger.info("Finding bodies on multiple faces")
        # see if body has already been added
        for body in body_list:
            if body in all_bodies:
                disjoint_face_bodies.add(body)
            else:
                all_bodies.add(body)

    return overlap, b1_prediction, b1_seg, b2_prediction, b2_seg


def run_stitching(session_location, options, master_logger): 
    # Assumptions
    # 1.  assume global id space (unique ids between partitions)
    # 2.  segmentations will be label volumes and mappings
    # 3.  segmentations are z,y,x and predictions are x,y,z (must transpose)
    # 4.  0,0 is the lower-left corner of the image
    # 5.  assume coordinates in json is x,y,z
    
    # prevent stitch if hashes are different

    match_hash = None
    for region in options.regions:
        hashes = re.findall(r'-[0-9a-f]+-',region)
        match_hash_temp = hashes[-1]
        if match_hash is not None and match_hash_temp != match_hash:
            raise Exception("Incompatible segmentations: hashes do not match")
        match_hash = match_hash_temp

    md5_str = hashlib.md5(' '.join(sys.argv)).hexdigest()

    cl = classify.load_classifier(options.classifier)
    fm_info = json.loads(str(cl.feature_description))


    border_size = None
    regions_blocks = []
    regions_blocks_temp = []
    for region in options.regions:
        blocks = []

        region_json = json.load(open(region))
        blocks_temp = region_json["subvolumes"]
        regions_blocks_temp.append(blocks_temp)

        border_size_temp = region_json["border"]
        if border_size is not None and border_size != border_size_temp:
            raise Exception("border attrubute not the same in all regions") 
        border_size = border_size_temp

        for block in blocks_temp:
            if 'config-file' in block:
                config_file = block['config-file']
                config_data = json.load(open(config_file))
                block = config_data["subvolumes"][0]
            blocks.append(block)
        regions_blocks.append(blocks)

    pred_probe = regions_blocks[0][0]['prediction-file']
   
    num_channels = 1
    if True: 
        prediction = imio.read_image_stack(pred_probe,
            group=PREDICTIONS_HDF5_GROUP)
        num_channels = prediction.shape[prediction.ndim-1]
    
    master_logger.info("Number of prediction channels: " + str(num_channels))

    agglom_stack = stack_np.Stack(None, None, single_channel=False, classifier=cl, feature_info=fm_info,
                synapse_file=None, master_logger=master_logger, num_channels=num_channels, overlap=True)
    agglom_stack.set_overlap_cutoff(options.overlap_threshold)
    agglom_stack.set_border_weight(options.border_weight_factor)

    if options.aggressive_stitch:
        # use the maximum overlap between the two regions as the merge criterion
        agglom_stack.set_overlap_max()
    else:
        # use the minimum overlap between the two regions as the merge criterion
        agglom_stack.set_overlap_min()

    all_bodies = set()
    disjoint_face_bodies = set()

    def already_examined(faces, face_name):
        if face_name in faces:
            return True
        else:
            return False

    master_logger.info("Examining sub-blocks")
    for iter1 in range(0, len(regions_blocks)):
        region1 = regions_blocks[iter1]
        for block1 in region1:
            b1pt1 = block1["near-lower-left"]
            b1pt2 = block1["far-upper-right"]
            b1_prediction = None
            b1_seg = None

            faces = set()

            for iter2 in range(iter1+1, len(regions_blocks)):
                region2 = regions_blocks[iter2]

                for block2 in region2:
                    b2pt1 = block2["near-lower-left"]
                    b2pt2 = block2["far-upper-right"]

                    b2_prediction = None
                    b2_seg = None
                   
                    if "faces" not in block2:
                        block2["faces"] = set()

                    already_examined1 = already_examined(faces, "yz1")
                    already_examined2 = already_examined(block2["faces"], "yz2")
                    overlap, b1_prediction, b1_seg, b2_prediction, b2_seg = examine_boundary(0,
                            b1_prediction, b1_seg, b2_prediction, b2_seg, b1pt1, b2pt2, b1pt2,
                            b2pt1, block1, block2, agglom_stack, border_size, master_logger, options,
                            all_bodies, disjoint_face_bodies, already_examined1, already_examined2)
                    if overlap:
                        faces.add("yz1")
                        block2["faces"].add("yz2")

                    already_examined1 = already_examined(faces, "xz1")
                    already_examined2 = already_examined(block2["faces"], "xz2")
                    overlap, b1_prediction, b1_seg, b2_prediction, b2_seg = examine_boundary(1,
                            b1_prediction, b1_seg, b2_prediction, b2_seg, b1pt1, b2pt2,
                            b1pt2, b2pt1, block1, block2, agglom_stack, border_size, master_logger, options,
                            all_bodies, disjoint_face_bodies, already_examined1, already_examined2)
                    if overlap:
                        faces.add("xz1")
                        block2["faces"].add("xz2")

                    already_examined1 = already_examined(faces, "xy1")
                    already_examined2 = already_examined(block2["faces"], "xy2")
                    overlap, b1_prediction, b1_seg, b2_prediction, b2_seg = examine_boundary(2,
                            b1_prediction, b1_seg, b2_prediction, b2_seg, b1pt1, b2pt2,
                            b1pt2, b2pt1, block1, block2, agglom_stack, border_size, master_logger, options,
                            all_bodies, disjoint_face_bodies, already_examined1, already_examined2)
                    if overlap:
                        faces.add("xy1")
                        block2["faces"].add("xy2")

                    already_examined1 = already_examined(faces, "yz2")
                    already_examined2 = already_examined(block2["faces"], "yz1")
                    overlap, b1_prediction, b1_seg, b2_prediction, b2_seg = examine_boundary(0,
                            b1_prediction, b1_seg, b2_prediction, b2_seg, b1pt2, b2pt1,
                            b1pt1, b2pt2, block1, block2, agglom_stack, border_size, master_logger, options,
                            all_bodies, disjoint_face_bodies, already_examined1, already_examined2)
                    if overlap:
                        faces.add("yz2")
                        block2["faces"].add("yz1")

                    already_examined1 = already_examined(faces, "xz2")
                    already_examined2 = already_examined(block2["faces"], "xz1")
                    overlap, b1_prediction, b1_seg, b2_prediction, b2_seg = examine_boundary(1,
                            b1_prediction, b1_seg, b2_prediction, b2_seg, b1pt2, b2pt1,
                            b1pt1, b2pt2, block1, block2, agglom_stack, border_size, master_logger, options,
                            all_bodies, disjoint_face_bodies, already_examined1, already_examined2)
                    if overlap:
                        faces.add("xz2")
                        block2["faces"].add("xz1")

                    already_examined1 = already_examined(faces, "xy2")
                    already_examined2 = already_examined(block2["faces"], "xy1")
                    overlap, b1_prediction, b1_seg, b2_prediction, b2_seg = examine_boundary(2,
                            b1_prediction, b1_seg, b2_prediction, b2_seg, b1pt2, b2pt1,
                            b1pt1, b2pt2, block1, block2, agglom_stack, border_size, master_logger, options,
                            all_bodies, disjoint_face_bodies, already_examined1, already_examined2)
                    if overlap:
                        faces.add("xy2")
                        block2["faces"].add("xy1")

            block1["faces"] = faces

    # find all synapses that conflict by mapping to a global space and returning json data
    tbar_json = find_close_tbars(regions_blocks, options.tbar_proximity, border_size)
   
    blocks = None
    for region in regions_blocks:
        if blocks is None:
            blocks = list(region)
        else:
            blocks.extend(region)

    for block in blocks:
        faces = block["faces"]
   
        if len(faces) > 0 and options.buffer_width > 1:
            master_logger.info("Examining buffer area: " + block["segmentation-file"])
            pred_master, seg_master = grab_pred_seg(block["prediction-file"], block["segmentation-file"], border_size)
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


    # output stitched segmentation and update blocks accordingly

    if not os.path.exists(session_location+"/seg_data"):
        os.makedirs(session_location+"/seg_data")

    file_base = os.path.abspath(session_location)+"/seg_data/seg-"+str(options.segmentation_threshold) + "-" + md5_str + "-"
    # version is maintained relative to the hash    
    graph_loc = file_base+"graphv1.json"
    tbar_debug_loc = file_base+"synapse-verify.json"
    body_annotations_fn = file_base+"annotations-body.json"
   
    disjoint_face_bodies_mapped = set()
    for bodyid in disjoint_face_bodies:
        if bodyid in transaction_dict:
            disjoint_face_bodies_mapped.add(transaction_dict[bodyid])
        else:
            disjoint_face_bodies_mapped.add(bodyid)
    
    body_list = []
    for bodyid in disjoint_face_bodies_mapped:
         body_list.append({"status" : "uncorrected",
                          "body ID" : int(bodyid),
                          "anchor" : "anchor"})
    
    body_data = {}
    body_data["data"] = body_list
    body_data["metadata"] = {
                        "username" : "auto",
                        "description" : "anchor annotations",
                        "file version" : 3,
                        "software" : "Raveler"
                        }

    master_logger.info("Writing graph.json")

    # set threshold value for outputing as appropriate (edge size * 2)
    agglom_stack.set_overlap_cutoff(0)
    # use the maximum overlap between the two regions as the the split criterion (proofread 0-0.7)
    agglom_stack.set_overlap_max()
    agglom_stack.set_saved_probs()
    agglom_stack.write_plaza_json(graph_loc, None, 0, True)

    # write tbar debug file
    jw = open(tbar_debug_loc, 'w')
    jw.write(json.dumps(tbar_json, indent=4))

    # write body annotation file
    jw = open(body_annotations_fn, 'w')
    jw.write(json.dumps(body_data, indent=4))

    json_data = {}
    json_data['graph'] = graph_loc
    json_data['tbar-debug'] = tbar_debug_loc
    json_data['annotations-body'] = body_annotations_fn
    json_data['border'] = border_size  
    block_configs = []
    block_configs_orig = []


    # load config files into subvolumes
    for iter1 in range(0, len(regions_blocks_temp)):
        region = regions_blocks_temp[iter1]
        noconfig = True
        for block in region:
            if 'config-file' in block:
                block_configs_orig.append(block['config-file'])
                block_configs.append({'config-file': update_filename(block['config-file'], md5_str)}) 
                noconfig = False
        if noconfig:
            block_configs_orig.append(options.regions[iter1])
            block_configs.append({'config-file': update_filename(options.regions[iter1], md5_str)})
    
    json_data['subvolumes'] = block_configs
    
    # write out json file
    json_str = json.dumps(json_data, indent=4)
    json_file = session_location + "/seg-" + str(options.segmentation_threshold) + "-" + md5_str + "-v1.json"
    jw = open(json_file, 'w')
    jw.write(json_str)

    # copy volumes to new version
    for block_file in block_configs_orig:
        config_data = json.load(open(block_file))

        if 'log' not in config_data:
            config_data['log'] = []

        config_data['log'].append(str(datetime.datetime.utcnow()) + " " + (' '.join(sys.argv)))

        block = config_data["subvolumes"][0] 

        seg_file = block["segmentation-file"]
        new_seg_file = update_filename(seg_file, md5_str)
        block["segmentation-file"] = new_seg_file

        hfile = h5py.File(seg_file, 'r')
        trans = numpy.array(hfile["transforms"])
        stack = numpy.array(hfile["stack"])

        for mapping in trans:
            if mapping[1] in transaction_dict:
                mapping[1] = transaction_dict[mapping[1]]

        hfile_write = h5py.File(new_seg_file, 'w')
        hfile_write.create_dataset("transforms", data=trans)
        hfile_write.create_dataset("stack", data=stack)
        
        if 'synapse-annotations' in hfile:
            hfile_write.copy(hfile['synapse-annotations'], 'synapse-annotations') 
        if 'bookmark-annotations' in hfile:
            hfile_write.copy(hfile['bookmark-annotations'], 'bookmark-annotations') 
       
        jw = open(update_filename(block_file, md5_str), 'w') 
        jw.write(json.dumps(config_data, indent=4))

       
def regions_file_verify(options_parser, options, master_logger):
    if options.regions:
        for region in options.regions:        
            if not os.path.exists(region):
                raise Exception("Region file " + region + " not found")
            if not region.endswith('.json'):
                raise Exception("Region file " + region + " does not end with .json")

def classifier_verify(options_parser, options, master_logger):
    if options.classifier is not None:
        if not os.path.exists(options.classifier):
            raise Exception("Classifier " + options.classifier + " not found")
    # Note -- Classifier could be a variety of extensions (.h5, .joblib, etc) depending
    #  on whether classifier is sklearn or vigra.


def create_stitching_options(options_parser):
    options_parser.create_option("regions", "JSON files corresponding to disjoint region that contain an array for each block providing info on segmentation, boundary prediction, x-y-z lower-left location, x-y-z upper right",
            default_val=None, required=True, dtype=str, verify_fn=regions_file_verify, num_args='+', shortcut='r')
    
    options_parser.create_option("classifier", "H5 file containing RF (specific for border or just used in one of the substacks)", 
        default_val=None, required=True, dtype=str, verify_fn=classifier_verify, num_args=None,
        shortcut='k', warning=False, hidden=False) 

    options_parser.create_option("aggressive-stitch", "More aggressively stitch segments to reduce remaining work",
            default_val=False, required=False, dtype=bool, num_args=None, warning=False, hidden=False)
    
    options_parser.create_option("vertical-mode", "Enables special handling of bodies across horizontally stitched substacks when vertically stitching",
            default_val=False, required=False, dtype=bool, num_args=None, warning=False, hidden=False)
    
    options_parser.create_option("run-watershed", "Generate a watershed to estimate potential edges",
            default_val=False, required=False, dtype=bool, num_args=None, warning=False, hidden=False)

    options_parser.create_option("segmentation-threshold", "Segmentation threshold", 
        default_val=0.3, required=False, dtype=float, verify_fn=None, num_args=None,
        shortcut='ST', warning=False, hidden=False) 

    options_parser.create_option("border-weight-factor", "Weight to give pixels likely on a boundary -- 0 is no weight", 
        default_val=1.0, required=False, dtype=float, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=False) 

    options_parser.create_option("overlap-threshold", "Minimum size of overlap considered for stitching", 
        default_val=0, required=False, dtype=int, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=False) 

    options_parser.create_option("tbar-proximity", "Minimum pixel separation between different tbars in a border region beyond which the tbars get flagged", 
        default_val=10, required=False, dtype=int, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=False) 

    options_parser.create_option("buffer-width", "Width of the stitching region", 
        default_val=0, required=False, dtype=int, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 
    

    options_parser.create_option("border-size", "DEPRECATED: Size of the border in pixels of the denormalized cubes", 
        default_val=10, required=False, dtype=int, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 




def entrypoint(argv):
    applogger = app_logger.AppLogger(False, 'seg-stitch')
    master_logger = applogger.get_logger()

    try: 
        session = session_manager.Session("seg-stitch", 
            "Stitches multiple regions by recomputing the RAG along the border and reports changes to the RAG and mappings of each block -- the regions are not actually concatenated in Raveler space", 
            master_logger, applogger, create_stitching_options)    
        master_logger.info("Session location: " + session.session_location)
        run_stitching(session.session_location, session.options, master_logger) 
    except Exception, e:
        master_logger.error(str(traceback.format_exc()))
    except KeyboardInterrupt, err:
        master_logger.error(str(traceback.format_exc()))
 
   
if __name__ == "__main__":
    sys.exit(main(sys.argv))
