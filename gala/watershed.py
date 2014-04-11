import os
import sys
import glob
import h5py
import numpy
import json
import shutil
import traceback
from skimage import morphology as skmorph
from scipy.ndimage import label

import imio, morpho, option_manager, app_logger, session_manager, util

# Group where we store predictions in HDF5 file
PREDICTIONS_HDF5_GROUP = '/volume/predictions'

def ilp_file_verify(options_parser, options, master_logger):
    if options.classifier is not None:
        if not os.path.exists(options.classifier):
            raise Exception("ILP file " + options.classifier + " not found")

def temp_dir_verify(options_parser, options, master_logger):
    """
    If a base temporary directory has been specified, make sure it exists or
    can be created.
    """
    if options.temp_dir is not None:
        util.make_dir(options.temp_dir)

def create_watershed_options(options_parser):
    options_parser.create_option("datasrc", "location of datatype on DVID", 
        default_val=None, required=True, dtype=str, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=False) 

    options_parser.create_option("classifier", "ILP file containing pixel classifier", 
        default_val=None, required=True, dtype=str, verify_fn=ilp_file_verify, num_args=None,
        shortcut=None, warning=False, hidden=False) 

    options_parser.create_option("border", "Border surrounding dataset", 
        default_val=None, required=True, dtype=int, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=False) 
 
    options_parser.create_option("bbox1", "Bounding box first coordinate", 
        default_val=None, required=True, dtype=int, verify_fn=None, num_args='+',
        shortcut=None, warning=False, hidden=False) 
    
    options_parser.create_option("bbox2", "Bounding box second coordinate", 
        default_val=None, required=True, dtype=int, verify_fn=None, num_args='+',
        shortcut=None, warning=False, hidden=False) 

    options_parser.create_option("temp-dir", "Path to writable temporary directory", 
        default_val=None, required=False, dtype=str, verify_fn=temp_dir_verify, num_args=None,
        shortcut=None, warning=False, hidden=False) 

    options_parser.create_option("seed-size", "Minimum size of seeded region", 
        default_val=5, required=False, dtype=int, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=False) 

    options_parser.create_option("pixelprob-name", "Name for pixel classification", 
        default_val="pixel_boundpred.h5", required=False, dtype=str, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=False) 

def create_labels(border_size, prediction_file, options, master_logger):
    """Returns ndarray labeled using watershed algorithm

    Args:
        options:  OptionNamespace.
        prediction_file:  String.  File name of prediction hdf5 file where predictions
            are assumed to be in group PREDICTIONS_HDF5_GROUP.

    Returns:
        A 2-tuple of supervoxel.
    """
    master_logger.debug("Generating supervoxels")
    if not os.path.isfile(prediction_file):
        raise Exception("Training file not found: " + prediction_file)

    prediction = imio.read_image_stack(prediction_file, group=PREDICTIONS_HDF5_GROUP)
    master_logger.info("Transposed boundary prediction")
    prediction = prediction.transpose((2, 1, 0, 3))

    #if options.extract_ilp_prediction:
    #   prediction = prediction.transpose((2, 1, 0))

    # TODO -- Refactor.  If 'single-channel' and hdf5 prediction file is given, it looks like
    #   read_image_stack will return a modified volume and the bound-channels parameter must
    #   be 0 or there'll be conflict.

    master_logger.debug("Grabbing boundary labels: " + str([0]))
    boundary = prediction[...,0] 

    master_logger.info("Shape of boundary: %s" % str(boundary.shape))

    # Prediction file is in format (t, x, y, z, c) but needs to be in format (z, x, y).
    # Also, raveler convention is (0,0) sits in bottom left while ilastik convention is
    # origin sits in top left.
    # imio.read_image_stack squeezes out the first dim.

    seeds = label(boundary==0)[0]

    if options.seed_size > 0:
        master_logger.debug("Removing small seeds")
        seeds = morpho.remove_small_connected_components(seeds, options.seed_size)
        master_logger.debug("Finished removing small seeds")

    master_logger.info("Starting watershed")
    
    boundary_cropped = boundary
    seeds_cropped = seeds 
    if border_size > 0:
        boundary_cropped = boundary[border_size:(-1*border_size), border_size:(-1*border_size),border_size:(-1*border_size)]
        seeds_cropped = label(boundary_cropped==0)[0]
        if options.seed_size > 0:
            seeds_cropped = morpho.remove_small_connected_components(seeds_cropped, options.seed_size)

    supervoxels_cropped = skmorph.watershed(boundary_cropped, seeds_cropped)
    
    supervoxels = supervoxels_cropped
    if border_size > 0:
        supervoxels = seeds.copy()
        supervoxels.dtype = supervoxels_cropped.dtype
        supervoxels[:,:,:] = 0 
        supervoxels[border_size:(-1*border_size), 
                border_size:(-1*border_size),border_size:(-1*border_size)] = supervoxels_cropped

    master_logger.info("Finished watershed")
   
    return supervoxels


def gen_watershed(session_location, options, master_logger, image_filename=None):
    """
    Generates pixel probabilities using classifier in options.clasifier.

    Args:
        session_location:  String.  Where we should export generated pixel probabilities.
        options:  OptionNamespace.
        image_filename:  String.  Input image file name.  If given, overrides image-stack
            key in options.

    Returns:
        Filename of pixel probabilities

    Side-effects:
        Generates hdf5 file of pixel probabilities in session_location directory.
        File will be named 'STACKED_prediction.h5' and probabilities will be in
        hdf group /volume/predictions
    """

    master_logger.info("Generating Pixel Probabilities") 

    # do not add to the border size
    border2 = options.border
    coord0 = options.bbox1[2] - border2, options.bbox1[1] - border2, options.bbox1[0] - border2, 0
    # hard code output channel to 6 for now
    coord1 = options.bbox2[2] + border2, options.bbox2[1] + border2, options.bbox2[0] + border2, 6
    coords = [coord0, coord1]

    master_logger.info("Running Ilastik in headless mode")
    pixel_prob_filename = os.path.join(session_location, 'STACKED_prediction.h5')
    ilastik_command = ( "ilastik_headless"
                       #" --headless"
                       ' --cutout_subregion="{coords}"'
                       " --preconvert_stacks"
                       " --project={project_file}"
                       " --output_axis_order=txyzc" # gala assumes ilastik output is always txyzc
                       " --output_format=hdf5"
                       " --output_filename_format={pixel_prob_filename}"
                       " --output_internal_path=/volume/predictions"
                       "".format( coords=coords, project_file=options.classifier,
                                  pixel_prob_filename=pixel_prob_filename ) )
    if options.temp_dir is not None:
        temp_dir = util.make_temp_dir(options.temp_dir)
        ilastik_command += " --sys_tmp_dir={}".format( options.temp_dir )

    # Add the input file as the last arg
    ilastik_command += ' "dvid://' + options.datasrc + '"'
    master_logger.info("Executing ilastik headless command for pixel classification:\n%s" % ilastik_command)
   
    """
    import subprocess
    p = subprocess.Popen(ilastik_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    master_logger.info("before!!!")
    output, errors = p.communicate()

    master_logger.info(output)
    master_logger.info(errors)
    """

    os.system(ilastik_command)
    if options.temp_dir is not None:
        shutil.rmtree(temp_dir)

    labels = create_labels(0, pixel_prob_filename, options, master_logger)
   
    imio.write_image_stack(labels,
        session_location + "/" + "supervoxels.h5")

    fout = open(session_location + "/max_body.json", 'w')
    fout.write(json.dumps({ "max_id": int(labels.max()) }, indent=4))

def entrypoint(argv):
    applogger = app_logger.AppLogger(False, 'gen-watershed')
    master_logger = applogger.get_logger()

    try:
        session = session_manager.Session("gen-watershed", "Watershed generation based on Ilastik boundary prediction", 
            master_logger, applogger, create_watershed_options)    

        gen_watershed(session.session_location, session.options, master_logger)
    except Exception, e:
        master_logger.error(str(traceback.format_exc()))
    except KeyboardInterrupt, err:
        master_logger.error(str(traceback.format_exc()))
 
