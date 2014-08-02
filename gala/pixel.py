import os
import sys
import glob
import h5py
import numpy
import json
import shutil
import traceback

import imio, option_manager, app_logger, session_manager, util

def image_stack_verify(options_parser, options, master_logger):
    if options.image_stack is not None:
        if options.image_stack.endswith('.png'):
            images = glob.glob(options.image_stack)
            if len(images) == 0:
                raise Exception("No images found at " + options.image_stack)
        else:            
            if not os.path.exists(options.image_stack):
                raise Exception("Image volume does not exist at " + options.image_stack)

def gen_pixel_verify(options_parser, options, master_logger):
    if options.gen_pixel:
        if options.ilp_file is None:
            raise Exception("Generating pixel probabilities cannot be done without an ILP")
        if "extract-ilp-prediction" in options and not options.extract_ilp_prediction and options.image_stack is None:
            raise Exception("Image volume needs to be supplied to generate pixel probabilities")

def pixelprob_file_verify(options_parser, options, master_logger):
    if options.pixelprob_file is not None:
        if not os.path.exists(options.pixelprob_file):
            raise Exception("Pixel prob file " + options.ilp_file + " not found")

def ilp_file_verify(options_parser, options, master_logger):
    if options.ilp_file is not None:
        if not os.path.exists(options.ilp_file):
            raise Exception("ILP file " + options.ilp_file + " not found")

def temp_dir_verify(options_parser, options, master_logger):
    """
    If a base temporary directory has been specified, make sure it exists or
    can be created.
    """
    if options.temp_dir is not None:
        util.make_dir(options.temp_dir)


def create_pixel_options(options_parser, standalone=True):
    options_parser.create_option("image-stack", "image file(s) (h5 or png format)", 
        required=standalone, verify_fn=image_stack_verify,
         shortcut='I', warning=True) 

    options_parser.create_option("pixelprob-name", "Name for pixel classification", 
        default_val="pixel_boundpred.h5", required=False, dtype=str, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=(not standalone)) 
 
    
    options_parser.create_option("ilp-file", "ILP file containing pixel classifier", 
        default_val=None, required=standalone, dtype=str, verify_fn=ilp_file_verify, num_args=None,
        shortcut=None, warning=False, hidden=False) 
    
    options_parser.create_option("temp-dir", "Path to writable temporary directory", 
        default_val=None, required=standalone, dtype=str, verify_fn=temp_dir_verify, num_args=None,
        shortcut=None, warning=False, hidden=False) 
    
    if not standalone:
        options_parser.create_option("pixelprob-file", "Pixel classification file", 
            default_val=None, required=False, dtype=str, verify_fn=pixelprob_file_verify, num_args=None,
            shortcut=None, warning=False, hidden=True) 
    
        options_parser.create_option("extract-ilp-prediction", "Extract prediction from ILP", 
            default_val=False, required=False, dtype=bool, verify_fn=None, num_args=None,
            shortcut=None, warning=False, hidden=True) 
        
        options_parser.create_option("gen-pixel", "Enable pixel prediction", 
            default_val=False, required=False, dtype=bool, verify_fn=gen_pixel_verify, num_args=None,
            shortcut='GP', warning=True, hidden=False) 


def gen_pixel_probabilities(session_location, options, master_logger, image_filename=None):
    """
    Generates pixel probabilities using classifier in options.ilp_file.

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
 
    if image_filename is None:
        image_filename = options.image_stack

    if "extract-ilp-prediction" in options and options.extract_ilp_prediction:
        master_logger.info("Extract .ilp prediction option has been deprecated")
        sys.exit(2)
    else:
        master_logger.info("Running Ilastik in headless mode")

        pixel_prob_filename = os.path.join(session_location, 'STACKED_prediction.h5')
        ilastik_command = ( "ilastik_headless"
                           #" --headless"
                           " --preconvert_stacks"
                           " --project={project_file}"
                           " --output_axis_order=xyzc" # gala assumes ilastik output is always xyzc
                           " --output_format=hdf5"
                           " --output_filename_format={pixel_prob_filename}"
                           " --output_internal_path=/volume/predictions"
                           "".format( project_file=options.ilp_file,
                                      pixel_prob_filename=pixel_prob_filename ) )
        if options.temp_dir is not None:
            temp_dir = util.make_temp_dir(options.temp_dir)
            ilastik_command += " --sys_tmp_dir={}".format( options.temp_dir )

        # Add the input file as the last arg
        ilastik_command += ' "' + image_filename + '"'
        master_logger.info("Executing ilastik headless command for pixel classification:\n%s" % ilastik_command)
        os.system(ilastik_command)
        if options.temp_dir is not None:
            shutil.rmtree(temp_dir)

        return pixel_prob_filename


def entrypoint(argv):
    applogger = app_logger.AppLogger(False, 'gen-pixel')
    master_logger = applogger.get_logger()
   
    try:
        session = session_manager.Session("gen-pixel", "Pixel classification wrapper for Ilastik", 
            master_logger, applogger, create_pixel_options)    

        gen_pixel_probabilities(session.session_location, session.options, master_logger)
    except Exception, e:
        master_logger.error(str(traceback.format_exc()))
    except KeyboardInterrupt, err:
        master_logger.error(str(traceback.format_exc()))
 
