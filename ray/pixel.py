import imio, option_manager, app_logger, session_manager
import os
import sys
import glob
import h5py
import numpy
import json
import traceback

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


def gen_pixel_probabilities(session_location, options, master_logger, image_stack=None):
    master_logger.info("Generating Pixel Probabilities") 
 
    if image_stack is None:
        image_stack = imio.read_image_stack(options.image_stack)

    if "extract-ilp-prediction" in options and options.extract_ilp_prediction:
        master_logger.info("Loading saved ilastik volume")
        filename = session_location + "/" + options.pixelprob_name
        os.remove(filename)
        imio.write_ilastik_batch_volume(image_stack, filename)
        image_stack = image_stack.transpose((0, 2, 1))
        f1 = h5py.File(filename, 'a')
        f2 = h5py.File(options.ilp_file, 'r')
        f1.copy(f2['/DataSets/dataItem00/prediction'], 'volume/prediction')
        f1.close()
        f2.close()
    else:
        ilastik_h5 = h5py.File(options.ilp_file, 'r')
        lset = ilastik_h5['DataSets/dataItem00/labels/data']
        lsetrest = lset[0,:,:,:]

        master_logger.debug("Extracting labels")
        (dim1, dim2, dim3, dim4) = numpy.where(lsetrest > 0)
        num_labels = lsetrest.max()
        master_logger.debug(str(num_labels) + " class labels")
        master_logger.debug(str(len(dim1)) + " total label points extracted")

        for i in xrange(num_labels):
            (dim1, dum1, dum2, dum3) = numpy.where(lsetrest == (i+1))
            master_logger.info(str(len(dim1)) + " class " + str(i + 1) + " label points")
        featureset = ilastik_h5['Project/FeatureSelection/UserSelection']
        master_logger.debug("Features selected...")
        for val in featureset:
            master_logger.debug(str(val) + "\n")
        ilastik_h5.close()

        if os.path.exists(session_location + "/image_stack.h5"):
            os.remove(session_location + "/image_stack.h5")
        imio.write_ilastik_batch_volume(image_stack, session_location + "/image_stack.h5")

        #create temporary json for ilastik batch
        json_val_out = {}
        json_val_out["output_dir"] = session_location 
        json_val_out["session"] = options.ilp_file
        json_val_out["memory_mode"] = False
        image_array = [] 
        image_array.append( { "name" : (session_location + "/image_stack.h5") } )
        json_val_out["images"] = image_array
        json_val_out["features"] = []

        json_file_out = open(session_location + "/ilastik.json", "w")
        json_out = json.dumps(json_val_out, indent=4)
        master_logger.debug("ilastik batch_fast json file: " + json_out)
        json_file_out.write(json_out)
        json_file_out.close()

        #run ilastik
        master_logger.info("Running Ilastik batch")
        ilastik_command = str("ilastik_batch_fast --config_file=" + session_location + "/ilastik.json")
        os.system(ilastik_command)

        #deleting ilastik
        os.remove(session_location + "/image_stack.h5")
        os.remove(session_location + "/ilastik.json")
        os.rename(session_location + "/image_stack.h5_boundpred.h5",
            session_location + "/" + options.pixelprob_name)


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
 
