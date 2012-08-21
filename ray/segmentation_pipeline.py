#!/usr/bin/env python


import sys
import os
import argparse
import h5py
import numpy
import shutil
import logging
from skimage import morphology as skmorph
from scipy.ndimage import label

from . import imio, agglo, morpho, classify, evaluate, app_logger, \
    session_manager, pixel

try:
    from ray import stack_np
except ImportError:
    np_installed = False   
else:
    np_installed = True

try:
    import syngeo
except ImportError:
    logging.warning('Could not import syngeo. ' +
                                        'Synapse-aware mode not available.')

def grab_boundary(prediction, channels, master_logger):
    boundary = None
    master_logger.debug("Grabbing boundary labels: " + str(channels))
    for channel_id in channels:
        if boundary is None:
            boundary = prediction[...,channel_id] 
        else:
            boundary += prediction[...,channel_id]

    return boundary


def gen_supervoxels(session_location, options, prediction_file, master_logger):
    master_logger.debug("Generating supervoxels")
    if not os.path.isfile(prediction_file):
        raise Exception("Training file not found: " + prediction_file)

    prediction = imio.read_image_stack(prediction_file, group='/volume/prediction', single_channel=False)

    if options.extract_ilp_prediction:
        prediction = prediction.transpose((2, 1, 0))

    boundary = grab_boundary(prediction, options.bound_channels, master_logger) 

    master_logger.debug("watershed seed value threshold: " + str(options.seed_val))
    seeds = label(boundary<=options.seed_val)[0]

    if options.seed_size > 0:
        master_logger.debug("Removing small seeds")
        seeds = morpho.remove_small_connected_components(seeds, options.seed_size)
        master_logger.debug("Finished removing small seeds")

    master_logger.info("Starting watershed")
    supervoxels = skmorph.watershed(boundary, seeds)
    master_logger.info("Finished watershed")

    if options.synapse_file is not None:
        master_logger.info("Processing synapses")
        pre_post_pairs = syngeo.io.raveler_synapse_annotations_to_coords(
            options.synapse_file)
        synapse_volume = syngeo.io.volume_synapse_view(pre_post_pairs, boundary.shape)
        supervoxels = morpho.split_exclusions(boundary, supervoxels, synapse_volume, 
                                                        options.synapse_dilation)
        master_logger.info("Finished processing synapses")
   
    return supervoxels, prediction


def agglomeration(options, agglom_stack, supervoxels, prediction, 
        image_stack, session_location, sp_outs, master_logger):
    
    seg_thresholds = sorted(options.segmentation_thresholds)
    for threshold in seg_thresholds:
        if threshold != 0 or not options.use_neuroproof:
            master_logger.info("Starting agglomeration to threshold " + str(threshold)
                + " with " + str(agglom_stack.number_of_nodes()))
            agglom_stack.agglomerate(threshold)
            master_logger.info("Finished agglomeration to threshold " + str(threshold)
                + " with " + str(agglom_stack.number_of_nodes()))
            
            if options.inclusion_removal:
                inclusion_removal(agglom_stack, master_logger)

        segmentation = agglom_stack.get_segmentation()     

        if options.h5_output:
            imio.write_image_stack(segmentation,
                session_location+"/agglom-"+str(threshold)+".lzf.h5", compression='lzf')
           
        if options.raveler_output:
            sps_outs = output_raveler(segmentation, supervoxels, image_stack, "agglom-" + str(threshold),
                session_location, master_logger)   
            master_logger.info("Writing graph.json")
            agglom_stack.write_plaza_json(session_location+"/raveler-export/agglom-"+str(threshold)+"/graph.json",
                                            options.synapse_file)
            master_logger.info("Finished writing graph.json")



def inclusion_removal(agglom_stack, master_logger):
    master_logger.info("Starting inclusion removal with " + str(agglom_stack.number_of_nodes()) + " nodes")
    agglom_stack.remove_inclusions()
    master_logger.info("Finished inclusion removal with " + str(agglom_stack.number_of_nodes()) + " nodes")


def output_raveler(segmentation, supervoxels, image_stack, name, session_location, master_logger, sps_out=None):
    outdir = session_location + "/raveler-export/" + name + "/"
    master_logger.info("Exporting Raveler directory: " + outdir)

    rav = imio.segs_to_raveler(supervoxels, segmentation, 0, do_conn_comp=False, sps_out=sps_out)
    sps_out, dummy1, dummy2 = rav
    
    if os.path.exists(outdir):
        master_logger.warning("Overwriting Raveler diretory: " + outdir)
        shutil.rmtree(outdir)
    imio.write_to_raveler(*rav, directory=outdir, gray=image_stack)
    return sps_out



def flow_perform_agglomeration(options, supervoxels, prediction, image_stack,
                                session_location, sps_out, master_logger): 
    # make synapse constraints
    synapse_volume = numpy.array([])
    if not options.use_neuroproof and options.synapse_file is not None:
        pre_post_pairs = syngeo.io.raveler_synapse_annotations_to_coords(
            options.synapse_file)
        synapse_volume = \
            syngeo.io.volume_synapse_view(pre_post_pairs, supervoxels.shape)

     # ?! build RAG (automatically load features if classifier file is available, default to median
    # if no classifier, check if np mode or not, automatically load features in NP as well)

    if options.classifier is not None:
        cl = classify.RandomForest()
        fm_info = cl.load_from_disk(options.classifier)

        master_logger.info("Building RAG")
        if options.use_neuroproof:
            if not fm_info["neuroproof_features"]:
                raise Exception("random forest created not using neuroproof") 
            agglom_stack = stack_np.Stack(supervoxels, prediction,
                single_channel=False, classifier=cl, feature_info=fm_info, synapse_file=options.synapse_file,
                master_logger=master_logger) 
        else:
            if fm_info["neuroproof_features"] is not None:
                master_logger.warning("random forest created using neuroproof features -- should still work") 
            fm = features.io.create_fm(fm_info)
            if options.expected_vi:
                mpf = agglo.expected_change_vi(fm, cl, beta=options.vi_beta)
            else:
                mpf = agglo.classifier_probability(fm, cl)
            
            agglom_stack = agglo.Rag(supervoxels, prediction, mpf,
                    feature_manager=fm, show_progress=True, nozeros=True, 
                    exclusions=synapse_volume)
        master_logger.info("Finished building RAG")
    else:
        boundary = grab_boundary(prediction, options.bound_channels, master_logger)   
        if options.use_neuroproof:
            agglom_stack = stack_np.Stack(supervoxels, boundary, synapse_file=options.synapse_file,
                        master_logger=master_logger)
        else:
            agglom_stack = agglo.Rag(supervoxels, boundary, merge_priority_function=agglo.boundary_median,
                show_progress=True, nozeros=True, exclusions=synapse_volume)


    # remove inclusions 
    if options.inclusion_removal:
        inclusion_removal(agglom_stack, master_logger) 

    # actually perform the agglomeration
    agglomeration(options, agglom_stack, supervoxels, prediction, image_stack,
        session_location, sps_out, master_logger) 





def run_segmentation_pipeline(session_location, options, master_logger): 
    # read grayscale
    image_stack = None
    if options.image_stack is not None:
        image_stack = imio.read_image_stack(options.image_stack)

    prediction_file = None
    # run boundary prediction -- produces a prediction file 
    if options.gen_pixel:
        pixel.gen_pixel_probabilities(session_location, options, master_logger, image_stack)
        prediction_file = session_location + "/" + options.pixelprob_name
    else:
        prediction_file  = options.pixelprob_file
        

    # generate supervoxels -- produces supervoxels and output as appropriate
    supervoxels = None
    prediction = None
    if options.gen_supervoxels:
        supervoxels, prediction = gen_supervoxels(session_location, options, prediction_file, master_logger) 
    elif options.supervoxels_file:
        master_logger.info("Reading supervoxels: " + options.supervoxels_file)
        supervoxels = imio.read_image_stack(options.supervoxels_file) 
        master_logger.info("Finished reading supervoxels")

    sps_out = None  
    if supervoxels is not None:
        if options.h5_output:
            imio.write_image_stack(supervoxels,
                session_location + "/" + options.supervoxels_name, compression='lzf')

        if options.raveler_output:
            sps_out = output_raveler(supervoxels, supervoxels, image_stack, "supervoxels", session_location, master_logger)

    # agglomerate and generate output
    if options.gen_agglomeration:
        if prediction is None and options.pixelprob_file is not None:
            master_logger.info("Reading pixel prediction: " + options.pixelprob_file)
            prediction = imio.read_image_stack(options.pixelprob_file, group='/volume/prediction', single_channel=False)
            master_logger.info("Finished reading pixel prediction")
        elif prediction is None:
            raise Exception("No pixel probs available for agglomeration")

        flow_perform_agglomeration(options, supervoxels, prediction, image_stack,
                                session_location, sps_out, master_logger) 
                


def np_verify(options_parser, options, master_logger):
    if options.use_neuroproof and not np_installed:
        raise Exception("NeuroProof not properly installed on your machine.  Install or disable neuroproof")
        
def synapse_file_verify(options_parser, options, master_logger):
    if options.synapse_file:
        if not os.path.exists(options.synapse_file):
            raise Exception("Synapse file " + options.synapse_file + " not found")
        if not options.synapse_file.endswith('.json'):
            raise Exception("Synapse file " + options.synapse_file + " does not end with .json")

def classifier_verify(options_parser, options, master_logger):
    if options.classifier is not None:
        if not os.path.exists(options.classifier):
            raise Exception("Classifier " + options.classifier + " not found")
        if not options.classifier.endswith('.h5'):
            raise Exception("Classifier " + options.classifier + " does not end with .h5")

def gen_supervoxels_verify(options_parser, options, master_logger):
    if options.gen_supervoxels and not options.gen_pixel and options.pixelprob_file is None:
        raise Exception("Must have a pixel prediction to generate supervoxels")
     

def supervoxels_file_verify(options_parser, options, master_logger):
    if options.supervoxels_file is not None:
        if not os.path.exists(options.supervoxels_file):
            raise Exception("Supervoxel file " + options.supervoxels_file + " does not exist")

def gen_agglomeration_verify(options_parser, options, master_logger):
    if options.gen_agglomeration:
        if not options.gen_supervoxels and options.supervoxels_file is None:
            raise Exception("No supervoxels available for agglomeration")
        if not options.gen_pixel and options.pixelprob_file is None:
            raise Exception("No prediction available for agglomeration")


def create_segmentation_pipeline_options(options_parser):
    pixel.create_pixel_options(options_parser, False)
    
    options_parser.create_option("use-neuroproof", "Use NeuroProof", 
        default_val=False, required=False, dtype=bool, verify_fn=np_verify, num_args=None,
        shortcut='NP', warning=False, hidden=(not np_installed)) 

    options_parser.create_option("supervoxels-name", "Name for the supervoxel segmentation", 
        default_val="supervoxels.lzf.h5", required=False, dtype=str, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("supervoxels-file", "Supervoxel segmentation file or directory stack", 
        default_val=None, required=False, dtype=str, verify_fn=supervoxels_file_verify, num_args=None,
        shortcut=None, warning=False, hidden=True) 
   
    options_parser.create_option("gen-supervoxels", "Enable supervoxel generation", 
        default_val=False, required=False, dtype=bool, verify_fn=gen_supervoxels_verify, num_args=None,
        shortcut='GS', warning=True, hidden=False) 

    options_parser.create_option("inclusion-removal", "Disable inclusion removal", 
        default_val=True, required=False, dtype=bool, verify_fn=None, num_args=None,
        shortcut='IR', warning=False, hidden=False) 

    options_parser.create_option("seed-val", "Threshold for choosing seeds", 
        default_val=0, required=False, dtype=int, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("seed-size", "Threshold for seed size", 
        default_val=0, required=False, dtype=int, verify_fn=None, num_args=None,
        shortcut='SS', warning=False, hidden=False) 

    options_parser.create_option("synapse-file", "Json file containing synapse information", 
        default_val=None, required=False, dtype=str, verify_fn=synapse_file_verify, num_args=None,
        shortcut='SJ', warning=False, hidden=False) 

    options_parser.create_option("segmentation-thresholds", "Segmentation thresholds", 
        default_val=[], required=False, dtype=float, verify_fn=None, num_args='+',
        shortcut='ST', warning=True, hidden=False) 

    options_parser.create_option("gen-agglomeration", "Enable agglomeration", 
        default_val=False, required=False, dtype=bool, verify_fn=gen_agglomeration_verify, num_args=None,
        shortcut='GA', warning=True, hidden=False) 
    
    options_parser.create_option("raveler-output", "Disable Raveler output", 
        default_val=True, required=False, dtype=bool, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("h5-output", "Enable h5 output", 
        default_val=False, required=False, dtype=bool, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("classifier", "H5 file containing RF", 
        default_val=None, required=False, dtype=str, verify_fn=classifier_verify, num_args=None,
        shortcut='k', warning=False, hidden=False) 

    options_parser.create_option("bound-channels", "Channel numbers designated as boundary", 
        default_val=[0], required=False, dtype=int, verify_fn=None, num_args='+',
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("expected-vi", "Enable expected VI during agglomeration", 
        default_val=False, required=False, dtype=bool, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("vi-beta", "Relative penalty for false merges in weighted expected VI", 
        default_val=1.0, required=False, dtype=float, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("synapse-dilation", "Dilate synapse points by this amount", 
        default_val=1, required=False, dtype=int, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 


def entrypoint(argv):
    applogger = app_logger.AppLogger(False, 'seg-pipeline')
    master_logger = applogger.get_logger()

    try: 
        session = session_manager.Session("seg-pipeline", "Segmentation pipeline (featuring boundary prediction, median agglomeration or trained agglomeration, inclusion removal, and raveler exports)", 
            master_logger, applogger, create_segmentation_pipeline_options)    

        run_segmentation_pipeline(session.session_location, session.options, master_logger) 
    except Exception, e:
        master_logger.error(e)
    except KeyboardInterrupt, err:
        master_logger.error(e)
 
   
if __name__ == "__main__":
    sys.exit(main(sys.argv))
