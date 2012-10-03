import imio, option_manager, app_logger, session_manager
import os
import sys
import glob
import h5py
import numpy
import json
import traceback

def image_stack_verify(options_parser, options, master_logger):
    if options.test_stack is not None:
        if not os.path.exists(options.test_stack):
            raise Exception("Image volume does not exist at " + options.test_stack)

def image_stack_verify2(options_parser, options, master_logger):
    if options.gt_stack is not None:
        if not os.path.exists(options.gt_stack):
            raise Exception("Image volume does not exist at " + options.gt_stack)

def graph_file_verify(options_parser, options, master_logger):
    if options.ragprob_file is not None:
        if not os.path.exists(options.ragprob_file):
            raise Exception("ragprob file not found " + options.ragprob_file + " not found")

def create_valprob_options(options_parser):
    options_parser.create_option("test-stack", "initial segmentation (to any percentage)", 
        required=True, verify_fn=image_stack_verify,
         shortcut='T', warning=True) 
    
    options_parser.create_option("gt-stack", "ground truth stack (~100 percent complete)", 
        default_val=None, required=True, dtype=str, verify_fn=image_stack_verify2, num_args=None,
        shortcut='G', warning=False) 

    options_parser.create_option("ragprob-file", "RAG probability file", 
        default_val=None, required=True, dtype=str, verify_fn=graph_file_verify, num_args=None,
        shortcut='R', warning=False) 

def load_graph_json(json_file):
    json_file_handle = open(json_file)
    json_data = json.load(json_file_handle)

    pairprob_list = []
    for edge in json_data["edge_list"]:
        node1 = edge["node1"]
        node2 = edge["node2"]
        if node1 > node2:
            node2, node1 = node1, node2
        weight = edge["weight"]
        pairprob_list.append((node1, node2, weight))        

    return pairprob_list

def find_gt_bodies(gt_stack, test_stack):
    body2indices = {}    
    for index, value in numpy.ndenumerate(test_stack):
        body2indices.setdefault(value, {})
        value2 = gt_stack[index]
        body2indices[value].setdefault(value2, 0)
        (body2indices[value])[value2] += 1

    body2gtbody = {}
    for key, val in body2indices.items():
        max_val = 0
        max_id = 0
        for key2, val2 in val.items():
            if val2 > max_val:
                max_val = val2
                max_id = key2
        body2gtbody[key] = max_id
    return body2gtbody

def valprob(session_location, options, master_logger):
    master_logger.info("Reading gt_stack")
    gt_stack = imio.read_image_stack(options.gt_stack)
    master_logger.info("Reading test_stack")
    test_stack = imio.read_image_stack(options.test_stack)
    master_logger.info("Finished reading stacks")

    
    master_logger.info("Loading graph json")
    pairprob_list = load_graph_json(options.ragprob_file)
    master_logger.info("Finished loading graph json")

    master_logger.info("Matching bodies to GT")
    body2gtbody = find_gt_bodies(gt_stack, test_stack)
    master_logger.info("Finished matching bodies to GT")

    nomerge_hist = []
    tot_hist = []
    for iter1 in range(0, 101):
        nomerge_hist.append(0)
        tot_hist.append(0)

    for (node1, node2, prob) in pairprob_list:
        tot_hist[int(prob*100)] += 1
        if body2gtbody[node1] != body2gtbody[node2]:
            nomerge_hist[int(prob*100)] += 1

    for iter1 in range(0, 101):
        if tot_hist[iter1] == 0:
            per = 0
        else:
            per = (float(nomerge_hist[iter1])/float(tot_hist[iter1]) * 100)
        print iter1, ", ", per , ", " , tot_hist[iter1] 
    


def entrypoint(argv):
    applogger = app_logger.AppLogger(False, 'valprob')
    master_logger = applogger.get_logger()
   
    try:
        session = session_manager.Session("valprob", "Validate the predicted probabilities against 100% groundtruth", 
            master_logger, applogger, create_valprob_options)    

        valprob(session.session_location, session.options, master_logger)
    except Exception, e:
        master_logger.error(str(traceback.format_exc()))
    except KeyboardInterrupt, err:
        master_logger.error(str(traceback.format_exc()))
 
