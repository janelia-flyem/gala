import imio, option_manager, app_logger, session_manager
import libNeuroProofPriority as neuroproof
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

def create_auto_options(options_parser):
    options_parser.create_option("test-stack", "initial segmentation (to any percentage)", 
        required=True, verify_fn=image_stack_verify,
         shortcut='T', warning=True) 
    
    options_parser.create_option("gt-stack", "ground truth stack (~100 percent complete)", 
        default_val=None, required=True, dtype=str, verify_fn=image_stack_verify2, num_args=None,
        shortcut='G', warning=False) 

    options_parser.create_option("ragprob-file", "RAG probability file", 
        default_val=None, required=True, dtype=str, verify_fn=graph_file_verify, num_args=None,
        shortcut='R', warning=False) 

    options_parser.create_option("size-threshold", "Number of voxels used in threshold", 
        default_val=25000, required=False, dtype=int, verify_fn=None, num_args=None,
        shortcut='ST', warning=False) 

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

def process_edge(body2gtbody, nomerge_hist, tot_hist, nomerge_hist2, tot_hist2, dirtybodies, bodyremap):
    priority = neuroproof.get_next_edge()
    (body1, body2) = priority.body_pair
    weight = neuroproof.get_edge_val(priority)  

    if body1 not in dirtybodies and body2 not in dirtybodies:
        tot_hist[int(weight*100)] += 1
    tot_hist2[int(weight*100)] += 1
    link = True
    if body2gtbody[body1] != body2gtbody[body2]:
        if body1 not in dirtybodies and body2 not in dirtybodies:
            nomerge_hist[int(weight*100)] += 1
        nomerge_hist2[int(weight*100)] += 1
        link = False
    else:
        if body2 not in bodyremap:
            bodyremap[body2] = [body2]
        if body1 not in bodyremap:
            bodyremap[body1] = [body1]

        dirtybodies.add(body1)
        bodyremap[body1].extend(bodyremap[body2])
        del bodyremap[body2]


    neuroproof.set_edge_result(priority.body_pair, link)


def auto_proofread(body2gtbody, rag_file, size_threshold, master_logger, test_stack, session_location):
    nomerge_hist = []
    tot_hist = []
    nomerge_hist2 = []
    tot_hist2 = []
    dirtybodies = set()
    for iter1 in range(0, 101):
        nomerge_hist.append(0)
        tot_hist.append(0)
        nomerge_hist2.append(0)
        tot_hist2.append(0)

    neuroproof.initialize_priority_scheduler(rag_file, 0.1, 0.9, 0.1)

    bodyremap = {}

    num_body = 0   
    neuroproof.set_body_mode(size_threshold, 0) 
    while neuroproof.get_estimated_num_remaining_edges() > 0:
        process_edge(body2gtbody, nomerge_hist, tot_hist, nomerge_hist2, tot_hist2, dirtybodies, bodyremap)
        num_body += 1

    num_synapse = 0   
    neuroproof.set_synapse_mode(0.1) 
    while neuroproof.get_estimated_num_remaining_edges() > 0:
        process_edge(body2gtbody, nomerge_hist, tot_hist, nomerge_hist2, tot_hist2, dirtybodies, bodyremap)
        num_synapse += 1

    num_orphan = 0   
    neuroproof.set_orphan_mode(size_threshold, size_threshold, size_threshold) 
    while neuroproof.get_estimated_num_remaining_edges() > 0:
        process_edge(body2gtbody, nomerge_hist, tot_hist, nomerge_hist2, tot_hist2, dirtybodies, bodyremap)
        num_orphan += 1

    master_logger.info("Probability Actual Agreement with Groundtruth Flat")
    for iter1 in range(0, 101):
        if tot_hist[iter1] == 0:
            per = 0
        else:
            per = (float(nomerge_hist[iter1])/float(tot_hist[iter1]) * 100)
        print iter1, ", ", per , ", " , tot_hist[iter1] 

    master_logger.info("Probability Actual Agreement with Groundtruth Est")
    for iter1 in range(0, 101):
        if tot_hist2[iter1] == 0:
            per = 0
        else:
            per = (float(nomerge_hist2[iter1])/float(tot_hist2[iter1]) * 100)
        print iter1, ", ", per , ", " , tot_hist2[iter1] 

    body2body = {}
    for key, vallist in bodyremap.items():
        for body in vallist:
            body2body[body] = key

    os.system("cp -R " + test_stack + "/superpixel_maps " + session_location + "/") 
    os.system("cp " + test_stack + "/superpixel_to_segment_map.txt " + session_location + "/") 

    mapping_file = open(test_stack + "/segment_to_body_map.txt")
    outfile = open(session_location + "/segment_to_body_map.txt", 'w')

    for line in mapping_file.readlines():
        vals = line.split(' ')
        
        seg = int(vals[0])
        body = int(vals[1])

        if body in body2body:
            body = body2body[body]
        outfile.write(str(seg) + " " + str(body) + "\n")

    master_logger.info("Num body: " + str(num_body))
    master_logger.info("Num synapse: " + str(num_synapse))
    master_logger.info("Num orphan: " + str(num_orphan))
    master_logger.info("Num total: " + str(num_body + num_synapse + num_orphan))


def auto(session_location, options, master_logger):
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

    body2body = {}

    for (node1, node2, dummy) in pairprob_list:
        body2body[node1] = node1
        body2body[node2] = node2
    
    for (node1, node2, dummy) in pairprob_list:
        if body2gtbody[node1] == body2gtbody[node2]:
            print "merge: " , node1, node2
            node2 = body2body[node2]
            node1 = body2body[node1]
            body2body[node1] = node2
            remap_list = []
            for b1, b2 in body2body.items():
                if b2 == node1:
                    remap_list.append(b1)
            for b1 in remap_list:
                body2body[b1] = node2
        else:
            print "split: " , node1, node2

    f1 = h5py.File('proofread.h5', 'w')
    f1.create_dataset('stack', data=test_stack)
    arr = numpy.array(body2body.items())
    f1.create_dataset('transforms', data=arr)
        

def entrypoint(argv):
    applogger = app_logger.AppLogger(False, 'auto')
    master_logger = applogger.get_logger()
   
    try:
        session = session_manager.Session("auto", "Validate the predicted probabilities against 100% groundtruth", 
            master_logger, applogger, create_auto_options)    

        auto(session.session_location, session.options, master_logger)
    except Exception, e:
        master_logger.error(str(traceback.format_exc()))
    except KeyboardInterrupt, err:
        master_logger.error(str(traceback.format_exc()))
 
