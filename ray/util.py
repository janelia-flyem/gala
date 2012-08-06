import json
import itertools as it

all_sizes = [0.7, 1.0, 1.6, 3.5, 5.0]
all_types = ['Color', 'Texture', 'Edge', 'Orientation']

full_feature_set = list(it.product(all_types, all_sizes))
default_feature_set = list(it.product(all_types[:-1], all_sizes[1:-1]))

def write_segmentation_pipeline_json(jsonfn, ilfn, ilbfn, outdir='.'):
    d = {}
    d['images'] = [{'name': ilbfn}]
    d['session'] = ilfn
    d['output_dir'] = outdir
    d['features'] = default_feature_set
    with open(jsonfn, 'w') as f:
        json.dump(d, f)
