import errno
import itertools as it
import json
import os
import uuid

all_sizes = [0.7, 1.0, 1.6, 3.5, 5.0]
all_types = ['Color', 'Texture', 'Edge', 'Orientation']

full_feature_set = list(it.product(all_types, all_sizes))
default_feature_set = list(it.product(all_types[:-1], all_sizes[1:-1]))

def write_segmentation_pipeline_json(jsonfn, ilfn, ilbfns, outdir='.'):
    if isinstance(ilbfns, str) or isinstance(ilbfns, unicode):
        ilbfns = [ilbfns]
    d = {}
    d['images'] = [{'name': ilbfn} for ilbfn in ilbfns]
    d['session'] = ilfn
    d['output_dir'] = outdir
    d['features'] = default_feature_set
    with open(jsonfn, 'w') as f:
        json.dump(d, f)

def make_dir(dirname):
    """
    Make a directory if it doesn't already exist.
    """
    try:
        os.makedirs(dirname)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else:
            raise Exception("Unable to create directory: " + dirname)

def make_temp_dir(base_dirname, prefix='tmp'):
    """
    Make a unique temporary directory as a subdirectory of passed base
    directory name.

    Returns:
        The unique temporary directory created.
    """
    make_dir(base_dirname)
    uid_hex = prefix + uuid.uuid4().hex
    tmp_dir = os.path.join(base_dirname, uid_hex)
    make_dir(tmp_dir)
    return tmp_dir
