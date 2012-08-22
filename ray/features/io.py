from . import base
from . import inclusion, moments, histogram

def create_fm(fm_info):
    children = []
    for feature in fm_info['feature_list']:
        if feature == "histogram":
            children.append(histogram.Manager.load_dict(fm_info[feature]))
        elif feature == "moments":
            children.append(moments.Manager.load_dict(fm_info[feature]))
        elif feature == "inclusiveness":
            children.append(inclusion.Manager.load_dict(fm_info[feature]))
        else:
            raise Exception("Feature " + feature + " not found") 
    if len(children) == 0:
        raise RuntimeError("No features loaded")
    if len(children) == 1:
        return children[0]
    return base.Composite(children=children) 

