from numpy import bool, uint8


def pixel_wise_boundary_precision_recall(aseg, gt):
    gt = (1-gt.astype(bool)).astype(uint8)
    aseg = (1-aseg.astype(bool)).astype(uint8)
    tp = float((gt * aseg).sum())
    fp = (aseg * (1-gt)).sum()
    fn = (gt * (1-aseg)).sum()
    return tp/(tp+fp), tp/(tp+fn)

def body_rand(aseg, gt):
    amap = dict()
    asizes = dict()
    gmap = dict()
    gsizes = dict()
    for a, g in zip(aseg.ravel(), gt.ravel()):
        try:
            amap[a][g] += 1
        except KeyError:
            try:
                amap[a][g] = 1
            except KeyError:
                amap[a] = {g: 1}
        try:
            gmap[g][a] += 1
        except KeyError:
            try:
                gmap[g][a] = 1
            except KeyError:
                gmap[g] = {a: 1}
        try:
            asizes[a] += 1
        except KeyError:
            asizes[a] = 1
        try:
            gsizes[g] += 1
        except KeyError:
            gsizes[g] = 1
    
