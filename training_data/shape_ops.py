import os
from collections import OrderedDict
from random import shuffle, random
import fiona
from shapely.geometry import shape


def balance_features(grids, n_features=13000, out_file=None):
    d = {'F': [],
         'S': [],
         'P': [],
         'NI': [],
         'NC': []}
    first = True
    dup = 0
    features, centroids = [], []
    for shp in grids:
        with fiona.open(shp) as src:
            if first:
                [features.append(f) for f in src]
                [centroids.append(shape(c['geometry']).centroid.wkb_hex) for c in src]
                meta = src.meta
                first = False
            else:
                for f in src:
                    c = shape(f['geometry']).centroid.wkb_hex
                    if c not in centroids:
                        features.append(f)
                        centroids.append(c)
                    else:
                        dup += 1

    meta['schema']['properties'] = OrderedDict([('FID', 'int:10'),
                                                ('SPLIT', 'str:10'),
                                                ('IType', 'str:10')])
    shuffle(features)
    for f in features:
        itype = f['properties']['IType']
        if itype in ['F', 'NI', 'NC'] and len(d[itype]) >= n_features:
            continue
        else:
            d[itype].append(f)
    ct = 1
    with fiona.open(out_file, 'w', **meta) as dst:
        for k, v in d.items():
            for f in v:
                r = random()
                if r <= 0.6:
                    split = 'train'
                if 0.6 < r <= 0.8:
                    split = 'test'
                if r > 0.8:
                    split = 'valid'
                out_feat = {'type': 'Feature',
                            'properties': OrderedDict([('FID', ct),
                                                       ('SPLIT', split),
                                                       ('IType', f['properties']['IType'])]),
                            'geometry': f['geometry']}
                dst.write(out_feat)
                ct += 1


if __name__ == '__main__':
    in_ = '/media/hdisk/itype/grid'
    _shapes = ['itype_grid.shp', 'wetlands_grid.shp', 'dryland_grid.shp']
    _files = [os.path.join(in_, s) for s in _shapes]
    # out_ = os.path.join(in_, 'grid_2009', 'mt_grid_bal_2009.shp')
    out_ = '/home/dgketchum/Downloads/test_write.shp'
    balance_features(_files, n_features=5300, out_file=out_)
# ========================= EOF ====================================================================
