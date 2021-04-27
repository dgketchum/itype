import os
from collections import OrderedDict
from random import shuffle, random
import fiona
from shapely.geometry import shape


def balance_features(grids, n_features=13000, out_file=None):
    nones = 0
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
            try:
                d[itype].append(f)
            except KeyError:
                nones += 1
    print('{} IType none'.format(nones))
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
    _shapes = ['/media/hdisk/itype/wa/grid_select_attr_1536.shp']
    out_ = '/media/hdisk/itype/wa/wa_grid_bal_2019.shp'
    balance_features(_shapes, n_features=10000, out_file=out_)
# ========================= EOF ====================================================================
