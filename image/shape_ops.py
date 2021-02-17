import os
from collections import OrderedDict
from random import shuffle
import fiona


def balance_features(shp, n_features=13000, out_file=None):
    d = {'F': [],
         'S': [],
         'P': [],
         'NI': [],
         'NC': []}
    with fiona.open(shp) as src:
        features = [f for f in src]
        meta = src.meta
    meta['schema']['properties'] = OrderedDict([('FID', 'int:10'),
                                                ('SPLIT', 'str:10'),
                                                ('IType', 'str:10')])
    shuffle(features)
    for f in features:
        itype = f['properties']['IType']
        if len(d[itype]) >= n_features:
            continue
        else:
            d[itype].append(f)
    ct = 1
    with fiona.open(out_file, 'w', **meta) as dst:
        for k, v in d.items():
            for f in v:
                out_feat = {'type': 'Feature', 'properties': OrderedDict([('FID', ct),
                                                                          ('SPLIT', f['properties']['SPLIT']),
                                                                          ('IType', f['properties']['IType'])]),
                            'geometry': f['geometry']}
                dst.write(out_feat)
                ct += 1


if __name__ == '__main__':
    d = '/media/hdisk/itype/grid'
    in_ = os.path.join(d, 'shards_5class.shp')
    out_ = os.path.join(d, 'balanced_shards.shp')
    balance_features(in_, n_features=13800, out_file=out_)
# ========================= EOF ====================================================================
