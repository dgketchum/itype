import os
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
    shuffle(features)
    for f in features:
        itype = f['properties']['IType']
        if len(d[itype]) >= n_features:
            continue
        else:
            d[itype].append(f)
    with fiona.open(out_file, 'w', **meta) as dst:
        for k, v in d.items():
            for f in v:
                dst.write(f)


if __name__ == '__main__':
    d = '/media/hdisk/itype/grid'
    in_ = os.path.join(d, 'shards_5class.shp')
    out_ = os.path.join(d, 'balanced_shards.shp')
    balance_features(in_, n_features=13800, out_file=out_)
# ========================= EOF ====================================================================
