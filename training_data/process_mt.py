import os
from collections import OrderedDict
import fiona


def process_mt(shp, year, out_file=None):
    features, centroids = [], []
    with fiona.open(shp) as src:
        [features.append(f) for f in src]
        meta = src.meta

    meta['schema']['properties'] = OrderedDict([('FID', 'int:10'),
                                                ('YEAR', 'int:10'),
                                                ('IType', 'str:10')])
    ct = 0
    with fiona.open(out_file, 'w', **meta) as dst:
        for f in features:
            if not f['properties']['IType']:
                continue
            if f['geometry']['type'] != 'Polygon':
                continue
            out_feat = {'type': 'Feature',
                        'properties': OrderedDict([('FID', ct),
                                                   ('YEAR', year),
                                                   ('IType', f['properties']['IType'])]),
                        'geometry': f['geometry']}
            dst.write(out_feat)
            ct += 1


if __name__ == '__main__':
    in_ = '/media/hdisk/itype/mt_flu/raw_flu_wgs'
    out_ = '/media/hdisk/itype/mt_flu/flu_itype'
    _shapes = sorted([os.path.join(in_, x) for x in os.listdir(in_) if x.endswith('.shp')])
    for s in _shapes:
        yr = int(s.split('.')[0][-4:])
        outfile = os.path.join(out_, 'mt_itype_{}.shp'.format(yr))
        print(outfile)
        process_mt(s, yr, out_file=outfile)
# ========================= EOF ====================================================================

