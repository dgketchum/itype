import os
from collections import OrderedDict
import fiona

CLASS_MAP = {'Big Gun': 'S',
             'Big Gun/Center Pivot': 'P',
             'Big Gun/Drip': 'S',
             'Big Gun/Sprinkler': 'S',
             'Big Gun/Sprinkler/Wheel Line': 'S',
             'Big Gun/Wheel Line': 'S',
             'Center Pivot': 'P',
             'Center Pivot/Drip': 'P',
             'Center Pivot/Drip/Sprinkler': 'P',
             'Center Pivot/None': 'P',
             'Center Pivot/Rill': 'P',
             'Center Pivot/Rill/Sprinkler': 'P',
             'Center Pivot/Rill/Wheel Line': 'P',
             'Center Pivot/Sprinkler': 'P',
             'Center Pivot/Sprinkler/Wheel Line': 'P',
             'Center Pivot/Wheel Line': 'P',
             'Drip': 'S',
             'Drip/Big Gun': 'S',
             'Drip/Micro-Sprinkler': 'S',
             'Drip/None': 'S',
             'Drip/Rill': 'S',
             'Drip/Sprinkler': 'S',
             'Drip/Sprinkler/Wheel Line': 'S',
             'Drip/Wheel Line': 'S',
             'Flood': 'F',
             'Hand': 'S',
             'Hand/Rill': 'F',
             'Hand/Sprinkler': 'S',
             'Micro-Sprinkler': 'S',
             'None': 'NI',
             'None/Rill': 'NI',
             'None/Sprinkler': 'NI',
             'None/Wheel Line': 'NI',
             'Rill': 'F',
             'Rill/Sprinkler': 'F',
             'Rill/Sprinkler/Wheel Line': 'F',
             'Rill/Wheel Line': 'F',
             'Sprinkler': 'S',
             'Sprinkler/Wheel Line': 'S',
             'Unknown': None,
             'Wheel Line': 'S'}


def process_mt(shp, out_file=None):
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
            if not f['properties']['Irrigation']:
                continue
            if f['geometry']['type'] != 'Polygon':
                continue
            itype = CLASS_MAP[f['properties']['Irrigation']]
            out_feat = {'type': 'Feature',
                        'properties': OrderedDict([('FID', ct),
                                                   ('YEAR', 2020),
                                                   ('IType', itype)]),
                        'geometry': f['geometry']}
            dst.write(out_feat)
            ct += 1


if __name__ == '__main__':
    in_ = '/media/hdisk/itype/wa/wa_2020.shp'
    out_ = '/media/hdisk/itype/wa/wa_2020_sort4class.shp'
    process_mt(in_, out_)
# ========================= EOF ====================================================================
