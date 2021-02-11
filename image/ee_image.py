# ===============================================================================
# Copyright 2021 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import time
import ee

ee.Initialize()

BOUNDS = 'users/dgketchum/boundaries/MT'
POINTS = 'users/dgketchum/itype/bvrhd_pts'
GRID = 'users/dgketchum/itype/bvrhd_grd'
LABELS = 'users/dgketchum/itype/mt_test'
GS_BUCKET = 'itype'

KERNEL_SIZE = 512
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
list_ = ee.List.repeat(1, KERNEL_SIZE)
lists = ee.List.repeat(list_, KERNEL_SIZE)
KERNEL = ee.Kernel.fixed(KERNEL_SIZE, KERNEL_SIZE, lists)


def create_labels():
    class_labels = ee.Image().byte()
    flood = ee.FeatureCollection(LABELS).filter(ee.Filter.eq("IType", 'F'))
    pivot = ee.FeatureCollection(LABELS).filter(ee.Filter.eq("IType", 'P'))
    sprinkler = ee.FeatureCollection(LABELS).filter(ee.Filter.eq("IType", 'S'))
    class_labels = class_labels.paint(flood, 1)
    class_labels = class_labels.paint(pivot, 2)
    class_labels = class_labels.paint(sprinkler, 3)
    return class_labels


def create_image(roi, start, end):
    def mask(x):
        qa = x.select(['QA60'])
        x = x.updateMask(qa.lt(1))
        return x

    naip = ee.ImageCollection('USDA/NAIP/DOQQ').filterDate(start, end).mosaic()
    proj = naip.projection().getInfo()
    sent = ee.ImageCollection('COPERNICUS/S2').filterDate(start, end).filterBounds(roi)
    sent = sent.map(lambda x: mask(x))
    ndvi = sent.map(lambda x: x.addBands(x.normalizedDifference(['B8', 'B4'])))
    std_ndvi = ndvi.select('nd').reduce(ee.Reducer.stdDev()).reproject(crs=proj['crs'], scale=1.0)
    max_ndvi = ndvi.select('nd').reduce(ee.Reducer.max()).reproject(crs=proj['crs'], scale=1.0)
    naip = naip.addBands([std_ndvi.rename('std_ndvi'), max_ndvi.rename('mx_ndvi')])
    band_names = naip.bandNames().getInfo()
    return naip, band_names


def extract_by_point(year, n_shards=10):
    roi = ee.FeatureCollection(BOUNDS).geometry()
    points_fc = ee.FeatureCollection(POINTS)
    points_fc = points_fc.toList(points_fc.size())

    s, e = '{}-01-01'.format(year), '{}-12-31'.format(year)
    image_stack, features = create_image(roi, start=s, end=e)

    i_labels = create_labels()
    image_stack = ee.Image.cat([image_stack, i_labels]).float()
    features = features + ['label']

    projection = ee.Projection('EPSG:3857')
    image_stack = image_stack.reproject(projection, None, 1.0)
    data_stack = image_stack.neighborhoodToArray(KERNEL)

    ct = 0
    geometry_sample = None
    for idx in range(points_fc.size().getInfo()):
        point = ee.Feature(points_fc.get(idx))
        geometry_sample = ee.ImageCollection([])

        sample = data_stack.sample(
            region=point.geometry(),
            scale=1.0,
            numPixels=1,
            tileScale=16,
            dropNulls=False)

        geometry_sample = geometry_sample.merge(sample)
        if (ct + 1) % n_shards == 0:
            name_ = '{}_{}'.format(str(year), idx)
            export_task(geometry_sample, features, filename=name_)
            geometry_sample = None
        ct += 1
    if geometry_sample:
        name_ = '{}_{}'.format(str(year), idx)

        export_task(geometry_sample, features, filename=name_)

    print('exported {}, {} features'.format(year, ct))


def export_task(sample, features, filename):
    task = ee.batch.Export.table.toCloudStorage(
        collection=sample,
        bucket=GS_BUCKET,
        description=filename,
        fileNamePrefix=filename,
        fileFormat='TFRecord',
        selectors=features)

    try:
        task.start()
        print(filename)
    except ee.ee_exception.EEException:
        print('waiting 50 minutes to export {}'.format(filename))
        time.sleep(3000)
        task.start()


if __name__ == '__main__':
    extract_by_point(2019)
# ========================= EOF ====================================================================
