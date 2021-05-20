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
import os
import time

import ee
from google.cloud import storage
from image.landsat import landsat_composite
from image.sentinel import sentinel_composite

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/dgketchum/ssebop-montana-57d2b4da4339.json'

ee.Initialize()

GS_BUCKET = 'itype_ndvi'

KERNEL_SIZE = 512
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
list_ = ee.List.repeat(1, KERNEL_SIZE)
lists = ee.List.repeat(list_, KERNEL_SIZE)
KERNEL = ee.Kernel.fixed(KERNEL_SIZE, KERNEL_SIZE, lists)


class ITypeDataStack(object):

    def __init__(self, year, split='train', satellite='sentinel', fid=None, dataset='mt'):

        self.fid = fid
        self.year = year
        self.start, self.end = '{}-01-01'.format(year), '{}-12-31'.format(year)
        self.split = split
        self.dataset = dataset

        self.bounds = 'users/dgketchum/boundaries/{}'.format(dataset.upper())

        self.grid = 'users/dgketchum/itype/{}_grid_{}'.format(dataset, year)
        self.irr_labels = 'users/dgketchum/itype/{}_itype_{}'.format(dataset, year)

        self.dryland_labels = 'users/dgketchum/itype/{}_dryland'.format(dataset)
        self.uncult_labels = 'users/dgketchum/itype/{}_uncultivated'.format(dataset)
        self.wetland_labels = 'users/dgketchum/itype/{}_wetlands'.format(dataset)
        self.projection = ee.Projection('EPSG:3857')

        self.satellite = satellite

        self.basename = os.path.basename(self.irr_labels)
        self.points_fc, self.features = None, None
        self.data_stack, self.image_stack = None, None
        self.gcs_bucket = GS_BUCKET
        self.kernel = KERNEL
        self.task = None

    def export_geotiff(self, overwrite=False):
        self._build_data()
        bucket_contents = None
        if not overwrite:
            bucket_contents = self._get_bucket_contents()

        idxs = list(range(self.grid_fc.size().getInfo()))
        idxs.sort(reverse=False)
        for idx in idxs:
            patch = ee.Feature(self.grid_fc.get(idx))
            fid = patch.getInfo()['properties']['FID']
            name_ = '{}_{}_{}'.format(self.dataset, self.split, str(fid).rjust(7, '0'))
            if not overwrite:
                if name_ in bucket_contents:
                    print('{} exists, skippping'.format(name_))
                    continue
            kwargs = {'image': self.image_stack,
                      'bucket': self.gcs_bucket,
                      'description': name_,
                      'fileNamePrefix': name_,
                      'crs': 'EPSG:4326',
                      'region': patch.geometry(),
                      'dimensions': '1536x1536',
                      'fileFormat': 'GeoTIFF',
                      'maxPixels': 1e13}

            self.task = ee.batch.Export.image.toCloudStorage(**kwargs)
            print('export {}'.format(name_))
            self._start_task()

    def _build_labels(self):

        class_labels = ee.Image(0).byte().rename('itype')
        flood = ee.FeatureCollection(self.irr_labels).filter(ee.Filter.eq("IType", 'F'))
        sprinkler = ee.FeatureCollection(self.irr_labels).filter(ee.Filter.eq("IType", 'S'))
        pivot = ee.FeatureCollection(self.irr_labels).filter(ee.Filter.eq("IType", 'P'))
        dryland = ee.FeatureCollection(self.dryland_labels)
        uncultivated = ee.FeatureCollection(self.uncult_labels).merge(ee.FeatureCollection(self.wetland_labels))

        class_labels = class_labels.paint(uncultivated, 5)
        class_labels = class_labels.paint(dryland, 4)

        class_labels = class_labels.paint(flood, 1)
        class_labels = class_labels.paint(sprinkler, 2)
        class_labels = class_labels.paint(pivot, 3)

        return class_labels

    def _get_bucket_contents(self):
        dct = {}
        client = storage.Client()
        blob_list = client.list_blobs(self.gcs_bucket)
        for blob in blob_list:
            dirname = os.path.dirname(blob.name)
            b_name = os.path.basename(blob.name)
            if dirname not in dct.keys():
                dct[dirname] = [b_name]
            else:
                dct[dirname].append(b_name)
        l = [[x.split('.')[0] for x in dct[k] if x.endswith('.tif')] for k, v in dct.items()]
        l = [item for sublist in l for item in sublist]
        return l

    def _build_image(self, roi, start, end):

        naip = ee.ImageCollection('USDA/NAIP/DOQQ').filterDate(start, end).mosaic()

        if self.satellite == 'sentinel':
            sent = sentinel_composite(2019, roi)
            naip = naip.addBands(sent)

        if self.satellite == 'landsat':
            lst = landsat_composite(start, end, roi)
            naip = naip.addBands(lst)

        band_names = naip.bandNames().getInfo()

        return naip, band_names

    def _build_data(self):

        roi = ee.FeatureCollection(self.bounds).geometry()

        if self.fid:
            grid_fc = ee.FeatureCollection(self.grid).filter(ee.Filter.eq('FID', self.fid))
        else:
            grid_fc = ee.FeatureCollection(self.grid).filter(ee.Filter.eq('SPLIT', self.split))

        self.grid_fc = grid_fc.toList(grid_fc.size())
        image_stack, features = self._build_image(roi, start=self.start, end=self.end)
        i_labels = self._build_labels()
        self.image_stack = ee.Image.cat([image_stack, i_labels]).float()
        self.features = features + ['itype']

    def _start_task(self):
        try:
            self.task.start()
        except ee.ee_exception.EEException as e:
            print('waiting 50 minutes to export')
            print('{}'.format(e))
            time.sleep(3000)
            self.task.start()


if __name__ == '__main__':
    for split in ['valid']:
        stack = ITypeDataStack(2019, split=split, satellite='sentinel', dataset='wa')
        stack.export_geotiff(overwrite=False)
# ========================= EOF ====================================================================
