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

GS_BUCKET = 'itype'

KERNEL_SIZE = 512
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
list_ = ee.List.repeat(1, KERNEL_SIZE)
lists = ee.List.repeat(list_, KERNEL_SIZE)
KERNEL = ee.Kernel.fixed(KERNEL_SIZE, KERNEL_SIZE, lists)


class ITypeDataStack(object):

    def __init__(self, year, split='train', satellite='landsat', fid=None):

        self.fid = fid
        self.year = year
        self.start, self.end = '{}-01-01'.format(year), '{}-12-31'.format(year)
        self.split = split

        self.bounds = 'users/dgketchum/boundaries/MT'

        self.grid = 'users/dgketchum/itype/mt_grid_{}'.format(year)
        self.irr_labels = 'users/dgketchum/itype/mt_itype_{}'.format(year)

        self.dryland_labels = 'users/dgketchum/itype/dryland'
        self.uncult_labels = 'users/dgketchum/itype/uncultivated'
        self.wetland_labels = 'users/dgketchum/itype/wetlands'
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
        if not overwrite:
            bucket_contents = self._get_bucket_contents()
        for idx in range(self.grid_fc.size().getInfo()):
            patch = ee.Feature(self.grid_fc.get(idx))
            fid = patch.getInfo()['properties']['FID']
            name_ = '{}_{}'.format(self.split, str(fid).rjust(7, '0'))
            if not overwrite:
                if '{}.tif'.format(name_) in bucket_contents:
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
        for blob in client.list_blobs(self.gcs_bucket):
            dirname = os.path.dirname(blob.name)
            b_name = os.path.basename(blob.name)
            if dirname not in dct.keys():
                dct[dirname] = [b_name]
            else:
                dct[dirname].append(b_name)
        l = [x.split('.')[0] for x in dct['']]
        return l

    def _build_image(self, roi, start, end):

        naip = ee.ImageCollection('USDA/NAIP/DOQQ').filterDate(start, end).mosaic()

        if self.satellite == 'sentinel':
            sent = sentinel_composite(start, end, roi)
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
        except ee.ee_exception.EEException:
            print('waiting 50 minutes to export')
            time.sleep(3000)
            self.task.start()


if __name__ == '__main__':
    for split in ['train']:
        stack = ITypeDataStack(2009, split=split, satellite='landsat')
        stack.export_geotiff(overwrite=True)
# ========================= EOF ====================================================================
