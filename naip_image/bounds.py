# =============================================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2 (the "License");
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
# =============================================================================================

from math import floor

from pyproj import Proj


class BBox(object):
    def __init__(self):
        self.west = None
        self.east = None
        self.north = None
        self.south = None

    def to_geographic(self, epsg):
        in_proj = Proj({'init': 'epsg:{}'.format(epsg)})
        w, s = in_proj(self.west, self.south, inverse=True)
        e, n = in_proj(self.east, self.north, inverse=True)
        return w, s, e, n

    def geographic_to_utm_zone(self, lat, lon):
        utm_band = str((floor((lon + 180) / 6) % 60) + 1)
        if len(utm_band) == 1:
            utm_band = '0' + utm_band
        if lat >= 0:
            epsg_code = '326' + utm_band
        else:
            epsg_code = '327' + utm_band

        return epsg_code


class BufferPoint(BBox):
    def __init__(self):
        BBox.__init__(self)

    def buffer_meters(self, lat, lon, distance):
        y, x = lat, lon
        self.west = x - distance
        self.south = y - distance
        self.east = x + distance
        self.north = y + distance
        return self.west, self.south, self.east, self.north


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
