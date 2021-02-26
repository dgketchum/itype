import ee


def sentinel_composite(start, end, roi):
    def mask(x):
        qa = x.select(['QA60'])
        x = x.updateMask(qa.lt(1))
        return x

    sent = ee.ImageCollection('COPERNICUS/S2').filterDate(start, end).filterBounds(roi)
    sent = sent.map(lambda x: mask(x))
    ndvi = sent.map(lambda x: x.addBands(x.normalizedDifference(['B8', 'B4'])))
    std_ndvi = ndvi.select('nd').reduce(ee.Reducer.stdDev())
    max_ndvi = ndvi.select('nd').reduce(ee.Reducer.max())
    std_ndvi, max_ndvi = std_ndvi.rename('std_ndvi'), max_ndvi.rename('mx_ndvi')
    return [std_ndvi, max_ndvi]


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
