import ee


def sentinel_composite(yr, roi):

    spring_s, spring_e = '{}-03-01'.format(yr), '{}-05-01'.format(yr),
    late_spring_s, late_spring_e = '{}-05-01'.format(yr), '{}-07-01'.format(yr)
    summer_s, summer_e = '{}-07-01'.format(yr), '{}-09-01'.format(yr)
    fall_s, fall_e = '{}-09-01'.format(yr), '{}-12-31'.format(yr)

    periods = [(spring_s, spring_e),
               (late_spring_s, late_spring_e),
               (summer_s, summer_e),
               (fall_s, fall_e)]

    def mask(x):
        qa = x.select(['QA60'])
        x = x.updateMask(qa.lt(1))
        return x

    def get_period(start, end, idx):
        sent = ee.ImageCollection('COPERNICUS/S2').filterDate(start, end).filterBounds(roi)
        sent = sent.map(lambda x: mask(x))
        ndvi = sent.map(lambda x: x.addBands(x.normalizedDifference(['B8', 'B4'])))
        max_ndvi = ndvi.select('nd').reduce(ee.Reducer.mean())
        max_ndvi = max_ndvi.rename('ndvi_{}'.format(idx))
        return max_ndvi

    first = True
    for i, (start, end) in enumerate(periods):
        bands = get_period(start, end, i)
        if first:
            input_bands = bands
            proj = bands.select('ndvi_0').projection().getInfo()
            first = False
        else:
            input_bands = input_bands.addBands(bands)

    return input_bands,

if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
