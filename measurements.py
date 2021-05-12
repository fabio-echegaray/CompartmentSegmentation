import itertools

import numpy as np
import pandas as pd
import skimage.draw as draw
from shapely.geometry import LineString, Polygon

from logger import get_logger

log = get_logger(name='measurements')


class MeasurementException(Exception):
    pass


def vector_column_to_long_fmt(a, val_col, ix_col):
    # transform signal and domain vectors into long format (see https://stackoverflow.com/questions/27263805
    b = pd.DataFrame({
        col: pd.Series(data=np.repeat(a[col].values, a[val_col].str.len()))
        for col in a.columns.drop([val_col, ix_col])}
        ).assign(**{ix_col: np.concatenate(a[ix_col].values), val_col: np.concatenate(a[val_col].values)})[a.columns]
    return b


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def integral_over_surface(image, polygon: Polygon):
    assert polygon.is_valid, "Polygon is invalid"

    try:
        c, r = polygon.exterior.xy
        rr, cc = draw.polygon(r, c)
        ss = np.sum(image[rr, cc])
        for interior in polygon.interiors:
            c, r = interior.xy
            rr, cc = draw.polygon(r, c)
            ss -= np.sum(image[rr, cc])
        return ss
    except MeasurementException:
        log.warning('There was a problem in function integral_over_surface.')
        return np.nan


def histogram_of_surface(image, polygon: Polygon, bins=None):
    assert polygon.is_valid, "Polygon is invalid."

    try:
        hh = np.zeros(shape=image.shape, dtype=np.bool)
        c, r = polygon.exterior.xy
        rr, cc = draw.polygon(r, c)
        hh[rr, cc] = True
        for interior in polygon.interiors:
            c, r = interior.xy
            rr, cc = draw.polygon(r, c)
            hh[rr, cc] = False
        hist, edges = np.histogram(image[hh].ravel(), bins)
        return hist, edges
    except MeasurementException:
        log.warning('There was a problem in function histogram_of_surface.')
        return np.nan, np.nan


def integral_over_line(image, line: LineString):
    assert line.is_valid, "LineString is invalid"
    try:
        for pt0, pt1 in pairwise(line.coords):
            r0, c0, r1, c1 = np.array(list(pt0) + list(pt1)).astype(int)
            rr, cc = draw.line(r0, c0, r1, c1)
            ss = np.sum(image[rr, cc])
            return ss
    except MeasurementException:
        log.warning('There was a problem in function integral_over_line.')
        return np.nan


def generate_mask_from(polygon: Polygon, shape=None):
    if shape is None:
        minx, miny, maxx, maxy = polygon.bounds
        image = np.zeros((maxx - minx, maxy - miny), dtype=np.bool)
    else:
        image = np.zeros(shape, dtype=np.bool)

    c, r = polygon.boundary.xy
    rr, cc = draw.polygon(r, c)
    image[rr, cc] = True
    return image


def exclude_contained(polygons: pd.DataFrame):
    if polygons is None: return []
    for p in polygons:
        p['valid'] = True
    for p1, p2 in itertools.combinations(polygons, 2):
        if not p1['valid'] or not p2['valid']: continue
        if p1['boundary'].contains(p2['boundary']):
            p2['valid'] = False
        if p2['boundary'].contains(p1['boundary']):
            p1['valid'] = False
    return [p for p in polygons if p['valid']]


def simple_polygon(polygons: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts polygons that don't cross themselves
    """
    ix = polygons['boundary'].apply(lambda b: b.is_simple)
    return polygons.loc[ix, :]


def concentric(polygons: pd.DataFrame) -> pd.DataFrame:
    """
    Label polygons that don't touch and have a common centre
    """
    log.debug("Labelling concentric polygons.")
    p = polygons
    p = p.assign(concentric=0)
    con_id = 1
    for ix1, ix2 in itertools.combinations(p.index, 2):
        if p.loc[ix1, 'boundary'].contains(p.loc[ix2, 'boundary']):
            if p.loc[ix2, 'concentric'] == 0:
                p.loc[ix2, 'concentric'] = con_id
                con_id += 1
            p.loc[ix1, 'concentric'] = p.loc[ix2, 'concentric']
        if p.loc[ix2, 'boundary'].contains(p.loc[ix1, 'boundary']):
            if p.loc[ix1, 'concentric'] == 0:
                p.loc[ix1, 'concentric'] = con_id
                con_id += 1
            p.loc[ix2, 'concentric'] = p.loc[ix1, 'concentric']

    return p[p['concentric'] > 0]
