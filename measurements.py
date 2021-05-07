import itertools
import math
from math import sqrt

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skimage.draw as draw
import skimage.feature as feature
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology
import skimage.segmentation as segmentation
import skimage.transform as tf
from shapely.geometry import LineString, Polygon
from scipy.ndimage.morphology import distance_transform_edt

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


def nuclei_segmentation(image, compute_distance=False, radius=10, simp_px=None):
    # apply threshold
    log.debug('Thresholding images.')
    thresh_val = filters.threshold_otsu(image)
    thresh = image >= thresh_val
    thresh = morphology.remove_small_holes(thresh)
    thresh = morphology.remove_small_objects(thresh)

    # remove artifacts connected to image border
    cleared = segmentation.clear_border(thresh)

    if len(cleared[cleared > 0]) == 0: return None, None

    if compute_distance:
        distance = distance_transform_edt(cleared)
        local_maxi = feature.peak_local_max(distance, indices=False, labels=cleared,
                                            min_distance=radius / 4, exclude_border=False)
        markers, num_features = ndi.label(local_maxi)
        if num_features == 0:
            log.info('No nuclei found in current stack.')
            return None, None

        labels = morphology.watershed(-distance, markers, watershed_line=True, mask=cleared)
    else:
        labels = cleared

    log.info('Storing nuclear features.')

    # store all contours found
    contours = measure.find_contours(labels, 0.9)

    _list = list()
    for k, contr in enumerate(contours):
        # as the find_contours function returns values in (row, column) form,
        # we need to flip the columns to match (x, y) = (col, row)
        pol = Polygon(np.fliplr(contr))
        if simp_px is not None:
            pol = (pol.buffer(simp_px, join_style=1)
                   # .simplify(simp_px / 10, preserve_topology=True)
                   .buffer(-simp_px, join_style=1)
                   )

        _list.append({
            'id':       k,
            'boundary': pol
            })

    return labels, _list


def centrosomes(image, min_size=0.2, max_size=0.5, threshold=0.1):
    # FIXME: change transform for simple rr=y cc=x swap
    blobs_log = feature.blob_log(image, min_sigma=min_size, max_sigma=max_size, num_sigma=10, threshold=threshold)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    tform = tf.SimilarityTransform(rotation=math.pi / 2)
    blobs_log[:, 0:2] = tform(blobs_log[:, 0:2])
    blobs_log[:, 0] *= -1

    return blobs_log


def exclude_contained(polygons):
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
