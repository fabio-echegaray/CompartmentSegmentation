import numpy as np
import scipy.ndimage as ndi
import skimage.feature as feature
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology
import skimage.segmentation as segmentation
from shapely.geometry import Polygon
from scipy.ndimage.morphology import distance_transform_edt

from logger import get_logger

log = get_logger(name='segmentation-nucleus')


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
