import numpy as np
from skimage import img_as_bool, img_as_float, img_as_ubyte
from skimage.filters import threshold_local
from skimage.exposure import rescale_intensity
from skimage.morphology import disk
from skimage.measure import find_contours, label
from skimage.util import invert
from skimage.filters.rank import entropy
from shapely.geometry import Polygon

from logger import get_logger

log = get_logger(name='segmentation-compartment')


def segment_compartments_from_holes(image):
    data = rescale_intensity(image, out_range=(0, np.iinfo(np.uint16).max))
    data = invert(data)
    data = img_as_ubyte(rescale_intensity(img_as_float(data), out_range=(0, 1)))

    entr_img = img_as_ubyte(rescale_intensity(entropy(data, disk(30)), out_range=(0, 1)))
    entr_img = invert(entr_img)

    segmented_polygons = list()
    for offst in np.arange(start=70, stop=300, step=10):
        local_thresh = threshold_local(entr_img, block_size=35, offset=offst)
        binary_local = img_as_bool(local_thresh)
        label_image = label(binary_local)

        # store all contours found
        contours = find_contours(label_image, 0.9)
        log.debug(f"Number of blobs found at offset {offst} ={len(contours)}. "
                  f"Local threshold stats: min={np.min(local_thresh):4.1f} max={np.max(local_thresh):4.1f}")

        for contr in contours:
            if len(contr) < 3:
                continue
            # as the find_contours function returns values in (row, column) form,
            # we need to flip the columns to match (x, y) = (col, row)
            pol = Polygon(np.fliplr(contr))
            segmented_polygons.append({
                'offset':   offst,
                'boundary': pol
                })

    return segmented_polygons
