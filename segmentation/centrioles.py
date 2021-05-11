import math
from math import sqrt

import skimage.feature as feature
import skimage.transform as tf


def centrioles(image, min_size=0.2, max_size=0.5, threshold=0.1):
    # FIXME: change transform for simple rr=y cc=x swap
    blobs_log = feature.blob_log(image, min_sigma=min_size, max_sigma=max_size, num_sigma=10, threshold=threshold)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    tform = tf.SimilarityTransform(rotation=math.pi / 2)
    blobs_log[:, 0:2] = tform(blobs_log[:, 0:2])
    blobs_log[:, 0] *= -1

    return blobs_log
