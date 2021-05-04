import numpy as np
import matplotlib.pyplot as plt

import javabridge

from skimage.segmentation import random_walker
from skimage.exposure import rescale_intensity

from cached import CachedImageFile
from logger import get_logger

log = get_logger(name='__main__')

if __name__ == "__main__":
    # open file and select timeseries 4
    path = "/Volumes/AYDOGAN - DROPBOX/Cycles/" \
           "Sas6-GFP_Sqh-mCh_WT_PE_20210211/" \
           "Sas6-GFP_Sqh-mCh_WT_PE_20210211.mvd2"

    img_struc = CachedImageFile(path, image_series=3)

    # get the image based on the metadata given index
    ix = 300
    image = img_struc.image(ix)

    log.debug("Processing image.")
    sigma = 0.35
    data = rescale_intensity(1.0 * image, out_range=(-1, 1))
    # The range of the binary image spans over (-1, 1).
    # We choose the hottest and the coldest pixels as markers.
    markers = np.zeros(data.shape, dtype=np.uint)
    markers[data < np.percentile(data, 1)] = 1
    markers[data > np.percentile(data, 99)] = 2

    # Run random walker algorithm
    labels = random_walker(data, markers, beta=10, mode='bf')

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2), sharex=True, sharey=True)
    ax1.imshow(data, cmap='gray')
    ax1.axis('off')
    ax1.set_title('Noisy data')
    ax2.imshow(markers, cmap='magma')
    ax2.axis('off')
    ax2.set_title('Markers')
    ax3.imshow(labels, cmap='gray')
    ax3.axis('off')
    ax3.set_title('Segmentation')

    fig.tight_layout()
    plt.savefig("segmentation.png")
    plt.show()

    javabridge.kill_vm()
