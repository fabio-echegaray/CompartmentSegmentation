import numpy as np
import javabridge

import scipy.stats as stats
import scipy.ndimage as ndi

from skimage import img_as_bool, img_as_float, img_as_ubyte
from skimage.segmentation import random_walker
from skimage.filters import threshold_local
from skimage.exposure import rescale_intensity
from skimage.morphology import disk, remove_small_holes
from skimage.measure import label, find_contours
from skimage.util import invert
from skimage.filters.rank import entropy
from skimage.color import label2rgb

from shapely.geometry import LineString, Polygon

import matplotlib.pyplot as plt

from cached import CachedImageFile, cached_step
from logger import get_logger
from render import render_polygon

log = get_logger(name='__main__')

if __name__ == "__main__":
    # open file and select timeseries 4
    path = "/Volumes/AYDOGAN - DROPBOX/Cycles/" \
           "Sas6-GFP_Sqh-mCh_WT_PE_20210211/" \
           "Sas6-GFP_Sqh-mCh_WT_PE_20210211.mvd2"

    img_struc = CachedImageFile(path, image_series=3)

    # get the image based on the metadata given index
    ix = img_struc.ix_at(c=0, z=14, t=8)
    img_md = img_struc.image(ix)

    log.debug("Processing image.")


    def segment_embryo(image):
        data = rescale_intensity(img_as_float(image), out_range=(-1, 1))
        # The range of the binary image spans over (-1, 1).
        # We choose the hottest and the coldest pixels as markers.
        markers = np.zeros(data.shape, dtype=np.uint)
        markers[data < np.percentile(data, 1)] = 1
        markers[data > np.percentile(data, 99)] = 2

        # Run random walker algorithm
        labels = random_walker(data, markers, beta=10, mode='bf')
        return labels.astype(np.uint8)


    def segment_compartments_from_holes(image, mask, radius=1):
        data = rescale_intensity(image, out_range=(0, np.iinfo(np.uint16).max))
        data = invert(data)
        data = img_as_ubyte(rescale_intensity(img_as_float(data), out_range=(0, 1)))

        entr_img = img_as_ubyte(rescale_intensity(entropy(data, disk(30)), out_range=(0, 1)))
        entr_img = invert(entr_img)
        local_thresh = threshold_local(entr_img, block_size=35, offset=100)

        label_image = label(img_as_bool(local_thresh))
        # image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
        log.info('Storing compartment features.')

        # store all contours found
        contours = find_contours(label_image, 0.9)

        segmented_polygons = list()
        for k, contr in enumerate(contours):
            # as the find_contours function returns values in (row, column) form,
            # we need to flip the columns to match (x, y) = (col, row)
            pol = Polygon(np.fliplr(contr))
            segmented_polygons.append({
                'id':       k,
                'boundary': pol
                })

        return segmented_polygons


    cdir = img_struc.cache_path
    embryo = cached_step(f"z{img_md.z}c{img_md.channel}t{img_md.frame}-seg-embryo.obj", segment_embryo, img_md.image,
                         cache_folder=cdir)
    print(np.unique(embryo))
    embryo[embryo <= 1] = 0
    embryo[embryo > 1] = 1
    embryo = remove_small_holes(embryo)
    compartments = cached_step(f"z{img_md.z}c{img_md.channel}t{img_md.frame}-bags.obj",
                               segment_compartments_from_holes, img_md.image, embryo, radius=20 * img_md.pix_per_um,
                               cache_folder=cdir, override_cache=True)

    # compute properties of compartments
    areas = [c['boundary'].area for c in compartments]
    median = ndi.median(areas)
    percentile = stats.percentileofscore(areas, median)
    area_perc_value = np.percentile(areas, percentile + 30)
    print(areas)
    print(stats.describe(areas))
    print(f"median={median} value={area_perc_value} percentile={percentile}")

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2))  # , sharex=True, sharey=True)
    ax1.imshow(img_md.image, cmap='gray')
    ax1.axis('off')
    ax1.set_title('Myosin')

    ax2.imshow(img_as_ubyte(rescale_intensity(embryo, out_range=(0, 1))), cmap='magma')
    # n_um = affinity.scale(selected_nucleus, xfact=me.um_per_pix, yfact=me.um_per_pix, origin=(0, 0, 0))
    for c in compartments:
        polygon = c['boundary']
        print(polygon.area)
        # if polygon.area < area_perc_value:
        if 2000 < polygon.area < 10e4:
            render_polygon(polygon, c='yellow', zorder=100, ax=ax1)
        if 10e4 < polygon.area:
            render_polygon(polygon, c='red', zorder=100, ax=ax1)
    ax2.axis('off')
    ax2.set_title('Embryo mask')

    # ax3.imshow(compartments, cmap='gray')
    ax3.boxplot(areas, showfliers=False)
    # ax3.axis('off')
    ax3.set_title('Compartments')

    fig.tight_layout()
    plt.savefig("segmentation.png")
    # plt.show()

    javabridge.kill_vm()
