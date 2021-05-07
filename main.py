import numpy as np
import javabridge

import pandas as pd

import scipy.stats as stats
import scipy.ndimage as ndi

from skimage import img_as_bool, img_as_float, img_as_ubyte
from skimage.segmentation import random_walker
from skimage.filters import threshold_local, threshold_otsu
from skimage.exposure import rescale_intensity
from skimage.morphology import disk, remove_small_holes
from skimage.measure import label, find_contours
from skimage.util import invert
from skimage.filters.rank import entropy
from skimage.color import label2rgb

from shapely.geometry import LineString, Polygon

import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns

from cached import CachedImageFile, cached_step
from logger import get_logger
from render import render_polygon
from tools import annotated_boxplot

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

        segmented_polygons = list()
        for offst in np.arange(start=1, stop=200, step=10):
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
    comp = pd.DataFrame(compartments)
    comp.loc[:, 'area'] = comp['boundary'].apply(lambda c: c.area)
    print(comp)

    # compute properties of compartments
    areas = [c['boundary'].area for c in compartments]
    median = ndi.median(areas)
    percentile = stats.percentileofscore(areas, median)
    area_perc_value = np.percentile(areas, percentile)
    print(stats.describe(areas))
    print(f"median={median:0.3f} value={area_perc_value:0.3f} percentile={percentile:0.3f}")

    # Plot results
    log.info("First plot")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2), dpi=150)  # , sharex=True, sharey=True)
    ax1.imshow(img_md.image, cmap='gray')
    # n_um = affinity.scale(selected_nucleus, xfact=me.um_per_pix, yfact=me.um_per_pix, origin=(0, 0, 0))
    offsets = sorted(comp['offset'].unique())
    level_palette = sns.color_palette("mako", n_colors=len(offsets))
    offset_map = {o: k for k, o in enumerate(offsets)}
    for c in compartments:
        polygon = c['boundary']
        if 2000 < polygon.area < 10e4:
            render_polygon(polygon, c=level_palette[offset_map[c['offset']]], zorder=100, ax=ax1)
        if 10e4 < polygon.area:
            render_polygon(polygon, c='red', zorder=100, ax=ax1)
    sm = plt.cm.ScalarMappable(cmap=mpl.colors.ListedColormap(level_palette))
    # sm.set_array([])
    cb1 = plt.colorbar(sm, ax=ax1, boundaries=offsets, orientation='horizontal')

    ax1.axis('off')
    ax1.set_title('Myosin')

    ax2.imshow(img_as_ubyte(rescale_intensity(embryo, out_range=(0, 1))), cmap='magma')
    ax2.axis('off')
    ax2.set_title('Embryo mask')

    # ax3.imshow(compartments, cmap='gray')
    ax3.axis('off')
    ax3.set_title('Compartments')

    fig.tight_layout()
    plt.savefig("segmentation.pdf")

    # --------------------------------------
    #  Next Image
    # --------------------------------------
    log.info("Second plot")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), dpi=150)  # , sharex=True, sharey=True)
    annotated_boxplot(comp[comp['area'] > 200], 'area', group='offset', ax=ax1)
    ax1.set_ylim([0, 10e3])
    ax1.set_xlabel('offset level')
    ax1.set_title('Blob area')

    fig.tight_layout()
    plt.savefig("compartments.pdf")

    javabridge.kill_vm()
