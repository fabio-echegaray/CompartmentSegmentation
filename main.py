import numpy as np
import javabridge

import pandas as pd

import scipy.stats as stats
import scipy.ndimage as ndi

import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns

from cached import CachedImageFile, cached_step
from logger import get_logger
from render import render_polygon
from segmentation.compartments import segment_compartments_from_holes
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

    cdir = img_struc.cache_path
    compartments = cached_step(f"z{img_md.z}c{img_md.channel}t{img_md.frame}-bags.obj",
                               segment_compartments_from_holes, img_md.image,
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

    # ax2.imshow(img_as_ubyte(rescale_intensity(embryo, out_range=(0, 1))), cmap='magma')
    # ax2.axis('off')
    # ax2.set_title('Embryo mask')

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
