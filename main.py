import numpy as np
import javabridge
from collections import Counter

import pandas as pd

import scipy.stats as stats
import scipy.ndimage as ndi

import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns

from cached import CachedImageFile, cached_step
from logger import get_logger
from measurements import concentric, simple_polygon
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
    comps_df = pd.DataFrame(compartments)
    comps_df.loc[:, 'area'] = comps_df['boundary'].apply(lambda c: c.area)
    comps_df.loc[:, 'radius'] = comps_df['area'].apply(lambda a: np.sqrt(a) / np.pi)
    # comps_df.assign(area=lambda r: r['boundary'].area)
    comps_df = (comps_df
                .pipe(simple_polygon)
                .pipe(concentric)
                )
    number_of_concentric_polygons = Counter(comps_df['concentric'])
    comps_df = comps_df[comps_df['concentric'].isin([k for k, v in number_of_concentric_polygons.items() if v > 3])]
    print(comps_df)
    print(number_of_concentric_polygons)

    # compute properties of compartments
    areas = [c['boundary'].area for c in compartments]
    median = ndi.median(areas)
    percentile = stats.percentileofscore(areas, median)
    area_perc_value = np.percentile(areas, percentile)
    print(stats.describe(areas))
    print(f"median={median:0.3f} value={area_perc_value:0.3f} percentile={percentile:0.3f}")

    # Plot results
    log.info("First plot")
    fig = plt.figure(figsize=(3, 4), dpi=150)
    ax = fig.gca()
    ax.imshow(img_md.image, cmap='gray')
    # n_um = affinity.scale(selected_nucleus, xfact=me.um_per_pix, yfact=me.um_per_pix, origin=(0, 0, 0))
    offsets = sorted(comps_df['concentric'].unique())
    level_palette = sns.color_palette("hls", n_colors=len(offsets))
    offset_map = {o: k for k, o in enumerate(offsets)}
    for ix, c in comps_df.iterrows():
        polygon = c['boundary']
        if 2000 < polygon.area < 10e4:
            render_polygon(polygon, c=level_palette[offset_map[c['concentric']]], zorder=100, ax=ax)
    for cn in comps_df['concentric'].unique():
        pol = comps_df.loc[comps_df['concentric'] == cn].iloc[0]['boundary']
        ax.text(pol.centroid.x, pol.centroid.y, int(cn), zorder=10, fontsize=7,
                bbox={'facecolor': 'white', 'linewidth': 0, 'alpha': 0.5, 'pad': 2})
    sm = plt.cm.ScalarMappable(cmap=mpl.colors.ListedColormap(level_palette))
    cb1 = plt.colorbar(sm, ax=ax, boundaries=offsets, orientation='horizontal')

    ax.axis('off')
    ax.set_title('Myosin')
    fig.tight_layout()
    plt.savefig("segmentation.pdf")

    # --------------------------------------
    #  Next Plot
    # --------------------------------------
    log.info("Second plot")
    fig = plt.figure(figsize=(10, 3.2), dpi=150)
    ax = fig.gca()
    annotated_boxplot(comps_df[comps_df['area'] > 200], 'area', group='offset', ax=ax)

    ax.set_ylim([0, 10e3])
    ax.set_xlabel('offset level')
    ax.set_title('Blob area')
    fig.tight_layout()
    plt.savefig("compartments.pdf")

    # --------------------------------------
    #  Next Plot
    # --------------------------------------
    log.info("Third plot")
    fig = plt.figure(figsize=(10, 4), dpi=150)
    ax = fig.gca()
    comp_tidy = comps_df.melt(id_vars=['offset', 'concentric'], value_vars=['radius'])
    sns.scatterplot(x='offset', y='radius',
                    data=comps_df,
                    hue='concentric', style='concentric',
                    palette=sns.color_palette("hls", n_colors=len(offsets)))

    ax.set_ylim([0, 40])
    ax.set_xlabel('offset level')
    ax.set_title('Blob geometrical features')
    fig.tight_layout()
    plt.savefig("geometry_features.pdf")

    javabridge.kill_vm()
