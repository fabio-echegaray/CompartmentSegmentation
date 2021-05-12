from collections import Counter

import numpy as np
import javabridge
import matplotlib.pyplot as plt
import seaborn as sns

from cached import CachedImageFile, cached_step
from render import render_polygon
from segmentation.compartments import segment_zstack
from logger import get_logger

log = get_logger(name='__main__')

if __name__ == "__main__":
    # open file and select timeseries 4
    path = "/Volumes/AYDOGAN - DROPBOX/Cycles/" \
           "Sas6-GFP_Sqh-mCh_WT_PE_20210211/" \
           "Sas6-GFP_Sqh-mCh_WT_PE_20210211.mvd2"

    img_struc = CachedImageFile(path, image_series=3)

    comps_df = cached_step(f"c{0}t{8}-segmentation-dataframe.obj",
                           segment_zstack, img_struc, frame=8,
                           cache_folder=img_struc.cache_path)

    log.info("Computing geometric features.")
    comps_df.loc[:, 'area'] = comps_df['boundary'].apply(lambda c: c.area)
    comps_df.loc[:, 'radius'] = comps_df['area'].apply(lambda a: np.sqrt(a) / np.pi)
    number_of_concentric_polygons = Counter(comps_df['concentric'])
    comps_df = comps_df[comps_df['concentric'].isin([k for k, v in number_of_concentric_polygons.items() if v > 3])]
    comps_df = comps_df[(comps_df['area'] > 500) & (comps_df['area'] < 10e4)]
    print(comps_df)
    print(number_of_concentric_polygons)

    # --------------------------------------
    #  Plot
    # --------------------------------------
    log.info("First plot")

    offsets = sorted(comps_df['offset'].unique())
    level_palette = sns.color_palette("viridis", n_colors=len(offsets))
    offset_map = {o: k for k, o in enumerate(offsets)}


    def segmentations(*args, **kwargs):
        ax = plt.gca()
        data = kwargs.pop("data")
        mdi = img_struc.image(img_struc.ix_at(c=0, z=data['z'].iloc[0], t=8))
        ax.imshow(mdi.image, cmap='gray')
        # n_um = affinity.scale(selected_nucleus, xfact=me.um_per_pix, yfact=me.um_per_pix, origin=(0, 0, 0))

        for ix, c in data.iterrows():
            polygon = c['boundary']
            render_polygon(polygon, c=level_palette[offset_map[c['offset']]], zorder=100, ax=ax)

        # # plot segmentations with the minimum radius of each concentric group
        # for cn in data['concentric'].unique():
        #     df = comps_df.loc[comps_df['concentric'] == cn]
        #     pol = df[df['area'] == min(df['area'])]['boundary'].iloc[0]
        #     render_polygon(pol, zorder=100, ax=ax)
        #     ax.text(pol.centroid.x, pol.centroid.y, int(cn), zorder=10, fontsize=7,
        #             bbox={'facecolor': 'white', 'linewidth': 0, 'alpha': 0.5, 'pad': 2})


    g = sns.FacetGrid(data=comps_df,
                      row='z',
                      # ylim=[0, 50],
                      height=3, aspect=1.65,
                      despine=True, margin_titles=True,
                      gridspec_kws={"wspace": 0.4}
                      )
    g.map_dataframe(segmentations)
    g.savefig("slices.pdf")

    # --------------------------------------
    #  Finish
    # --------------------------------------
    javabridge.kill_vm()
