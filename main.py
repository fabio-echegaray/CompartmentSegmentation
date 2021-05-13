from collections import Counter

import numpy as np
import javabridge
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns

from cached import CachedImageFile, cached_step
from filters import polsby_popper
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
    # filter by number of concentric polygons of each subset
    print([i for i, r in comps_df.groupby(['z', 'concentric']).count().iterrows()])
    comps_df = comps_df[comps_df[['z', 'concentric']].apply(tuple, axis=1).isin(
        [i for i, r in comps_df.groupby(['z', 'concentric']).count().iterrows() if r['offset'] > 10]
        )]
    comps_df = (comps_df
                # .pipe(lambda df: df[df['offset'] > 80])
                .pipe(lambda df: df[(df['area'] > 500) & (df['area'] < 10e4)])
                .pipe(polsby_popper, 'boundary', pp_threshold=0.7)
                )

    # --------------------------------------
    #  Some stats
    # --------------------------------------
    print(comps_df)
    print("Number of polygons per concentric cluster:\n",
          comps_df.groupby(['z', 'concentric']).count()['offset'].describe())
    sns.histplot(comps_df.groupby(['z', 'concentric']).count()['offset'])
    plt.xlabel('Number of polygons in the concentric subsets')
    plt.savefig('concentric-histo.pdf')
    offset_stats = comps_df.groupby(['z', 'concentric'])['offset'].describe()
    print("Offset stats per z and concentric cluster:\n", offset_stats)
    offset_stats.to_excel("offset.xlsx")

    offsets = sorted(comps_df['offset'].unique())
    level_palette = sns.color_palette("viridis", n_colors=len(offsets))
    offset_map = {o: k for k, o in enumerate(offsets)}

    # --------------------------------------
    #  Plot
    # --------------------------------------
    log.info("Plot of all level curves in one projected graph.")
    fig = plt.figure(figsize=(4, 3), dpi=150)
    ax = fig.gca()
    for ix, c in comps_df.iterrows():
        polygon = c['boundary']
        render_polygon(polygon, c=level_palette[offset_map[c['offset']]], alpha=0.3, draw_hatch=False, zorder=100,
                       ax=ax)
        ax.plot(polygon.centroid.x, polygon.centroid.y, c='k', marker='+', zorder=1000)
    sm = plt.cm.ScalarMappable(cmap=mpl.colors.ListedColormap(level_palette))
    cb1 = plt.colorbar(sm, ax=ax, boundaries=offsets, orientation='vertical')

    ax.axis('off')
    ax.set_title('Myosin')
    fig.tight_layout()
    fig.savefig("concentric-all-z.png")

    # --------------------------------------
    #  Plot
    # --------------------------------------
    log.info("Plot of all the segmentations, per offset level, separated by z-stack value.")


    def segmentations(*args, **kwargs):
        ax = plt.gca()
        data = kwargs.pop("data")
        mdi = img_struc.image(img_struc.ix_at(c=0, z=data['z'].iloc[0], t=8))
        ax.imshow(mdi.image, cmap='gray')
        # n_um = affinity.scale(selected_nucleus, xfact=me.um_per_pix, yfact=me.um_per_pix, origin=(0, 0, 0))

        for ix, c in data.iterrows():
            polygon = c['boundary']
            render_polygon(polygon, c=level_palette[offset_map[c['offset']]], zorder=100, ax=ax)
        for cn in data['concentric'].unique():
            df = data.loc[comps_df['concentric'] == cn]
            pol = df[df['area'] == min(df['area'])]['boundary'].iloc[0]
            ax.text(pol.centroid.x, pol.centroid.y, int(cn), c='white', zorder=10, fontsize=7,
                    # bbox={'facecolor': 'white', 'linewidth': 0, 'alpha': 0.5, 'pad': 2}
                    )


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
    #  Plot
    # --------------------------------------
    log.info("Plot of the colorbar.")
    fig = plt.figure(figsize=(3, 2), dpi=150)
    ax = fig.gca()
    sm = plt.cm.ScalarMappable(cmap=mpl.colors.ListedColormap(level_palette))
    plt.colorbar(sm, ax=ax, boundaries=offsets, orientation='horizontal')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig("colorbar.pdf")

    # --------------------------------------
    #  Finish
    # --------------------------------------
    javabridge.kill_vm()
