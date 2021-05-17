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
from segmentation.compartments import segment_zstack, cluster_by_centroid
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
    comps_df = (comps_df
                # .pipe(lambda df: df[df['offset'] > 80])
                .pipe(lambda df: df[(df['area'] > 500) & (df['area'] < 10e4)])
                .pipe(polsby_popper, 'boundary', pp_threshold=0.7)
                .pipe(cluster_by_centroid, eps=0.13)
                .pipe(lambda df: df[df['cluster'] > 0])
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
    clustering_stats = comps_df.groupby(['cluster'])['offset'].describe()
    print("Cluster stats:\n", clustering_stats)
    clustering_stats.to_excel("cluster.xlsx")

    offsets = sorted(comps_df['offset'].unique())
    level_palette = sns.color_palette("viridis", n_colors=len(offsets))
    offset_map = {o: k for k, o in enumerate(offsets)}

    center_clusters = sorted(comps_df['cluster'].unique())
    center_palette = sns.color_palette("tab10", n_colors=len(center_clusters))
    center_map = {o: k for k, o in enumerate(center_clusters)}

    cluster_strength = Counter(comps_df['cluster'])
    cs_palette = sns.color_palette("rocket", n_colors=max(cluster_strength.values()))
    cs_map = {k: cluster_strength[k] - 1 for k in cluster_strength.keys()}

    # --------------------------------------
    #  Plot
    # --------------------------------------
    log.info("Plot of all level curves in one projected graph.")
    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax = fig.gca()
    for ix, c in comps_df.iterrows():
        polygon = c['boundary']
        render_polygon(polygon, c=level_palette[offset_map[c['offset']]], alpha=0.3, draw_hatch=False, zorder=100,
                       ax=ax)
        ax.plot(polygon.centroid.x, polygon.centroid.y,
                c=center_palette[center_map[c['cluster']]],
                marker='+', zorder=1000)
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


    def segmentations(ax, data, c=0, z=0, t=8):
        mdi = img_struc.image(img_struc.ix_at(c=c, z=z, t=t))
        ax.imshow(mdi.image, cmap='gray')
        # n_um = affinity.scale(selected_nucleus, xfact=me.um_per_pix, yfact=me.um_per_pix, origin=(0, 0, 0))

        for ix, c in data.iterrows():
            polygon = c['boundary']
            render_polygon(polygon, c=cs_palette[cs_map[c['cluster']]], zorder=100, ax=ax)
        for cn in data['cluster'].unique():
            df = data.loc[comps_df['cluster'] == cn]
            pol = df[df['area'] == min(df['area'])]['boundary'].iloc[0]
            ax.text(pol.centroid.x, pol.centroid.y, int(cn), c='white', zorder=110, fontsize=7,
                    # bbox={'facecolor': 'white', 'linewidth': 0, 'alpha': 0.5, 'pad': 2}
                    )


    print(img_struc.zstacks)
    fig, f_axes = plt.subplots(ncols=4, nrows=5, constrained_layout=True, figsize=(10, 12),
                               sharex=True, sharey=True,
                               subplot_kw=dict(aspect=1),
                               gridspec_kw=dict(wspace=0.01, hspace=0.01),
                               )
    for r, row in enumerate(f_axes):
        for c, ax in enumerate(row):
            z = r * 4 + c
            in_zstack = z in img_struc.zstacks
            # label = f'Col: {c}\nRow: {r}\n z={z} {"in zstack" if in_zstack else "not found"}'
            # ax.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center')
            if in_zstack:
                dat = comps_df.query("z == @z")
                segmentations(ax, dat, z=z)
    fig.tight_layout()
    fig.savefig("slices.png", dpi=300)

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
