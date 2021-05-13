# import shapely.wkt
import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(name='filters')


def nucleus(df: pd.DataFrame, nucleus_col='nucleus', radius_min=0, radius_max=np.inf):
    """
    Returns a dataframe with all samples with nucleus area greater than π*radius^2.

    :param df: Input dataframe.
    :param nucleus_col: Column name for nucleus polygon data.
    :param radius_min: Minimum radius of disk in um for area comparison. Areas lesser than this area are discarded.
    :param radius_max: Areas greater than the equivalent disk area of radius radius_max are discarded.
    :return:  Filtered dataframe.
    """
    area_min_thresh = np.pi * radius_min ** 2
    area_max_thresh = np.pi * radius_max ** 2
    log.info("Filtering nuclei with area greater than %0.2f[pix^2] and less than %0.2f[pix^2]." % (
        area_min_thresh, area_max_thresh))
    n_idx = df.apply(lambda row: area_max_thresh > row[nucleus_col].area > area_min_thresh, axis=1)
    return df[n_idx]


def polsby_popper(df: pd.DataFrame, column, pp_threshold=0.8):
    def _pp(_df):
        # pol = shapely.wkt.loads(_df[column])
        pol = _df[column]
        pp = pol.area * np.pi * 4 / pol.length ** 2
        return pp > pp_threshold

    log.info("Filtering %s with a Polsby-Popper score greater than %0.2f." % (column, pp_threshold))
    n_idx = df.apply(_pp, axis=1)
    return df[n_idx]
