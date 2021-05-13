import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from shapely.geometry import Polygon
from numpy import asarray, concatenate, ones

matplotlib.rcParams['hatch.linewidth'] = 0.1

from logger import get_logger

log = get_logger(name='render')


def render_polygon(polygon: Polygon, zorder=0, c='red', ax=None, alpha=1.0, draw_hatch=True):
    """
        These next two functions are taken from Sean Gillies
        https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html
    """

    def ring_coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(ob.coords)
        codes = ones(n, dtype=Path.code_type) * Path.LINETO
        codes[0] = Path.MOVETO
        return codes

    def pathify(polygon):
        # Convert coordinates to path vertices. Objects produced by Shapely's
        # analytic methods have the proper coordinate order, no need to sort.
        vertices = concatenate(
            [asarray(polygon.exterior)]
            + [asarray(r) for r in polygon.interiors])
        codes = concatenate(
            [ring_coding(polygon.exterior)]
            + [ring_coding(r) for r in polygon.interiors])
        return Path(vertices, codes)

    if ax is None:
        ax = plt.gca()

    x, y = polygon.exterior.xy
    ax.plot(x, y, color=c, linestyle="--", linewidth=0.5, solid_capstyle='round', zorder=zorder, alpha=alpha)

    if draw_hatch:
        path = pathify(polygon)
        patch = PathPatch(path, facecolor='none', edgecolor=c, hatch='/////', lw=0.01, zorder=zorder, alpha=alpha)

        ax.add_patch(patch)

    ax.set_aspect(1.0)

    return
