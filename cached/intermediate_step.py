import os
import numpy as np
from tifffile import imsave

from image_loader import load_tiff
from logger import get_logger

log = get_logger(name='cached_ops')


def cached_step(filename, function, *args, cache_folder=None, override_cache=False, **kwargs):
    assert filename[-4:] == "tiff", "Function only supported for tiff images currently."
    cache_folder = os.path.abspath(".") if cache_folder is None else cache_folder
    output_path = os.path.join(cache_folder, filename)
    if not os.path.exists(output_path) or override_cache:
        log.debug(f"Generating data for step that calls function {function.__name__}.")
        out = function(*args, **kwargs)
        log.debug(f"Saving image {filename} in cache (path={output_path}).")
        imsave(output_path, np.array(out))
        return out
    else:
        log.debug(f"Loading image {filename} from cache (path={output_path}).")
        tiff = load_tiff(output_path)
        return tiff.image[0]
