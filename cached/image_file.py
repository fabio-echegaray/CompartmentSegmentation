import os
from collections import namedtuple

import numpy as np
from xml.etree import ElementTree as ET

from image_loader import load_tiff
from logger import get_logger

MetadataImage = namedtuple('MetadataImage', ['image', 'pix_per_um', 'um_per_pix',
                                             'time_interval', 'frame', 'channel',
                                             'z', 'width', 'height',
                                             'timestamp', 'intensity_range'])


def ensure_dir(file_path):
    file_path = os.path.abspath(file_path)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return file_path


class CachedImageFile:
    ome_ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    log = get_logger(name='CachedImageFile')

    def __init__(self, image_path: str, image_series=0):
        self.image_path = os.path.abspath(image_path)
        self.base_path = os.path.dirname(self.image_path)
        self.cache_path = os.path.join(self.base_path, '_cache')
        self.render_path = ensure_dir(os.path.join(self.base_path, '_cache', 'out', 'render'))

        self._jvm_on = False

        self.metadata_path = os.path.join(self.cache_path, 'ome_image_info.xml')
        self.md = self._get_metadata()
        self.images_md = self.md.findall('ome:Image', self.ome_ns)[image_series]
        self.instrument_md = self.md.findall('ome:Instrument', self.ome_ns)
        self.planes_md = self.images_md.find('ome:Pixels', self.ome_ns)
        self.all_planes = self.images_md.findall('ome:Pixels/ome:Plane', self.ome_ns)

        self.timestamps = sorted(np.unique([p.get('DeltaT') for p in self.all_planes]).astype(np.float64))
        self.channels = sorted(np.unique([p.get('TheC') for p in self.all_planes]).astype(int))
        self.zstacks = sorted(np.unique([p.get('TheZ') for p in self.all_planes]).astype(int))
        self.frames = sorted(np.unique([p.get('TheT') for p in self.all_planes]).astype(int))
        self.um_per_pix = float(self.planes_md.get('PhysicalSizeX')) if \
            self.planes_md.get('PhysicalSizeX') == self.planes_md.get('PhysicalSizeY') else np.nan
        self.width = int(self.planes_md.get('SizeX'))
        self.height = int(self.planes_md.get('SizeY'))

        self.log.info(f"{len(self.all_planes)} image planes in total.")

    def _check_jvm(self):
        if not self._jvm_on:
            import javabridge
            import bioformats as bf

            javabridge.start_vm(class_path=bf.JARS, run_headless=True)

    def ix_at(self, c, z, t):
        for i, plane in enumerate(self.all_planes):
            if int(plane.get('TheC')) == c and int(plane.get('TheZ')) == z and int(plane.get('TheT')) == t:
                return i

    def image(self, *args):
        if len(args) == 1 and isinstance(args[0], int):
            ix = args[0]
            plane = self.all_planes[ix]
            # return self._image(c=plane.get('TheC'), z=plane.get('TheZ'), t=plane.get('TheT'), row=0, col=0, fid=0)
            return self._image(plane)

    # def _image(self, c=0, z=0, t=0, row=0, col=0, fid=0):
    def _image(self, plane):
        c, z, t, row, col, fid = plane.get('TheC'), plane.get('TheZ'), plane.get('TheT'), 0, 0, 0
        # logger.debug('retrieving image id=%d row=%d col=%d fid=%d' % (_id, row, col, fid))
        # check if file is in cache
        fname = f"{row}{col}{fid}-{c}{z}{t}.tif"
        fpath = os.path.join(self.cache_path, fname)
        if os.path.exists(fpath):
            self.log.debug(f"Loading image {fname} from cache.")
            tiff = load_tiff(fpath)
            image = tiff.image[0]
        else:
            self.log.debug("Loading image from original file (starts JVM if not on).")
            import bioformats as bf
            from tifffile import imsave
            self._check_jvm()
            reader = bf.ImageReader(self.image_path, perform_init=True)
            image = reader.read(c=c, z=z, t=t, rescale=False)
            self.log.debug(f"Saving image {fname} in cache (path={fpath}).")
            imsave(fpath, np.array(image))

        return MetadataImage(image=image, pix_per_um=1. / self.um_per_pix, um_per_pix=self.um_per_pix,
                             time_interval=None, timestamp=float(plane.get('DeltaT')),
                             frame=int(t), channel=int(c), z=int(z),
                             width=int(self.planes_md.get('SizeX')), height=int(self.planes_md.get('SizeY')),
                             intensity_range=[np.min(image), np.max(image)])

    def _get_metadata(self):
        self._check_jvm()

        self.log.debug(f"metadata_path is {self.metadata_path}.")
        if not os.path.exists(self.metadata_path):
            self.log.warning("File ome_image_info.xml is missing in the folder structure, generating it now.\r\n"
                             "\tNew folders with the names '_cache' will be created. "
                             "You can safely delete this folder if you don't want \n\n"
                             "any of the analysis output from this tool.\r\n")
            ensure_dir(self.metadata_path)
            ensure_dir(self.render_path)

            import bioformats as bf
            md = bf.get_omexml_metadata(self.image_path)
            with open(self.metadata_path, 'w') as mdf:
                mdf.write(md)

            md = ET.fromstring(md.encode("utf-8"))

        else:
            self.log.debug("Loading metadata from file in cache.")
            with open(self.metadata_path, 'r') as mdf:
                md = ET.parse(mdf)

        return md
