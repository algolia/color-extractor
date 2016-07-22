from os.path import basename, join, splitext

from skimage.io import imread, imsave
from skimage.util import img_as_float
from skimage.color import gray2rgb

from .image_to_color import ImageToColor
from .task import Task


class FromFile(Task):
    def __init__(self, samples, labels, settings=None):
        if settings is None:
            settings = {}

        super(FromFile, self).__init__(settings)
        self._image_to_color = ImageToColor(samples, labels, self._settings)

    def get(self, uri):
        i = imread(uri)
        if len(i.shape) == 2:
            i = gray2rgb(i)
        else:
            i = i[:, :, :3]
        c = self._image_to_color.get(i)

        dbg = self._settings['debug']
        if dbg is None:
            return c

        c, imgs = c
        b = splitext(basename(uri))[0]
        imsave(join(dbg, b + '-resized.jpg'), imgs['resized'])
        imsave(join(dbg, b + '-back.jpg'), img_as_float(imgs['back']))
        imsave(join(dbg, b + '-skin.jpg'), img_as_float(imgs['skin']))
        imsave(join(dbg, b + '-clusters.jpg'), imgs['clusters'])

        return c, {
            'resized': join(dbg, b + '-resized.jpg'),
            'back': join(dbg, b + '-back.jpg'),
            'skin': join(dbg, b + '-skin.jpg'),
            'clusters': join(dbg, b + '-clusters.jpg'),
        }

    @staticmethod
    def _default_settings():
        return {
            'debug': None,
        }
