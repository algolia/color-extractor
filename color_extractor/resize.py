import numpy as np
from skimage.transform import resize

from .task import Task


class Resize(Task):
    """
    Resizes and crops given images to the specified shape. As most fashion
    have the subject centered, cropping may help reducing the background
    and help discarding background from foreground.
    Note that the background detection algorithm relies heavily on the
    corners, if the cropping is too important, the object itself may be
    disregarded.
    """
    def __init__(self, settings=None):
        """
        The possible settings are:
            - crop: The crop ratio to use. `1' means no cropping. A floating
              point number between `0' and `1' is expected.
              (default: 0.90)

            - shape: The height of the resized image. The ratio between height
              and width is kept.
              (default: 100)
        """
        if settings is None:
            settings = {}

        super(Resize, self).__init__(settings)

    def get(self, img):
        """Returns `img` cropped and resized."""
        return self._resize(self._crop(img))

    def _resize(self, img):
        src_h, src_w = img.shape[:2]
        dst_h = self._settings['rows']
        dst_w = int((dst_h / src_h) * src_w)
        return resize(img, (dst_h, dst_w))

    def _crop(self, img):
        src_h, src_w = img.shape[:2]
        c = self._settings['crop']
        dst_h, dst_w = int(src_h * c), int(src_w * c)
        rm_h, rm_w = (src_h - dst_h) // 2, (src_w - dst_w) // 2
        return img[rm_h:rm_h + dst_h, rm_w:rm_w + dst_w].copy()

    @staticmethod
    def _default_settings():
        return {
            'crop': 0.90,
            'rows': 100,
        }
