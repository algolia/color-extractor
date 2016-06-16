import numpy as np
import skimage.morphology as skm
from skimage.filters import gaussian
from skimage.color import rgb2hsv
from skimage.util import img_as_float

from .task import Task


class Skin(Task):
    def __init__(self, settings=None):
        """
        Skin is detected using color ranges.

        The possible settings are:
            - skin_type: The type of skin most expected in the given images.
              The value can be 'general' or 'none'. If 'none' is given the
              an empty mask is returned.
              (default: 'general')
        """
        if settings is None:
            settings = {}

        super(Skin, self).__init__(settings)
        self._k = skm.disk(1, np.bool)

        t = self._settings['skin_type']
        if t == 'general':
            self._lo = np.array([0, 0.19, 0.31], np.float64)
            self._up = np.array([0.1, 1., 1.], np.float64)
        elif t != 'none':
            raise NotImplementedError('Only general type is implemented')

    def get(self, img):
        t = self._settings['skin_type']
        if t == 'general':
            img = rgb2hsv(img)
        elif t == 'none':
            return np.zeros(img.shape[:2], np.bool)
        else:
            raise NotImplementedError('Only general type is implemented')

        return self._range_mask(img)

    def _range_mask(self, img):
        mask = np.all((img >= self._lo) & (img <= self._up), axis=2)

        # Smooth the mask.
        skm.binary_opening(mask, selem=self._k, out=mask)
        return gaussian(mask, 0.8, multichannel=True) != 0

    @staticmethod
    def _default_settings():
        return {
            'skin_type': 'general',
        }
