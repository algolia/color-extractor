import cv2
import numpy as np

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
        self._k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        t = self._settings['skin_type']
        if t == 'general':
            self._lo = np.array([0, 48, 80], np.uint8)
            self._up = np.array([20, 255, 255], np.uint8)
        elif t != 'none':
            raise NotImplementedError('Only general type is implemented')

    def get(self, img):
        t = self._settings['skin_type']
        if t == 'general':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif t == 'none':
            return np.zeros(img.shape[:2], np.bool)
        else:
            raise NotImplementedError('Only general type is implemented')

        return self._range_mask(img)

    def _range_mask(self, img):
        mask = cv2.inRange(img, self._lo, self._up)

        # Smooth the mask.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._k)
        return cv2.GaussianBlur(mask, (3, 3), 0) != 0

    @staticmethod
    def _default_settings():
        return {
            'skin_type': 'general',
        }
