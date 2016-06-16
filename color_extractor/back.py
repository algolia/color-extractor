import numpy as np
import skimage.filters as skf
import skimage.color as skc
import skimage.morphology as skm
from skimage.measure import label
from skimage.io import show, imshow

from .task import Task


class Back(Task):
    """
    Two algorithms are used together to separate background and foreground.
    One consider as background all pixel whose color is close to the pixels
    in the corners. This part is impacted by the `max_distance' and
    `use_lab' settings.
    The second one computes the edges of the image and uses a flood fill
    starting from all corners. This part is impacted by the `edge_thinning'
    and `blur_radius' settings.
    """
    def __init__(self, settings=None):
        """
        The possible settings are:
            - max_distance: The maximum distance for two colors to be
              considered closed. A higher value will yield to a more aggressive
              background removal.
              (default: 5)

            - use_lab: Whether to use the LAB color space to perform
              background removal. More expensive but closer to eye perception.
              (default: True)

            - edge_thinning: How much edges must be thinned in the resulting
              mask. `0' means no thinning at all, `-1` means considering
              edges as part of the foreground in the resulting mask.
              (default: 1)

            - blur_sigma: The sigma of the Gaussian blur used before applying
              edge detections. A higher value will yield to a more aggressive
              background removal.
              (default: 0.8)
        """
        if settings is None:
            settings = {}

        super(Back, self).__init__(settings)

        k = self._settings['edge_thinning']
        if k > 0:
            self._erode = skm.square(k, np.bool)

    def get(self, img):
        f = self._floodfill(img)
        g = self._global(img)
        m = f | g

        if np.count_nonzero(m) < 0.85 * m.size:
            return m
        if np.count_nonzero(g) < 0.85 * g.size:
            return g
        if np.count_nonzero(f) < 0.85 * f.size:
            return f

        return np.zeros_like(m)

    def _global(self, img):
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.bool)
        max_distance = self._settings['max_distance']

        if self._settings['use_lab']:
            img = skc.rgb2lab(img)

        corners = [(0, 0), (h - 1, 0), (0, w - 1), (h - 1, w - 1)]
        for color in (img[i, j] for i, j in corners):
            norm = np.sqrt(np.sum(np.square(img - color), 2))
            mask |= norm < max_distance

        return mask

    def _floodfill(self, img):
        back = 1. - img
        back = Back._sobel(back, self._settings['blur_sigma'])

        back = back > 0.05
        back = skm.skeletonize(back)
        # imshow(back)
        # show()
        labels = label(back, background=-1, connectivity=1)

        h, w = back.shape[:2]
        corners = [(0, 0), (h - 1, 0), (0, w - 1), (h - 1, w - 1)]
        flooded = back.copy()
        for l in (labels[i, j] for i, j in corners):
            flooded[labels == l] = True

        flooded[back] = True

        return flooded

    @staticmethod
    def _default_settings():
        return {
            'max_distance': 5,
            'use_lab': True,
            'edge_thinning': 1,
            'blur_sigma': 0.8,
        }

    @staticmethod
    def _sobel(img, sigma):
        gray = skc.rgb2gray(img)
        return skf.sobel(gray)
