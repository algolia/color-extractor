import numpy as np
import skimage.filters as skf
import skimage.color as skc
import skimage.morphology as skm
from skimage.measure import label

from .task import Task


class Back(Task):
    """
    Two algorithms are used together to separate background and foreground.
    One consider as background all pixel whose color is close to the pixels
    in the corners. This part is impacted by the `max_distance' and
    `use_lab' settings.
    The second one computes the edges of the image and uses a flood fill
    starting from all corners.
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
        """
        if settings is None:
            settings = {}

        super(Back, self).__init__(settings)

    def get(self, img):
        f = self._floodfill(img)
        g = self._global(img)
        m = f | g

        if np.count_nonzero(m) < 0.90 * m.size:
            return m

        ng = np.count_nonzero(g)
        nf = np.count_nonzero(f)

        if ng < 0.90 * g.size and nf < 0.90 * f.size:
            return g if ng > nf else f

        if ng < 0.90 * g.size:
            return g

        if nf < 0.90 * f.size:
            return f

        return np.zeros_like(m)

    def _global(self, img):
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.bool)
        max_distance = self._settings['max_distance']

        if self._settings['use_lab']:
            img = skc.rgb2lab(img)

        # Compute euclidean distance of each corner against all other pixels.
        corners = [(0, 0), (-1, 0), (0, -1), (-1, -1)]
        for color in (img[i, j] for i, j in corners):
            norm = np.sqrt(np.sum(np.square(img - color), 2))
            # Add to the mask pixels close to one of the corners.
            mask |= norm < max_distance

        return mask

    def _floodfill(self, img):
        back = Back._scharr(img)
        # Binary thresholding.
        back = back > 0.05

        # Thin all edges to be 1-pixel wide.
        back = skm.skeletonize(back)

        # Edges are not detected on the borders, make artificial ones.
        back[0, :] = back[-1, :] = True
        back[:, 0] = back[:, -1] = True

        # Label adjacent pixels of the same color.
        labels = label(back, background=-1, connectivity=1)

        # Count as background all pixels labeled like one of the corners.
        corners = [(1, 1), (-2, 1), (1, -2), (-2, -2)]
        for l in (labels[i, j] for i, j in corners):
            back[labels == l] = True

        # Remove remaining inner edges.
        return skm.opening(back)

    @staticmethod
    def _default_settings():
        return {
            'max_distance': 5,
            'use_lab': True,
        }

    @staticmethod
    def _scharr(img):
        # Invert the image to ease edge detection.
        img = 1. - img
        grey = skc.rgb2grey(img)
        return skf.scharr(grey)
