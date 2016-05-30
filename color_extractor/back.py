import cv2
import numpy as np

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
    def __init__(self, settings={}):
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
              mask. `0' or `1' means no thinning at all, `-1` means considering
              edges as part of the foreground in the resulting mask.
              (default: 3)

            - blur_radius: The radius of the Gaussian blur used before applying
              edge detections. A higher value will yield to a more aggressive
              background removal. This setting must be an odd integer.
              (default: 3)
        """
        super(Back, self).__init__(settings)

        k = self._settings['edge_thinning']
        if k > 1:
            self._erode = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    def get(self, img):
        return self._floodfill(img) | self._global(img)

    def _global(self, img):
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.bool)
        max_distance = self._settings['max_distance']
        corners = [(0, 0), (h - 1, 0), (0, w - 1), (h - 1, w - 1)]

        if self._settings['use_lab']:
            img = Back._bgr2lab(img)

        for color in (img[i, j] for i, j in corners):
            norm = np.sqrt(np.sum(np.square(img - color), 2))
            mask |= norm < max_distance

        return mask

    def _floodfill(self, img):
        back = 255 - img
        back = Back._sobel(back, self._settings['blur_radius'])
        _, back = cv2.threshold(back, 24, 128, cv2.THRESH_BINARY)

        h, w = back.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        corners = [(0, 0), (h - 1, 0), (0, w - 1), (h - 1, w - 1)]
        for corner in corners:
            cv2.floodFill(back, mask, corner, 255)

        s = self._settings['edge_thinning']
        if s > 1:
            # Thin edges.
            contours = np.zeros((h, w), np.uint8)
            contours[back == 128] = 255
            contours = cv2.erode(contours, self._erode)
            idx = (back != 128)
            contours[idx] = back[idx]
            back = contours
        elif s >= 0:
            # Do not thin the edges.
            back[back == 128] = 255
        else:
            # Ignore edges.
            back[back == 128] = 0

        return back.astype(np.bool)

    @staticmethod
    def _default_settings():
        return {
            'max_distance': 5,
            'use_lab': True,
            'edge_thinning': 3,
            'blur_radius': 3,
        }

    @staticmethod
    def _sobel(img, radius):
        if radius > 0:
            img = cv2.GaussianBlur(img, (radius, radius), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3))
        sy = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3))
        return (0.5 * sx + 0.5 * sy).astype(np.uint8)

    @staticmethod
    def _bgr2lab(img):
        floated = (img / 255.).astype(np.float32)
        return cv2.cvtColor(floated, cv2.COLOR_BGR2LAB)
