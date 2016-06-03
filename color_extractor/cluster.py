import cv2
import numpy as np

from .exceptions import KMeansException
from .task import Task


class Cluster(Task):
    """
    Use the K-Means algorithm to group pixels by clusters. The algorithm tries
    to determine the optimal number of clusters for the given pixels.
    """
    def __init__(self, settings=None):
        if settings is None:
            settings = {}

        super(Cluster, self).__init__(settings)
        self._flags = cv2.KMEANS_RANDOM_CENTERS
        self._crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50,
                      1.0)

    def get(self, img):
        a = self._settings['algorithm']
        if a == 'kmeans':
            return self._jump(img)
        else:
            raise ValueError('Unknown algorithm'.format(a))

    def _kmeans(self, img, k):
        try:
            ct, l, cr = cv2.kmeans(img, k, None, self._crit, 10, self._flags)
        except:
            raise KMeansException()

        return ct, l.reshape(-1), cr

    def _jump(self, img):
        img = img.astype(np.float32)
        npixels = img.size

        best = None
        prev_distorsion = 0
        largest_diff = float('-inf')

        for k in range(self._settings['min_k'], self._settings['max_k']):
            compact, labels, centers = self._kmeans(img, k)
            distorsion = Cluster._square_distorsion(npixels, compact, 1.5)
            diff = prev_distorsion - distorsion
            prev_distorsion = distorsion

            if diff > largest_diff:
                largest_diff = diff
                best = k, labels, centers

        return best

    @staticmethod
    def _default_settings():
        return {
            'min_k': 2,
            'max_k': 7,
            'algorithm': 'kmeans',
        }

    @staticmethod
    def _square_distorsion(npixels, compact, y):
        return pow(compact / npixels, -y)
