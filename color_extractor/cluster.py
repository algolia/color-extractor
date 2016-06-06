import numpy as np
from sklearn.cluster import KMeans
from skimage.util import img_as_float

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
        self._kmeans_args = {
            'init': 'random',
            'tol': 0.5,
            'max_iter': 50,
        }

    def get(self, img):
        a = self._settings['algorithm']
        if a == 'kmeans':
            return self._jump(img)
        else:
            raise ValueError('Unknown algorithm'.format(a))

    def _kmeans(self, img, k):
        kmeans = KMeans(n_clusters=k, **self._kmeans_args)
        try:
            kmeans.fit(img)
        except:
            raise KMeansException()

        return kmeans.inertia_, kmeans.labels_, kmeans.cluster_centers_

    def _jump(self, img):
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
