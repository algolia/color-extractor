import cv2
import numpy as np

from .task import Task


def _square_distorsion(npixels, compact, y):
    return pow(compact / npixels, -y)


class Cluster(Task):
    def __init__(self, settings={}):
        super(Cluster, self).__init__(settings)
        self._flags = cv2.KMEANS_RANDOM_CENTERS
        self._citeria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                         50, 1.0)

    def get(self, img):
        a = self.settings['algorithm']
        if a == 'kmeans':
            return self._jump(img)
        else:
            raise ValueError('Unknown algorithm'.format(a))

    def _kmeans(self, img, k):
        compact, labels, centers = cv2.kmeans(img, k, None, self._citeria, 10,
                                              self._flags)
        return compact, labels.reshape(-1), centers

    def _jump(self, img):
        img = img.astype(np.float32)
        npixels = img.size

        best = None
        prev_distorsion = 0
        largest_diff = float('-inf')

        for k in range(self.settings['min_k'], self.settings['max_k']):
            compact, labels, centers = self._kmeans(img, k)
            distorsion = _square_distorsion(npixels, compact, 1.5)
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
