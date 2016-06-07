import numpy as np

from .back import Back
from .cluster import Cluster
from .name import Name
from .resize import Resize
from .selector import Selector
from .skin import Skin
from .task import Task


class ImageToColor(Task):
    def __init__(self, samples, labels, settings=None):

        if settings is None:
            settings = {}

        super(ImageToColor, self).__init__(settings)
        self._resize = Resize(self._settings['resize'])
        self._back = Back(self._settings['back'])
        self._skin = Skin(self._settings['skin'])
        self._cluster = Cluster(self._settings['cluster'])
        self._selector = Selector(self._settings['selector'])
        self._name = Name(samples, labels, self._settings['name'])

    def get(self, img):
        resized = self._resize.get(img)
        back_mask = self._back.get(resized)
        skin_mask = self._skin.get(resized)
        mask = back_mask | skin_mask
        k, labels, clusters_centers = self._cluster.get(resized[~mask])
        centers = self._selector.get(k, labels, clusters_centers)
        colors = [self._name.get(c) for c in centers]
        flattened = list({c for l in colors for c in l})

        if self._settings['debug'] is None:
            return flattened

        colored_labels = np.zeros((labels.shape[0], 3), np.float64)
        for i, c in enumerate(clusters_centers):
            colored_labels[labels == i] = c

        clusters = np.zeros(resized.shape, np.float64)
        clusters[~mask] = colored_labels

        return flattened, {
            'resized': resized,
            'back': back_mask,
            'skin': skin_mask,
            'clusters': clusters
        }

    @staticmethod
    def _default_settings():
        return {
            'resize': {},
            'back': {},
            'skin': {},
            'cluster': {},
            'selector': {},
            'name': {},
        }
