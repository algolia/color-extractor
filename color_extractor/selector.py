import numpy as np

from .task import Task


class Selector(Task):
    def __init__(self, settings={}):
        super(Selector, self).__init__(settings)

    def get(self, k, labels, centers):
        s = self._settings['strategy']
        if s == 'largest':
            return Selector._largest(k, labels, centers)
        elif s == 'ratio':
            return self._ratio(k, labels, centers)
        elif s == 'all':
            return centers
        else:
            raise ValueError('Unknown strategy {}'.format(s))

    def _ratio(self, k, labels, centers):
        counts = [np.count_nonzero(labels == l) for l in range(k)]
        counts = np.array(counts, np.uint32)
        total = np.sum(counts)
        sort_idx = np.argsort(counts)[::-1]
        cum_counts = np.cumsum(counts[sort_idx])

        threshold = self._settings['ratio.threshold']
        for idx_stop in range(k):
            if cum_counts[idx_stop] >= threshold * total:
                break
        sort_centers = centers[sort_idx]
        return sort_centers[:idx_stop + 1]

    @staticmethod
    def _largest(k, labels, centers):
        counts = [np.count_nonzero(labels == l) for l in range(k)]
        sort_idx = np.argsort(counts)[::-1]
        return [centers[sort_idx[0]]]

    @staticmethod
    def _default_settings():
        return {
            'strategy': 'largest',
            'ratio.threshold': 0.75,
        }
