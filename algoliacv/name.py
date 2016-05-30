import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .task import Task


class Name(Task):
    def __init__(self, samples, labels, settings):
        """
        Create a color classifier trained on samples and labels. Samples should
        represent the actual value of the color (RGB, HSV, etc.) and labels
        should be the name of the color ('red', 'blue'...).
        Samples must be a `numpy` array of shape `(n_colors, 3)`.
        Labels must be a `numpy` array of `str` of shape `(n_colors,)`.

        The possible settings are:
            - algorithm: The algorithm to use for training the classifier.
              Possible values are 'knn' and 'custom'.
              If custom is provided, the_setting `classifier.class` must be
              set.
              (default: 'knn')

            - hard_monochrome: Use hardcoded values for white, black and gray.
              The check is performed in the BGR color space.
              The name returned depends on the settings 'white_name',
              'black_name' and 'gray_name'.
              (default: True)

            - {gray,white,black}_name: Name to give to the {gray,white,black}
              color when 'hard_monochrome' is used.
              (default: {'gray','white','black})

            - classifier.settings: Settings to pass to the scikit-learn
              algorithm used. The settings can be found on the scikit-learn
              documentation. Defaults are provided for the specific algorithm
              `knn` for an out-of-the-box experience.
              (default: {})

            - classifier.class: The class to use to perform the classification.
              when using the 'custom' algorithm. The class must support the
              method `fit` to train the model and `predict` to classify
              samples.
              (default: None)

            - classifier.scale: Use scikit-learn `StandardScaler` prior to
              train the model and classifying samples.
        """
        super(Name, self).__init__(settings)

        algo = self.settings['algorithm']
        if algo == 'knn':
            self.settings['classifier.scale'] = False
            args = self.settings['classifier.args'] or Name._knn_args()
            type_ = KNeighborsClassifier
        elif algo == 'custom':
            args = self.settings['classifier.args']
            type_ = self.settings['classifier.class']
        else:
            raise ValueError('Unknown algorithm {}'.format(algo))

        self.classifier = type_(**args)
        self.names, labels = np.unique(labels, return_inverse=True)

        if self.settings['classifier.scale']:
            self.scaler = StandardScaler()
            samples = self.scaler.fit_transform(samples)

        self.classifier.fit(samples, labels)

    def get(self, sample):
        """Return the color names for `sample`"""
        labels = []

        if self.settings['hard_monochrome']:
            labels = Name._hard_monochrome(sample)
            if labels:
                return labels

        if self.settings['classifier.scale']:
            sample = self.scaler.transform(sample)

        sample = sample.reshape((1, -1))
        labels += [self.names[i] for i in self.classifier.predict(sample)]
        return labels

    # Normalized identity (BGR gray) vector.
    _GRAY_UNIT = np.array([1, 1, 1]) / norm(np.array([1, 1, 1]))

    # Coefficients for BGR -> luminance.
    _GRAY_COEFF = np.array([0.114, 0.587, 0.299], np.float32)

    @staticmethod
    def _hard_monochrome(sample):
        """
        Return the monochrome colors corresponding to `sample`, if any.
        A boolean is also returned, specifying whether or not the saturation is
        sufficient for non monochrome colors.
        """
        gray_proj = np.inner(sample, Name._GRAY_UNIT) * Name._GRAY_UNIT
        gray_dist = norm(sample - gray_proj)

        if gray_dist > 15:
            return []

        colors = []
        luminance = np.sum(sample * Name._GRAY_COEFF)
        if luminance > 45 and luminance < 170:
            colors.append('gray')
        if luminance <= 45:
            colors.append('black')
        if luminance >= 170:
            colors.append('white')

        return colors

    @staticmethod
    def _knn_args():
        """Return the default arguments used by the `KNeighborsClassifier`"""
        return {
            'n_neighbors': 50,
            'weights': 'distance',
            'n_jobs': -1,
        }

    @staticmethod
    def _default_settings():
        return {
            'algorithm': 'knn',

            'hard_monochrome': True,
            'white_name': 'white',
            'black_name': 'black',
            'gray_name': 'gray',

            'classifier.class': None,
            'classifier.args': {},
            'classifier.scale': True,
        }
