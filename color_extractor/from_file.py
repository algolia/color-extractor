from skimage.io import imread

from .image_to_color import ImageToColor
from .task import Task


class FromFile(Task):
    def __init__(self, samples, labels, settings=None):
        if settings is None:
            settings = {}

        super(FromFile, self).__init__(settings)
        self._image_to_color = ImageToColor(samples, labels, self._settings)

    def get(self, uri):
        return self._image_to_color.get(imread(uri))
