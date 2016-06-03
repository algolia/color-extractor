from os.path import abspath
from urllib.parse import quote
from urllib.request import urlopen

import cv2
import numpy as np

from .image_to_color import ImageToColor
from .task import Task


class FromFile(Task):
    def __init__(self, samples, labels, settings=None):
        if settings is None:
            settings = {}

        super(FromFile, self).__init__(settings)
        self._image_to_color = ImageToColor(samples, labels, self._settings)

    def get(self, uri):
        if uri.find('//') == -1:
            uri = 'file://' + quote(abspath(uri))

        # TODO: Error reporting.
        resp = urlopen(uri)
        buf = np.fromstring(resp.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        return self._image_to_color.get(img)
