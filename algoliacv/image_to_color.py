from .back import Back
from .cluster import Cluster
from .name import Name
from .resize import Resize
from .selector import Selector
from .skin import Skin
from .task import Task


class ImageToColor(Task):
    def __init__(self, samples, labels, settings={}):
        super(ImageToColor, self).__init__(settings)
        self.resize = Resize(self.settings['resize'])
        self.back = Back(self.settings['back'])
        self.skin = Skin(self.settings['skin'])
        self.cluster = Cluster(self.settings['cluster'])
        self.selector = Selector(self.settings['selector'])
        self.name = Name(samples, labels, self.settings['name'])

    def get(self, img):
        resized = self.resize.get(img)
        mask = self.back.get(resized) | self.skin.get(resized)
        k, labels, centers = self.cluster.get(resized[~mask])
        centers = self.selector.get(k, labels, centers)
        return set([c for l in [self.name.get(c) for c in centers] for c in l])

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
