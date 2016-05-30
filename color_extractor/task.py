class Task(object):
    def __init__(self, settings):
        self._settings = self._default_settings()
        self._settings.update(settings)

    def get(self, img):
        raise NotImplementedError

    @staticmethod
    def _default_settings():
        return {}
