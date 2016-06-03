import json
import sys

import ijson

from .from_file import FromFile
from .task import Task


class FromJson(Task):
    def __init__(self, image_field, samples, labels,
                 colors_field='_color_tags', settings=None):
        if settings is None:
            settings = {}

        super(FromJson, self).__init__(settings)
        self._image_field = image_field
        self._colors_field = colors_field
        self._from_file = FromFile(samples, labels, self._settings)

    def get(self, handle, out=sys.stdout):
        prev_event = 'start_map'
        for prefix, event, value in ijson.parse(handle):
            FromJson._put_comma(event, prev_event, out)
            if event.startswith('start_'):
                out.write('{' if event == 'start_map' else '[')
            elif event.startswith('end_'):
                out.write('}' if event == 'end_map' else ']')
            elif event == 'map_key':
                out.write('{}:'.format(json.dumps(value)))
            elif event == 'number':
                out.write(str(value))
            else:
                out.write(json.dumps(value))

            if event == 'string' and prefix.endswith(self._image_field):
                self._add_colors_tags(value, out)

            prev_event = event

    def _add_colors_tags(self, uri, out):
        try:
            colors = self._from_file.get(uri)
        except Exception as e:
            colors = []
            m = 'Unable to find colors for {}: `{}`'.format(uri, e)
            sys.stderr.write(m)

        out.write(',"{}":{}'.format(self._colors_field, json.dumps(colors)))

    @staticmethod
    def _put_comma(ev, prev, out):
        if (ev != 'end_array' and ev != 'end_map' and prev != 'start_map' and
                prev != 'start_array' and prev != 'map_key'):
            out.write(',')
