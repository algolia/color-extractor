#! /usr/bin/env python

"""Color Extractor

Add color tags to your e-commerce pictures to improve the search experience!

This command line tool needs the path to a numpy npz file containing two
matrices allowing the script to learn how to give color names to color values.
By default the two matrices are expected to be named `samples` and `labels`.

Following the npz archive are expected local paths or URLs to images. Those
images will then be downloaded and their associated color computed. Colors will
be printed as a comma-separated list, a line per image.
A JSON can be used to parameterize the colors' computation using the
`--settings` option. For details about the available options, refer to the
README.md file.

Instead of paths to images, paths to JSON files can be given by supplying
`--enrich-json`. The script will then proceed to retrieve the
images found in the JSON objects and compute their associated colors.
The whole JSON objects are then written to the standard output with an
additional attribute containing the color tags.
By default the images are retrieved from the 'image' attribute and the colors
written to the `_color_tags` attribute.

Usage:
    color-extractor.py [options] <npz> <files>...

Options:
    -h --help               Show this message.

    -s, --settings <file>   Read configuration of the pipeline from the given
                            JSON file.

    --npz-samples <name>    Name of the samples matrix in the npz archive.
                            [default: samples]

    --npz-labels <name>     Name of the labels matrix in the npz archive.
                            [default: labels]

    -j, --enrich-json       Expect JSON files and enrich them with color tags.
                            [default: False]

    --image-field <field>   Use <field> to retrieve images from JSON files.
                            Must be used with `--enrich-json`.
                            [default: image]

    --colors-field <field>  Write found colors in <field> JSON field.
                            Must be used with `--enrich-json`.
                            [default: _color_tags]

"""

import json
from sys import stdout, stderr

import numpy as np

from color_extractor import FromJson, FromFile
from docopt import docopt


def _load_matrices(args):
    try:
        npz = np.load(args['<npz>'])
    except Exception as e:
        stderr.write('Failed to load npz archive: `{}`\n'.format(e))
        exit(1)

    def load_matrix(key, name):
        try:
            return npz[key]
        except KeyError as e:
            stderr.write('Failed to load {} matrix: `{}`\n'.format(name, e))
            exit(1)

    s = load_matrix(args['--npz-samples'], 'samples')
    l = load_matrix(args['--npz-labels'], 'labels')
    return s, l


def _load_settings(file_):
    try:
        with open(file_, 'r') as f:
            return json.load(f)
    except Exception as e:
        stderr.write('Failed to load settings file: `{}`\n'.format(e))
        exit(1)


def _json_files(args, samples, labels, settings):
    ifield = args['--image-field']
    cfield = args['--colors-field']
    j = FromJson(ifield, samples, labels, cfield, settings)

    stdout.write('[')

    for i, file_ in enumerate(args['<files>']):
        with open(file_, 'r') as f:
            j.get(f)

        if i < len(args['<files>']) - 1:
            stdout.write(',')

    stdout.write(']')


def _images_files(args, samples, labels, settings):
    f = FromFile(samples, labels, settings)
    for file_ in args['<files>']:
        try:
            colors = f.get(file_)
            if isinstance(colors, tuple):
                colors = colors[0]
            print(','.join(colors))
        except Exception as e:
            m = 'Unable to find colors for {}: `{}`\n'.format(file_, e)
            stderr.write(m)
            print('')


if __name__ == '__main__':
    args = docopt(__doc__, version='Color Extractor 1.0')
    samples, labels = _load_matrices(args)
    settings = {}
    if args['--settings'] is not None:
        settings = _load_settings(args['--settings'])

    if args['--enrich-json']:
        _json_files(args, samples, labels, settings)
    else:
        _images_files(args, samples, labels, settings)
