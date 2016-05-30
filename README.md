# Color Extractor

This project is both a library and a CLI tool to help tag images destined to be
searched afterwards. Most of the preprocessing steps assume that the images are
related to e-commerce, meaning that the objects targeted by the algorithms are
supposed to be mostly centered and with a fairly simple background
(single color, gradient, low contrast, etc.). The algorithm may still perform if
any of those two conditions is not met, but be aware that its precision will
certainly be hindered.

This tool only takes care of enriching your data. If you want to use Algolia to
search this enriched data, the configuration of the index is still up to you.

## Installation

The script and the library are currently targeting python 3 and won't work with
python 2.

Most of the dependencies can be installed using

```sh
pip install -r requirements.txt
```

Note that library and the CLI tool also depends on opencv 3.1.0 and its python 3
bindings.
For Linux users, the steps to install it are available
[here](http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/).
For OSX users, the steps to install it are available
[here](http://www.pyimagesearch.com/2015/06/29/install-opencv-3-0-and-python-3-4-on-osx/)

You then just have to ensure that this repository root is present in your
`PYTHONPATH`.

## Color tagging

Searching objects by color is a common practice while browsing e-commerce
web sites and relying only on the description and the title of the object may
not be enough to provide top-notch relevancy.
We propose this tool to automatically associate color tags to an image by
trying to guess the main object of the picture and extracting its dominant
color(s).

The design of the library can be viewed as a pipeline composed of several
sequential processing. Each of these processing accepts several options in order
to tune its behavior to better fit your catalog.
Those processings are (in order):

1. Resizing and cropping

2. Background detection

3. Skin detection

4. Clustering of remaining pixels

5. Selection of the _best_ clusters

6. Giving color names to clusters

### Usage

The library can be used as simply as this:

```python
import cv2
import numpy as np

from color_extractor import ImageToColor

npz = np.load('color_names.npz')
img_to_color = ImageToColor(npz['samples'], npz['labels'])

img = cv2.imread('image.jpg')
print(img_to_color.get(img))
```

The CLI tool as simply as this:

```sh
./color_extractor color_names.npz image.jpg
```

The file `color_names.pnz` can be found in this repository.

### Passing Settings

All algorithms can be used right out of the box because of settings tweaked for
the larger range of images possible. Because these settings don't target any
special kind of catalog, changing them may cause a gain of precision.

Settings can be passed at three different levels.

The lowest level is at the algorithm-level. Each algorithm is embodied by a
python class which accepts a `settings` dictionary. This dictionary is then
merged with its default settings. The given settings have precedence over the
default one.

A slightly higher level still concerns the library users. The process of chaining
all those algorithms together is also embedded in 3 classes called `FromJson`,
`FromFile` and `ImageToColor`. Those three classes also take a `settings`
parameter, composed of several dictionary to be forwarded to each algorithm.

The higher level is to pass those settings to the CLI tool. When passing the
`--settings` option with a JSON file the latter is parsed as a dictionary and
giving to the underlying `FromJson` or `FromFile` object (which in turn will
forward to the individual algorithms).

### Resizing and Cropping

This step is available as the `Resize` class.

Pictures with a too high resolution have too much details that can be considered
as noise when the goal is to find the most dominant colors. Moreover, smaller
images mean faster processing time. Most of the testing have been done on
`100x100` images, and it is usually the best compromise between precision and
speed.
Most of the time the object of the picture is centered, cropping can make sense
in order to reduce the quantity of background and ease its removal.

The available settings are:

- `'crop'` sets the cropping ratio. A ratio of `1.` means no cropping.
  Default is `0.9`.

- `'rows'` gives the number of rows to reduce the image to. The columns are
  computed to keep the same ratio.
  Default is `100`.

### Background Detection

This step is available as the `Back` class.

This algorithm tries to discard the background from the foreground by combining
two simple algorithms.

The first algorithm takes the colors of the four corners of the image and treat
as background all pixels _close_ to those colors.

The second algorithm uses a Sobel filter to detect edges and then runs a
flood fill algorithm from all four corners. All pixels touched by the flood fill
are considered background.

The masks created by the two algorithms are then combined together with a
logical `or`.

The available settings are:

- `'max_distance'` sets the maximum distance for two colors to be considered
  close by the first algorithm. A higher value means more pixels will be
  considered as background.
  Default is `5`.

- `'use_lab'` converts pixels to the LAB color space before using the first
  algorithm. The conversion makes the process a bit more expensive but the
  computed distances are closer to human perception.
  Default is `True`.

- `'edge_thinning'` The second algorithm has the choice of considering detected
  edges as part of the foreground or part of the background. It also can thin
  the edges while considering them part of the background, which avoid loosing
  too much of the object pixels.
  To consider edges as part of the object `-1` must be passed. To consider edges
  as part of the background `0` or `1` must be passed. All other positive
  integer will thin the edges; the bigger the value, the thinner the edges.
  Default is `3`.

- `'blur_radius'` Prior to applying the Sobel filter, the image is blurred using
  a Gaussian filter. The size of the filter can be specified by this setting.
  Bigger values will yield to less detected edges and as a result a more
  aggressive background detection.
  Default is `3`.

### Skin Detection

This step is available as the `Skin` class.

When working with fashion pictures models are usually present in the picture.
The main problem is that their skin color can be confused with the object color
and yield to incorrect tags. One way to avoid that is to ignore ranges of colors
corresponding to common color skins.

The available settings are:

- `'skin_type'` The skin type to target. At the moment only `'general'` and
  `'none'` are supported. `'none'` returns an empty mask every time,
  deactivating skin detection.
  Default is `'general'`.


### Clustering

This step is available as the `Cluster` class.

As we want to find the most dominant color(s) of an object, grouping them into
buckets allows us to retain only a few ones and to have a sense of which are the
most present.
The clustering is done using the K-Means algorithm. K-Means doesn't result
in the most accurate clusterings (compared to Mean Shift for example) but its
speed certainly compensate. Before all images are different, it's hard to
use a fixed number of clusters for the entire catalog. We implemented a method
that tries to find an optimal number of clusters called the
[jump](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set#An_Information_Theoretic_Approach)
method.

The available settings are:

- `'min_k'` The minimum number of clusters to consider.
  Default is `2`.

- `'max_k'` The maximum number of clusters to consider. Allowing more clusters
  results in greater computing times.
  Default is `7`.

### Selection of Clusters

This step is available as the `Selector` class.

Once clusters are made, all of them may not be worth a color tag: some may be
very tiny for example. The purpose of this step is to only keep the clusters
that are worth it.
We implemented different way of selecting clusters:

- `'all'` keeps all clusters.

- `'largest'` keeps only the largest cluster.

- `'ratio'` keeps the biggest clusters until their total number of pixels
  exceeds a certain percentage of all clustered pixels.

While the outcome off `all` is quite obvious, the use of `largest` versus
`ratio` is trickier. `largest` will yield very few colors, meaning the chance
of attributing a tag not really relevant is greatly diminished. On the other
hand objects with two colors in equal quantity will see one of them discarded.
It's up to you to decide which one behaves the best with your catalog.

The available settings are:

- `'strategy'`: The strategy to used among `'all'`, `'largest'` and `'ratio'`.
  Default is `'largest'`.

- `'ratio.threshold'`: The percentage of clustered pixels to target while
  selecting clusters with the `'ratio'` strategy.
  Default is `0.75`.

### Naming Color Values

This step is available as the `Name` class.

The last step is to give human readable color names to RGB values. To solve
this last step we use a K Nearest Neighbors algorithm applied to a large
dictionary of colors taken from the XKCD color survey. Because of the erratic
distribution of colors (some colors are far more represented that others) a
KNN behaves in most cases better than more statistical classifiers.
The "learning" phase of the classifier is done when the object is built, and
requires that two arrays are passed to its constructor: a array of BGR colors
and an array of the corresponding names. When using the CLI tool, the path
to an `.npz` numpy archive containing those two matrices must be given.

Even if the algorithm used defaults to KNN, it's still possible to use a custom
class to do it. The supplied class must support a `fit` method in lieu of
training phase and a `predict` method for the actual classification.

The available settings are:

- `'algorithm'` The algorithm to use to perform the classification. Must be
  either `'knn'` or `'custom'`. If custom is given, `'classifier.class'`' must
  also be given.
  Default is `'knn'`'

- `'hard_monochrome'` Monochrome colors (especially gray) may be hard to
  classify, this option makes use of a built in way of qualifying colors as
  "white", "gray" or "black". It uses the rejection of the color vector against
  the gray axis and uses a threshold to determine whether or not the color can
  be considered monochrome and the luminance to classify it as "black", "white"
  or "gray".
  Default is `True`.

- `'{gray,white,black}_name'` When using `'hard_monochrome'` changes the name
  actually given to "gray", "white" and "black" respectively. Useful when
  wanting color names in another language.
  Default is `"gray"`, `"white"` and `"black"`

- `'classifier.args'` Arguments passed to the classifier constructor. Default
  one are provided for `'knn'` being
  `{"n_neighbors": 50, "weights": "distance", "n_jobs": -1}`. The possible
  arguments are the ones available to the scikit-learn implementation of the
  `KNeighborsClassifier`.

- `'classifier.scale'` Many classification algorithms make strong assumption
  regarding the distribution of the samples, and may need some kind of
  standardization of the data to behave better. This settings controls the
  application of such a standardization before training and prediction.
  Default is `True` but is ignored when using `'knn'`.

### Complete Processing

Instead of instantiating each of the aforementioned classes, you can simply use
`ImageToColor` or `FromFile`. Those two classes take the same
arguments for their construction.

- An array of BGR colors to learn how to associate color names to color values.

- An array of strings corresponding to the labels of the previous array.

- A dictionary of settings to be passed to each processing.

The dictionary can have the following keys:

- `'resize'` settings to be given to the `Resize` object

- `'back'` settings to be given to the `Back` object

- `'skin'` settings to be given to the `Skin` object

- `'cluster'` settings to be given to the `Cluster` object

- `'selector'` settings to be given to the `Selector` object

- `'name'` settings to be given to the `Name` object


The main difference is the source of the image used. `ImageToColor` expects a
numpy array while `FromFile` expects both a local path or a URL where the
image can be (down)loaded from.

### Enriching JSON

Because we want Algolia customer to be able to enrich their JSON records easily
we provide a class able to stream JSON and add color tags on the fly.
The object is initialized with the same arguments as `FromFile` plus the name
of the field where the URI of the images can be found. While reading the JSON
file if the given name is encountered the corresponding image is downloaded and
its colors computed. Those colors are then added to the JSON object under the
field `_color_tags`. The name of this field can be changed thanks to an optional
parameter of the constructor.

Enriching JSON can be used directly from the command line as this:

```sh
./color_extractor -j color_names.npz file.json
```
