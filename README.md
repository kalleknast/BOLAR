# BOLAR
A Python implementation of BOLAR, developed by <cite>[Zelinsky (2003)][1]</cite>.

## What it is

**BOLAR** -- Bank Of Local Analyzer Responses -- Is a model for measuring image difference. This is useful in visual Psychology/Neuroscience experiments when some measure of image differences is needed.

It works by filtering images through banks of oriented filters at different spatial scales. In this way, it is similar to, for example, the <cite>[HMAx][2]</cite> model by <cite>[Riesenhuber & Poggio (1999)][3]</cite>, a computational model of neural responses in visual cortex. However, BOLAR is not developed as a model of cortical responses, but as a simple and flexible model that can account for some experimental effects in Psychology on vison.

## How it works
The images are filtered with <cite>[Gaussian derivative filters][4]</cite> up to the 4rd order, at 4 spatial scales. Each order provided order + 1 filters, so 9 (2 + 3 + 4) filters at each of the 4 scales.

```python
filters, n_filt = get_filters(image.shape)
```


<img src="https://github.com/kalleknast/BOLAR/blob/master/bolar_filters.png" width="400" />

These 36 filters are applied 3 times. First, achromatic, the average of the RGB channels, second, the red green difference, and third the difference between yellow (average or red and green) and blue. This done over all pixes in of an image, resulting in a 108 valued BOLAR vector of filter responses for each pixel.

The normalized Euclidean distance between BOLAR vectors from a pair of images provides a map of image differences.

```python
e = compare_image_pair(image0, image1)
```

<img src="https://github.com/kalleknast/BOLAR/blob/master/bolar_image_pair.png" width="600" />

Finally, this difference map can be summed, providing a single value quantifying image differences.
To do this over a list of images call:
```python
E = compare_images(images, verbose=False)
```
<img src="https://github.com/kalleknast/BOLAR/blob/master/bolar_image_differences.png" width="500" />

The square distance matrix can be used to further analyse image similarity with, for example, a dendrogram:

```python
from scipy.cluster import hierarchy

Z = hierarchy.linkage(E)
dd = hierarchy.dendrogram(Z)
```

<img src="https://github.com/kalleknast/BOLAR/blob/master/bolar_image_differences_dendrogram.png" width="600" />

[1]:http://www.psychology.sunysb.edu/gzelinsky-/index_htm_files/Z2003.pdf
[2]:http://maxlab.neuro.georgetown.edu/hmax.html
[3]:http://maxlab.neuro.georgetown.edu/docs/publications/nn99.pdf
[4]:http://campar.in.tum.de/Chair/HaukeHeibelGaussianDerivatives