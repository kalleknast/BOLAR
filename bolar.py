# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:34:01 2016

@author: hjalmar

BOLAR: Bank of Local Analyzer Responses

From the paper:
Rather, my goal is to distill from these models a simplified and easily
implemented approach tailored to the demands of a change detection task.
Many models of early vision strive to fit data from neurophysiological 
recording or near-threshold psycho- physics, but this level of specificity is 
probably unnecessary to a suprathreshold change detection task in which the 
stimuli are perceptuallydistinct real-world objects. 
More important to change detection is representational breadth, 
the coding of many different featural di- mensions at multiple levels. 
Given the multiplicative relationship between featural dimension 
(e.g., orientation) and level (e.g., 60º, 45º, etc.), and the need for 
representational breadth (many dimensions), it is easy to see how the number 
of filters needed to represent a real-world stimulus can quickly grow quite 
large. Such a high-dimensional representation would create a sort of featural 
signature for the patterns appearing in the change detection displays.
There are two clear advantages to this approach. First, because a filter-based 
model can be applied to arbitrarily complex stimuli, both simple and complex 
scenes can be coded in terms of the same base featural primitives. 
Second, because each of these featural signatures exist in a relatively 
high-dimensional space, it would no longer be necessary to hand-pick the 
features to be compared in the change detection process. A change between a 
horizontal and a vertical bar will generate a response in the feature vector 
coding for “horizontal” and “vertical,” and a change between a coffee cup and 
a plate of eggs will generate a response in whatever featural dimensions are 
specific to those patterns. To achieve somemeasure of representational breadth,

I currently use 108 Gaussian-derivative filters (GDFs) to code the visual 
features of a scene, with each filter being sensitive to a different spatial or 
chromatic property. As their name implies, the mathematical functions 
underlying GDFs are obtainedby differentiating a three-dimensional Gaussian.
The 108 GDFs used in this study can be broadly classified into three filter 
types, with each type corresponding to one of the first three derivatives of 
the Gaussian function. Recall that Gaussian differentiation yields as 
derivatives oriented functions numbering one more than the order of the 
derivation. A first-order derivation produces two directional (oriented)
derivatives, a second-order yields three, and a third-order yields four. 
These 9 oriented functions were used to represent or filter orientation 
information in the change detection stimuli. Specifically, first-order filter 
responses were collected at 0º and 90º orientations; second-order filters were 
constructed for 0º, 60º, and 120º orientations; and third-order responses were
collected at 0º, 45º, 90º, and 120º.

This set of nine oriented GDFs is repeated at four octave-separated spatial 
scales (3 x 3, 7 x 7, 15 x 15, and 31 x 31 pixels),
accounting for 36 of the 108  filters used in the current representation.
The term scale in this context can be understood as meaning the size of a 
filter’s “receptive field.”  Larger scale filters will “see” or acquire 
information over a relatively large region of the image, 
thereby enabling them to efficiently extract low spatial frequency patterns 
from a scene (e.g., the overall shape of an object). 
Small-scale filters will acquire in- formation over a comparatively narrow 
region of the image, making them well suited to extract high spatial frequency 
patterns (e.g., the fine visual structure corresponding to object parts).
Using filters of varying scales, the current representation therefore performs 
a coarse spatial frequency analysis of the changedetectionstimuli. 
These 36 multiscale oriented GDFs are in turn repeated for three 
color/intensity channels, thereby accounting for all 108 of the filters used in 
the representation. The first 36 filters perform an achromatic analysis of the 
scene. Prior to any filtering operations, the red, green, and blue pixel values 
of the image are simply averaged to produce a grayscale scene. 
The remaining two channels are indeed chromatic—more specifically, they are 
color-opponent much like the human visual system. The second channel, 
Filters 37–72 in the representation, operates on a red–green transformation of 
the image (meaning that the green pixel values are subtracted from the red, 
then normalized between 0 and 255). Similarly, the third channel, 
Filters 73–108, operates on a blue–yellow image transformation, with yellow 
being the average of the red and green pixel intensities. 
To summarize, this 108-dimensional filter-based representation can be evenly 
divided into three color/luminance channels, with each channel in turn being 
divided into four spatial scales and finally into nine oriented GDFs at each 
scale.

To obtain these responses, each of the 108 GDFs was centered over the 
midpoint of the trumpet image (i.e., the midpoint of the filter was aligned 
with the midpoint of the image) and then con- volved with the portion of the 
image “covered” by the filter (i.e., a 3 x 3 filter would operate on only the 9
pixels surrounding the image midpoint,whereas a 31 x 31 filter would analyze a 
larger 961-pixel region of the image).


In order to quantify the visual similarity (or dissimilarity) between any two
objects undergoing change in Experiment 1, I first computed BOLAR vectors for 
every pair of corresponding points in the two images, 
then found the normalized Euclidean distance for each of these vector pairs 
using the equation:

    E = 1/n * sqrt(sum(b1 - b2)**2)

where n is the dimensionality of the BOLAR representation and b1 and b2 the 
two BOLAR vectors being compared.

Performing the above-described operation for every corresponding point in the 
two images, and then normalizing these Euclidean distances between the range of 
0–255 for visualization, results in the creation of a difference map.

It is possible to further reduce the information on a difference map to obtain 
a singlenumber representing the visual similarity between two patterns.
Rather than plotting each of the pair-wise Euclidean distances in the form of 
a map, a single similarity estimate was obtained by summing these difference 
signals and convertingthem to a similarity metric. Intuitively, this method is
equivalent to comparing the overall amount ofwhite in two difference maps when 
judging whether one pair of objects is more visually similar than another.


"""

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
import sys


def bolar(image, filters=None, n_filt=None, verbose=True):
    """
    image : An RGB image
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('image needs for be RGB with shape: (im_h, im_w, 3)')
    
    if filters is None:
        filters, n_filt = get_filters(image.shape)
    
    if image.dtype is not np.dtype('float64'):
        raise ValueError('image needs to have dtype float64')
        
    if n_filt is None and not filters is None:
        print('Had to count the number of filters since "n_filt" was None.')
        n_filt = 0
        for f in filters:
            n_filt += len(f)

    im_h, im_w = image.shape[:2]
    b = np.empty((im_h, im_w, n_filt * 3), dtype='float64')
    im1 = image.mean(axis=2)
    im2 = image[:, :, 0] - image[:, :, 1]
    im2 -= im2.min()
    im2 /= im2.max()
    im3 = ((image[:, :, 0] + image[:, :, 1]) / 2.0)
    
    i = 0
    for scale in filters:
        for d in scale.keys():

            filt = scale[d]

            if i < 36:
                im = im1
            elif i < 72:
                im = im2
            else:
                im = im3

            # Use fftconvolve instead of ndimage.convolve bc it is 
            # many times faster
            b[:,:,i] = fftconvolve(im, filt, mode='same')
            
            i += 1
            
    return b


def get_filters(im_shape):
    
    """
    Zelinsky (2003) used 256 x 192 pixel images, and scales/sigmas of
    3 x 3, 7 x 7, 15 x 15 and 31 x 31. Thus from 3/256 to 31/256 of the image.
    Then the scale for an arbitrary size image would be:
    scales = [3, 7, 15, 31] * max(im.shape) / 256
    """    
    scales = np.array([3., 7., 15., 31.]) * np.max(im_shape) / 256.

    filters = []
    n_filt = 0
    for sigma in scales:
        filts = gaussian_derivative_filter(sigma, max_order=3)
        n_filt += len(filts)
        filters.append(filts)

    return filters, int(n_filt)


def gaussian_derivative_filter(sigma, max_order=3, truncate=4):
    """
    """
    
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    x = np.arange(-lw, lw + 1)
    x, y = np.meshgrid(x, x)
    # 2D Gaussian kernel:
    G = np.exp(- (x**2 + y**2) / (2 * sd * sd))

    dG = {}
        
    if int(max_order) == 1:
        dG['dy'], dG['dx'] = np.gradient(G)

    elif int(max_order) == 2:
        dG['dy'], dG['dx'] = np.gradient(G)
        dG['dxy'], dG['dxx'] = np.gradient(dG['dx'])
        dG['dyy'], _ = np.gradient(dG['dy'])

    elif int(max_order) == 3:
        dG['dy'], dG['dx'] = np.gradient(G)
        dG['dxy'], dG['dxx'] = np.gradient(dG['dx'])
        dG['dyy'], _ = np.gradient(dG['dy'])
        dG['dxxy'], dG['dxxx'] = np.gradient(dG['dxx'])
        dG['dyyy'], dG['dxyy'] = np.gradient(dG['dyy'])

    else:
        raise ValueError('order has to be 1, 2 or 3.')

    return dG
    
    
def compare_image_pair(image0, image1, verbose=True):
    """
    """
    
    if (image0.ndim != image1.ndim or image0.shape[0] != image1.shape[0] or
        image0.size != image1.size):
        raise ValueError('image0 and image1 needs to have the same dimensions.')

    if image0.dtype is np.dtype('uint8'):
        image0 = image0.astype('float64')
        image0 /= 255.

    if image1.dtype is np.dtype('uint8'):
        image1 = image1.astype('float64')
        image1 /= 255.        

    filters, n_filt = get_filters(image0.shape)
    
    if verbose:
        print('Filtering image0...')

    b0 = bolar(image0, filters=filters, n_filt=n_filt, verbose=verbose)
    if verbose:
        print('Filtering image1...')

    b1 = bolar(image1, filters=filters, n_filt=n_filt, verbose=verbose)

    e = (1 / n_filt) * np.sqrt(((b0 - b1)**2).sum(axis=2))
    
    return e


def compare_images(images, verbose=True):
    """
    """
    

    filters, n_filt = get_filters(images[0].shape)
    B = []
    N = len(images)
    for i, im in enumerate(images):
    
        if (images[0].ndim != im.ndim or images[0].shape[0] != im.shape[0] or
            images[0].size != im.size):
            raise ValueError('All images need to have the same dimensions.')

        if im.dtype is np.dtype('uint8'):
            im = im.astype('float64')
            im /= 255.

        if verbose:
            print('filtering image %d' % i)

        B.append(bolar(im, filters=filters, n_filt=n_filt, verbose=verbose))
    
    E = np.empty((N, N))
    for i, b0 in enumerate(B):
        if verbose:
            print(N-i)
        for j, b1 in enumerate(B):
            e = (1 / n_filt) * np.sqrt(((b0 - b1)**2).sum(axis=2))
            E[i, j] = e.sum()
    
    return E
