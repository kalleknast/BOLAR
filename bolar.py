# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:34:01 2016

@author: hjalmar

BOLAR: Bank of Local Analyzer Responses

"""

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
import sys


def bolar(image, filters=None, n_filt=None, verbose=True):
    """
    Arguments
    ----------
    image   : An RGB image, as a numpy array with ndim==3.
    filters : Gaussian derivative filters from get_filters
    n_filt  : number of filters
    
    Returns
    b       : The BOLAR representation of the image, ie the output from
              convolving the images with the filters.
    --------
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
    # Achromatic, average of RGB
    im1 = image.mean(axis=2)
    # Differenced of Red and Green
    im2 = image[:, :, 0] - image[:, :, 1]
    im2 -= im2.min()
    im2 /= im2.max()
    # Yellow - Blue
    im3 = image[:, :, :2].mean(axis=2) - image[:, :, 2]
    im3 -= im3.min()
    im3 /= im3.max()    
    
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
    Argument
    --------
    im_shape  : Shape of image to filter. The filters are scaled to the max
                of image height/widht.
                
    Returns
    -------
    filters   : A list of length 4 of dicts holding the individual filters.
                E.g.:
                filters[0] = {'dx': <np.array>, 'dy': <np.array>,
                              'dxx': <np.array>, 'dyy': <np.array>, 'dxy': <np.array>,
                              'dxxx': <np.array>, 'dyyy': <np.array>, 'dxxy': <np.array>, 'dxyy': <np.array>}
    n_filt     : The total number of filters in "filters".

    """    
    # Zelinsky (2003) used 256 x 192 pixel images and scales/sigmas of
    # 3 x 3, 7 x 7, 15 x 15 and 31 x 31. Thus from 3/256 to 31/256 of the image.
    # Then the scale for an arbitrary size image would be:
    # scales = [3, 7, 15, 31] * max(im.shape) / 256
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
    Arguments
    ---------
    sigma       : sigma/SD (ie the scale) of the Gaussian kernel from which the 
                  filters are derived.
    max_order   : Highest order of the derivatives. Filters are returned up to 
                  to max_order
    truncate    : Truncate the Gaussian at this many sigmas/SD
    
    Returns     
    -------    
    dG          : A dict holding Gaussian derivative filters.
                  E.g:
                  dG = {'dx': <np.array>, 'dy': <np.array>,
                        'dxx': <np.array>, 'dyy': <np.array>, 'dxy': <np.array>,
                        'dxxx': <np.array>, 'dyyy': <np.array>, 'dxxy': <np.array>, 'dxyy': <np.array>}
    
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
    
    
def compare_image_pair(image0, image1):
    """
    Arguments
    ---------
    image0   : An RGB image, as a numpy array with ndim==3.
    image1   : An RGB image, as a numpy array with ndim==3.    
    
    Returns
    -------
    e        : A difference map of image0 and image1, with the same size as the
               images. 
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
    
    b0 = bolar(image0, filters=filters, n_filt=n_filt, verbose=verbose)
    b1 = bolar(image1, filters=filters, n_filt=n_filt, verbose=verbose)

    e = (1 / n_filt) * np.sqrt(((b0 - b1)**2).sum(axis=2))
    
    return e


def compare_images(images, verbose=True):
    """
    Arguments
    ---------
    images   : A list of RGB images (as a numpy arrays with ndim==3) for
               pairwise comparison.
    
    Returns
    -------
    E        : A len(images) x len(images) numpy array of pairwise image 
               differences. 
               The difference between images[2] and images[3] would be located
               at E[2,3] and E[3,2].
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
        for j, b1 in enumerate(B):
            e = (1 / n_filt) * np.sqrt(((b0 - b1)**2).sum(axis=2))
            E[i, j] = e.sum()
    
    return E
