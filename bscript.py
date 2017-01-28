# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 18:09:15 2016

@author: hjalmar
"""

from glob import glob
from bolar import compare_images, get_filters, compare_image_pair
from scipy.misc import imresize
from scipy.ndimage import imread
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

verbose = True
im_dir = '/home/hjalmar/GDrive/data/PERPL/Fractals/512x512/jpg/'

fnames = glob(im_dir + '*.jpg')
fnames.sort()
N = len(fnames)

images = []
for fn in fnames:
    images.append(imresize(imread(fn), 0.5))
    
filters, n_filt = get_filters(images[0].shape)
scales = np.array([3., 7., 15., 31.]) * np.max(images[0].shape) / 256.
fig = plt.figure(figsize=[10,11.125], facecolor='w')
for i, flt in enumerate(filters):
    sz = 0.2*(flt['dx'].shape[0] / filters[-1]['dx'].shape[0])
    ax = fig.add_axes([0.1+0.178*i, 0.72, sz, sz], frameon=True)
    ax.imshow(flt['dx'], origin='lower')
    ax.set_title('Scale %d px' % scales[i])
    ax.set_xticks([])
    ax.set_yticks([])
    
pos = [(0, 0), (0, 1), (0, 2),
       (1, 0), (1, 1), (1, 2),
       (2, 0), (2, 1), (2, 2)]
flts = list(filters[-1].keys())
flts.sort()
sz = 0.2
step = 0.8/3
for i, dG in enumerate(flts):
    x = pos[i][0]*step+0.1
    y = pos[i][1]*step*0.85+0.02
    ax = fig.add_axes([x, y, sz, sz], frameon=True)
    ax.imshow(filters[-1][dG], origin='lower')
    ax.set_title(dG)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle('Gaussian derivative filters (4 scales and 9 derivatives)',
             fontsize=20)
    
fig.savefig('bolar_filters.png')
fig.savefig('bolar_filters.svg')


e = compare_image_pair(images[0], images[5], verbose=verbose)
fig = plt.figure(figsize=[12,5], facecolor='w')
ax = fig.add_axes([0, 0.1, 1/3, 0.75], frameon=True)
ax.imshow(images[0], origin='lower')
ax.set_title('Image 1')
ax.set_xticks([])
ax.set_yticks([])
ax = fig.add_axes([1/3, 0.1, 1/3, 0.75], frameon=True)
ax.imshow(e, origin='lower')
ax.set_title('Difference')
ax.set_xticks([])
ax.set_yticks([])
ax = fig.add_axes([2/3, 0.1, 1/3, 0.75], frameon=True)
ax.imshow(images[5], origin='lower')
ax.set_title('Image 2')
ax.set_xticks([])
ax.set_yticks([])

fig.savefig('bolar_image_pair_N%02d.png' % N)
fig.savefig('bolar_image_pair_N%02d.svg' % N)


E = compare_images(images, verbose=True)
fig = plt.figure(figsize=[12,12])
sz = 1/(N+1)
ax = fig.add_axes((sz, sz, sz*N, sz*N), frameon=False)
ax.imshow(E, interpolation='none', origin='lower')
ax.set_xticks([])
ax.set_yticks([])

for i, fn in enumerate(fnames):
    im = imresize(imread(fn), 0.5)[:,:,:3]
    ax = fig.add_axes((0, (1+i)*sz, sz, sz), frameon=False)
    ax.imshow(im, origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])    
    ax = fig.add_axes(((1+i)*sz, 0, sz, sz), frameon=False)
    ax.imshow(im, origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig('bolar_image_differences_N%02d.png' % N)
fig.savefig('bolar_image_differences_N%02d.svg' % N)


fig = plt.figure(figsize=[12,5], facecolor='w')
sz = 2/(N+1)
sz = 0.11
ax = fig.add_axes((0.02, sz*1.3, 0.96, 0.9-sz), frameon=False)
Z = linkage(E)
dd = dendrogram(Z, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ylim = ax.get_ylim()
xlim = ax.get_xlim()
ax.text(xlim[1]*0.45, ylim[1]*0.8, 'Images ordered by similarity', fontsize=23)
step = 0.96/N
for i, leave in enumerate(dd['leaves']):
    im = imresize(imread(fnames[leave]), 0.5)[:,:,:3]
    ax = fig.add_axes((0.02+i*step-sz/4, 0.02, sz, sz), frameon=False)
    ax.imshow(im, origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    
fig.savefig('bolar_image_differences_dendrogram_N%02d.png' % N)
fig.savefig('bolar_image_differences_dendrogram_N%02d.svg' % N)    