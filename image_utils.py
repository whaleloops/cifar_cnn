import urllib2, os, tempfile

import numpy as np
from scipy.misc import imread



"""
Utility functions used for viewing and processing images.
"""


def preprocess_image(img, mean_img, mean='image'):
  """
  Convert to float, transepose, and subtract mean pixel
  
  Input:
  - img: (H, W, 3)
  
  Returns:
  - (1, 3, H, 3)
  """
  if mean == 'image':
    mean = mean_img
  elif mean == 'pixel':
    mean = mean_img.mean(axis=(1, 2), keepdims=True)
  elif mean == 'none':
    mean = 0
  else:
    raise ValueError('mean must be image or pixel or none')
  return img.astype(np.float32).transpose(2, 0, 1)[None] - mean


def deprocess_image(img, mean_img, mean='image', renorm=False):
  """
  Add mean pixel, transpose, and convert to uint8
  
  Input:
  - (1, 3, H, W) or (3, H, W)
  
  Returns:
  - (H, W, 3)
  """
  if mean == 'image':
    mean = mean_img
  elif mean == 'pixel':
    mean = mean_img.mean(axis=(1, 2), keepdims=True)
  elif mean == 'none':
    mean = 0
  else:
    raise ValueError('mean must be image or pixel or none')
  if img.ndim == 3:
    img = img[None]
  img = (img + mean)[0].transpose(1, 2, 0)
  if renorm:
    low, high = img.min(), img.max()
    img = 255.0 * (img - low) / (high - low)
  return img.astype(np.uint8)


def image_from_url(url):
  """
  Read an image from a URL. Returns a numpy array with the pixel data.
  We write the image to a temporary file then read it back. Kinda gross.
  """
  try:
    f = urllib2.urlopen(url)
    _, fname = tempfile.mkstemp()
    with open(fname, 'wb') as ff:
      ff.write(f.read())
    img = imread(fname)
    os.remove(fname)
    return img
  except urllib2.URLError as e:
    print 'URL Error: ', e.reason, url
  except urllib2.HTTPError as e:
    print 'HTTP Error: ', e.code, url
