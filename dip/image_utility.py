# -*- coding: utf-8 -*-
import numpy as np
import skimage.io as iio

from os import listdir
from os.path import isfile, join

def norm(x):
    y = x.astype(float)
    y = y-np.min(y)
    return y/np.max(y)


def load_image(path):
    """
    Loads an image in gray scale.

    Parameters
    ----------
    path : str
        Path to the image

    Retruns
    -------
    image: np.ndarray
        Normalized gray-scale image
    """
    return norm(iio.imread(path, as_grey=True))


def load_images(folder, fformat=".png"):
    """
    Load all images contained in a folder.

    Parameters
    ----------
    folder : str
        Path to the folder
    fformat : str (default '.png')
        Selected image format

    Returns
    -------
    images : np.ndarray
        Normalized images
    """
    filt = lambda x, y: isfile(x) and x.endswith(y)
    onlyfiles = [f for f in listdir(folder) if filt(join(folder,f), fformat)]
    dataset = []
    for im_path in onlyfiles:
        i = load_image(folder+'/'+im_path)
        dataset.append(i)
    return np.array(dataset)

def save_image(img, path):
    """ Access Skimage.io.imsave function."""
    iio.imsave(path, np.round(np.clip(img, 0, 1)* 255).astype(int))