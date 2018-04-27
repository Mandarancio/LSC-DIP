#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utilities for LSC and DIP project.

@author: Martino Ferrari
@email: manda.mgf@gmail.com
"""

from os import listdir
from os.path import isfile, join

import numpy as np

import skimage.io as iio
from skimage.transform import resize

import pyfftw


# Some global variables for caching vectorization patterns
__supp__ = None
__ppus__ = None
__nsupp__ = None
__nppus__ = None
__zsupp__ = None
__zppus__ = None


# Some lambadas and namespaces
sci_fft = pyfftw.interfaces.scipy_fftpack
np_fft = pyfftw.interfaces.numpy_fft

fft2 = lambda x : np_fft.fftshift(np_fft.fft2(x))
ifft2 = lambda x : np_fft.ifft2(np_fft.ifftshift(x))
dct2 = lambda x : sci_fft.dct(sci_fft.dct(x.T, norm='ortho').T, norm='ortho')
idct2 = lambda x : sci_fft.idct(sci_fft.idct(x.T, norm='ortho').T, norm='ortho')
comulate = lambda x : np.array([np.sum(x[:i]) for i in range(len(x))])


def __gen_supp__ (shape):
    """Support function for sim-circle"""
    w, h = shape
    cx, cy = int(round(w / 2)), int(round(h / 2))
    x = np.linspace(0, w, w) - cx
    y = np.linspace(0, h, h) - cy
    x, y = np.meshgrid(y, x)
    M = (x**2 + y**2)
    M1 = M.copy()
    M1[:,cy:] = np.max(M) + 1
    M2 = M.copy()
    M2[:,:cy] = np.max(M) + 1
    supp = np.zeros(w*h, int)
    supp[:cy*w] = M1.reshape(-1).argsort()[:cy*w][::-1]
    supp[cy*w:] = M2.reshape(-1).argsort()[:cy*w]
    return supp


def __gen_nsupp__(shape):
    """Support function for circle"""
    w, h = shape
    cx, cy = int(round(w / 2)), int(round(h / 2))
    x = np.linspace(0, w, w) - cx
    y = np.linspace(0, h, h) - cy
    x, y = np.meshgrid(y, x)
    M = (x**2 + y**2)
    return M.reshape(-1).argsort()

def __gen_zsupp__(shape):
    n, m = shape
    def lmove(i, j):
        if j < (m - 1):
            return max(0, i-1), j+1
        else:
            return i+1, j
    def umove(i, j):
        if j < (n - 1):
            return max(0, i-1), j+1
        else:
            return i+1, j
    a = np.zeros((n, m), int)
    x, y = 0, 0
    for v in range(n * m):
        a[y][x] = v
        if (x + y) & 1:
            x, y = umove(x, y)
        else:
            y, x = lmove(y, x)
    return inv(a.reshape(-1))


def spiral_vect(x):
    """
    Vectorialization optimised for the 2d dft.
     
    Parameters
    ----------
    x : np.ndarray
        matrix signal
    
    Returns
    -------
    v : np.array
        vectorialized signal
    """
    if (len(x.shape) == 3):
        l, w, h = x.shape
        X = np.zeros((l, w*h), x.dtype)
        for i in range(l):
            X[i] = spiral_vect(x[i])
        return X
    elif len(x.shape) != 2:
        raise ValueError("Input must be 2D or 3D.")
        
    h, w = x.shape
    rs, re = 0, h - 1
    cs, ce = 0, w - 1
    j = 0
    l = x.size
    v = np.zeros(x.size, x.dtype)
    while ce > cs or re > rs:
        v[j:j+w] = (x[rs, np.arange(cs, ce+1)])
        rs +=1 
        j += w
        h -= 1
        if j >= l:
            break
        v[j:j+h] = (x[np.arange(rs, re+1), ce])
        j += h
        w -= 1
        ce -= 1
        if j >= l:
            break
        v[j:j+w] = (x[re, np.arange(ce ,cs-1,-1)])
        j += w
        h -= 1
        re -= 1
        if j >= l:
            break
        v[j:j+h] = (x[np.arange(re, rs-1,-1), cs])
        j += h
        w -= 1
        cs += 1
        if j >= l:
            break
    return v


def spiral_mat(v, shape):
    """
    Inverse-Vectorialization optimised for the 2d dft.
    
    Parameters
    ----------
    v : np.array
        vectorialized signal
    shape : tuple
        original shape
    
    Returns
    -------
    x : np.ndarray
        matrix form of v
    """
    if (len(v.shape) == 2):
        l = v.shape[0]
        X = np.zeros((l, *shape), v.dtype)
        for i in range(l):
            X[i] = spiral_mat(v[i], shape)
        return X
    elif len(v.shape) != 1:
        raise ValueError("Input must be 1D or 2D.")
    if np.prod(shape) != v.size:
        raise Exception("Input size does not match output shape.")

    h, w  = shape
    rs, re = 0, h - 1
    cs, ce = 0, w - 1
    j = 0 
    l = v.size
    x = np.zeros(shape, v.dtype)
    while ce > cs or re > rs:
        x[rs, np.arange(cs,ce+1)] = v[j:j+w]
        rs +=1 
        j += w
        h -= 1
        if j >= l:
            break
        x[np.arange(rs,re+1), ce] = v[j:j+h]
        j += h
        w -= 1
        ce -= 1
        if j >= l:
            break
        x[re, np.arange(ce,cs-1,-1)] = v[j:j+w]
        j += w
        h -= 1
        re -= 1
        if j >= l:
            break
        x[np.arange(re,rs-1,-1), cs] = v[j:j+h]
        j += h
        w -= 1
        cs += 1
        if j >= l:
            break
    return x


def sim_spiral_vect(x):
    """
    Simmetrical vectorialization optimised for the 2d dft.
     
    Parameters
    ----------
    x : np.ndarray
        matrix signal
    
    Returns
    -------
    v : np.array
        vectorialized signal
    """
    if (len(x.shape) == 3):
        l, w, h = x.shape
        X = np.zeros((l, w*h), x.dtype)
        for i in range(l):
            X[i] = sim_spiral_vect(x[i])
        return X
    h, w = x.shape
    rs, re = 0, h - 1
    cs, ce = 0, w - 1
    j = 0
    l = x.size
    i = l-1
    v = np.zeros(x.size, x.dtype)
    while j != i:
        v[j:j+w] = (x[rs, np.arange(cs, ce+1)])
        rs +=1 
        j += w
        h -= 1
        if j >= l:
            break
        
        v[np.arange(i, i-w,-1)] = (x[re, np.arange(ce ,cs-1,-1)])
        i -= w
        h -= 1
        re -= 1
        if j >= i:
            break
        
        v[j:j+h] = (x[np.arange(rs, re+1), ce])
        j += h
        w -= 1
        ce -= 1
        if j >= i:
            break
            
        v[np.arange(i, i-h, -1)] = (x[np.arange(re, rs-1,-1), cs])
        i -= h
        w -= 1
        cs += 1
        if j >= i:
            break
        
        v[j:j+w] = (x[re, np.arange(ce, cs-1,-1)])
        re -=1 
        j += w
        h -= 1
        if j >= l:
            break
            
        v[np.arange(i, i-w,-1)] = (x[rs, np.arange(cs ,ce+1)])
        i -= w
        h -= 1
        rs += 1
        if j >= i:
            break
            
        v[j:j+h] = (x[np.arange(re, rs-1,-1), cs])
        j += h
        w -= 1
        cs += 1
        if j >= i:
            break
            
        v[np.arange(i, i-h, -1)] = (x[np.arange(rs, re+1), ce])
        i -= h
        w -= 1
        ce -= 1
        if j >= i:
            break
    return v


def sim_spiral_mat(v, shape):
    """
    Inverse-Vectorialization (simmetrical) optimised for the 2d dft.
    
    Parameters
    ----------
    v : np.array
        vectorialized signal
    shape : tuple
        original shape
    
    Returns
    -------
    x : np.ndarray
        matrix form of v
    """
    if (len(v.shape) == 2):
        l = v.shape[0]
        X = np.zeros((l, *shape), v.dtype)
        for i in range(l):
            X[i] = sim_spiral_mat(v[i], shape)
        return X
    h, w = shape
    rs, re = 0, h - 1
    cs, ce = 0, w - 1
    j = 0
    l = v.size
    i = l-1
    x = np.zeros(shape, v.dtype)
    while j != i:
        x[rs, np.arange(cs, ce+1)] = v[j:j+w] 
        rs +=1 
        j += w
        h -= 1
        if j >= l:
            break
        
        x[re, np.arange(ce ,cs-1,-1)] = v[np.arange(i, i-w,-1)]
        i -= w
        h -= 1
        re -= 1
        if j >= i:
            break
        
        x[np.arange(rs, re+1), ce] = v[j:j+h]
        j += h
        w -= 1
        ce -= 1
        if j >= i:
            break
            
        x[np.arange(re, rs-1,-1), cs] = v[np.arange(i, i-h, -1)]
        i -= h
        w -= 1
        cs += 1
        if j >= i:
            break
        
        x[re, np.arange(ce, cs-1,-1)] = v[j:j+w]
        re -=1 
        j += w
        h -= 1
        if j >= l:
            break
            
        x[rs, np.arange(cs ,ce+1)] = v[np.arange(i, i-w,-1)]
        i -= w
        h -= 1
        rs += 1
        if j >= i:
            break
            
        x[np.arange(re, rs-1,-1), cs] = v[j:j+h]
        j += h
        w -= 1
        cs += 1
        if j >= i:
            break
            
        x[np.arange(rs, re+1), ce] = v[np.arange(i, i-h, -1)]
        i -= h
        w -= 1
        ce -= 1
        if j >= i:
            break
    return x


def default_vect(input):
    """
    Numpy default vectorization.
    Parameters
    ----------
    input : np.ndarray
        matrix signal
    
    Returns
    -------
    v : np.array
        vectorialized signal
    """
    if len(input.shape) == 3: 
        L = len(input)
        return input.reshape((L, -1))
    elif len(input.shape) != 2:
        raise ValueError("Input must be 2D or 3D.")
    return input.reshape(-1)


def default_mat(input, shape):
    """
    Numpy default inverse-vectorization.
    
    Parameters
    ----------
    input : np.array
        vectorialized signal
    shape : tuple
        original shape
    
    Returns
    -------
    x : np.ndarray
        matrix form of v
    """
    if len(input.shape) == 2: 
        L = len(input)
        res = (L, *shape)        
    elif len(input.shape) != 1:
        raise ValueError("Input must be 1D or 2D.")
    else:
        res = shape
    if np.prod(res) != input.size:
        raise Exception("Input size does not match output shape.")
    return input.reshape(res)


def zigzag_vect(input):
    """
    JPG Zig-Zag vectorialization optimised for the 2d dct.
     
    Parameters
    ----------
    input : np.ndarray
        matrix signal
    
    Returns
    -------
    v : np.array
        vectorialized signal
    """
    if len(input.shape) == 3: 
        L = len(input)
        S = input.shape[1]*input.shape[2]
        res = np.zeros((L, S), input.dtype)
        for i in range(L):
            res[i] = zigzag_vect(input[i])
        return res
    elif len(input.shape) != 2:
        raise ValueError("Input must be 2D or 3D.")
        
    global __zsupp__
    if __zsupp__ is None or not __zsupp__.size == input.size: 
        __zsupp__ = __gen_zsupp__(input.shape)   
    return input.reshape(-1)[__zsupp__]


def zigzag_mat(input, shape):
    """
    JPG Zig-Zag inverse-vectorialization optimised for the 2d dct.
     
    Parameters
    ----------
    input : np.array
        vectorialized signal
    shape : tuple
        original shape
    
    Returns
    -------
    output : np.ndarray
        matrix form of v
    """
    if len(input.shape) == 2: 
        L = len(input)
        res = np.zeros((L, *shape), input.dtype)
        for i in range(L):
            res[i] = zigzag_mat(input[i], shape)
        return res
    elif len(input.shape) != 1:
        raise ValueError("Input must be 1D or 2D.")
    if np.prod(shape) != input.size:
        raise Exception("Input size does not match output shape.")
    global __zsupp__
    global __zppus__
    if __zppus__ is None or not __zppus__.size == np.prod(shape):
        if __zsupp__ is None or not __zsupp__.size == np.prod(shape):
            __zsupp__ = __gen_zsupp__(shape)
        __zppus__ = inv(__zsupp__)
    return input[__zppus__].reshape(shape)

def circ_vect(X):
    """
    Circular vectorialization for the 2d dft.
    
    Parameters
    ----------
    x : np.ndarray
        matrix signal
    
    Returns
    -------
    v : np.array
        vectorialized signal
    """
    if len(X.shape) == 3:
        l, w, h = X.shape
        x = np.zeros((l, w*h), X.dtype)
        for i in range(l):
            x[i] = circ_vect(X[i])
        return x
    elif len(X.shape) != 2:
        raise ValueError("Input must be 2D or 3D.")
    global __nsupp__
    if __nsupp__ is None or not __nsupp__.size == X.size: 
        __nsupp__ = __gen_nsupp__(X.shape)   
    return X.reshape(-1)[__nsupp__]


def circ_mat(v, shape):
    """
    Circular inverse-vectorialization optimised for the 2d dft.
     
    Parameters
    ----------
    v : np.array
        vectorialized signal
    shape : tuple
        original shape
    
    Returns
    -------
    x : np.ndarray
        matrix form of v
    """
    if len(v.shape) == 2:
        l = v.shape[0]
        X = np.zeros((l, *shape), v.dtype)
        for i in range(l):
            X[i] = circ_mat(v[i], shape)
        return X
    elif len(v.shape) != 1:
        raise ValueError("Input must be 1D or 2D.")
    if np.prod(shape) != v.size:
        raise Exception("Input size does not match output shape.")
        
    global __nsupp__
    global __nppus__
    if __nppus__ is None or not __nppus__.size == np.prod(shape):
        if __nsupp__ is None or not __nsupp__.size == np.prod(shape):
            __nsupp__ = __gen_nsupp__(shape)
        __nppus__ = inv(__nsupp__)
    return v[__nppus__].reshape(shape)


def sim_circ_vect(X):
    """
    Circular simmetric vectorialization for the 2d dft.
    
    Parameters
    ----------
    x : np.ndarray
        matrix signal
    
    Returns
    -------
    v : np.array
        vectorialized signal
    """
    if len(X.shape) == 3:
        l, w, h = X.shape
        x = np.zeros((l, w*h), X.dtype)
        for i in range(l):
            x[i] = sim_circ_vect(X[i])
        return x
    elif len(input.shape) != 2:
        raise ValueError("Input must be 2D or 3D.") 
        
    global __supp__
    if __supp__ is None or not __supp__.size == X.size: 
        __supp__ = __gen_supp__(X.shape)   
    return X.reshape(-1)[__supp__]


def sim_circ_mat(v, shape):
    """
    Circular simmetric inverse-vectorialization optimised for the 2d dft.
     
    Parameters
    ----------
    v : np.array
        vectorialized signal
    shape : tuple
        original shape
    
    Returns
    -------
    x : np.ndarray
        matrix form of v
    """
    if len(v.shape) == 2:
        l = v.shape[0]
        X = np.zeros((l, *shape), v.dtype)
        for i in range(l):
            X[i] = sim_circ_mat(v[i], shape)
        return X
    elif len(v.shape) != 1:
        raise ValueError("Input must be 1D or 2D.")
    if np.prod(shape) != v.size:
        raise Exception("Input size does not match output shape.")
        
    global __supp__
    global __ppus__
    if __ppus__ is None or not __ppus__.size == np.prod(shape):
        if __supp__ is None or not __supp__.size == np.prod(shape):
            __supp__ = __gen_supp__(shape)
        __ppus__ = inv(__supp__)
    return v[__ppus__].reshape(shape)


def pos(x):
    """
    Trunkate negative values.

    Parameters
    ----------
    x : np.array
        A signal

    Returns
    -------
    y : np.array
        Positive only signal
    """
    y = x.copy()
    y[y < 0] = 0
    return y

def norm(x):
    """
    Normalizes a signal between 1 and 0.

    Parameters
    ----------
    x : np.array
        A signal

    Returns
    -------
    y : np.array
        The normalize signal (0-1)
    """
    y = np.array(x)
    y -= np.min(y)
    return y/np.max(y)


def load_image(path, size=None):
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
    x = iio.imread(path, as_grey=True)
    if size is not None:
        x = resize(x, size, mode='reflect')
    return norm(x)


def load_images(folder, fformat=".png", size=None):
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
    filt = lambda x, y : isfile(x) and x.endswith(y)
    onlyfiles = [f for f in sorted(listdir(folder)) if filt(join(folder,f), fformat)]
    dataset = []
    for im_path in onlyfiles:
        i = load_image(folder+'/'+im_path, size)
        dataset.append(i)
    return np.array(dataset)


def count_images(folder, fformat='.png'):
    """Counts all images in a folder."""
    filt = lambda x, y : isfile(x) and x.endswith(y)
    onlyfiles = [f for f in sorted(listdir(folder)) if filt(join(folder,f), fformat)]
    return len(onlyfiles)


def inv(order):
    """
    Inverses sorting order.

    Parameters
    ----------
    order : np.array (int)
        A sorting array

    Returns
    -------
    redro : np.array (int)
        The inverse order
    """
    redro = np.zeros(order.shape, int)
    for i, j in enumerate(order):
        redro[j] = i
    return redro


def load_dataset(path, fformat=".png", size=None):
    """
    Load a dataset and tranfsorm it using fft2 function.

    Parameters
    ----------
    path : str
        The path to the folder containing the dataset
    fformat : str
        Format images

    Returns
    -------
    s : np.ndarray
        The loaded dataset in the spatial domain
    S : np.ndarray (complex)
        The loaded dataset in the frequency domain
    """
    s = load_images(path, fformat=fformat, size=size)
    S = np.array([fft2(x) for x in s])

    return s, S


def retrive_basic_stats(dataset):
    """
    Retrive basic statistics from the dataset in the frequency domain.

    Parameters
    ----------
    dataset : np.ndarray (complex)
        The dataset in the frequency domain
    Returns
    -------
    mag : np.ndarray
        Mean frequency magnetude of the dataset
    mag_std : np.ndarray
        Frequency standard deviation (STD) of the dataset magnetude
    phs : np.ndarray
        Mean frequency phase of the dataset
    phs_std : np.ndarray
        Frequency standard deviation (STD) of the dataset phase
    """
    mag = np.mean(np.abs(dataset), 0)
    mag_std = np.std(np.abs(dataset), 0)
    phs = np.mean(np.angle(dataset), 0)
    phs_std = np.std(np.angle(dataset), 0)
    return mag, mag_std, phs, phs_std


def Psi_dft(l):
    """
    Generate the Psi ortho-normal matrix for the dft transform.
    
    Parameters
    ----------
    l : int
        The size of the signal
    
    Returns
    -------
    Psi : np.ndarray (complex128)
        The computed ortho-normal matrix
    """
    Psi = np.zeros((l, l), np.complex128)
    ns = np.arange(l)
    l2 = l/2
    ks = np.linspace(-l2, l2-1, l)
    for i, k in enumerate(ks): 
        Psi[i] = np.exp(-2j*np.pi*k*ns/l)
    return Psi


def noise(sigma, shape):
    """Generates complex Gaussian noise."""
    return np.random.normal(0, sigma, shape) * np.exp(1j * np.random.rand(shape) * 2 * np.pi)


def SER(x, xrec):
    """Computes SER as: ser(x, xrec) = 10 * log10 (sum(x^2) / sum((x-xrec)^2)) 
    """
    return 10 * np.log10( np.sum(x**2) / np.sum((x - xrec) ** 2))
