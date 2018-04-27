#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utilities for Multi Band Reconstruction
Created on Fri Feb  2 18:32:31 2018

@author: Martino Ferrari
@email: manda.mgf@gmail.com
"""

import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import k_means_


def version():
    """Library version."""
    return "1.0.0" 


def __ang_distance__(X, Y=None, Y_norm_squared=None, squared=False):
    """Internal: ANGULAR DISTANCE"""
    assert Y is not None
    W, N = X.shape
    H, M = Y.shape
    assert M == N
    distance = np.zeros((W, H))
    for j in range(H):
        distance[:, j] =  np.sqrt(np.sum(((X-Y[j]) % (2 * np.pi))**2, 1))
    return distance


class Codebook:
    """
    Codebook object for complex signal.

    Methods
    -------
     * predict
     * compress
     * reconstruct

    Constructor Paramters
    ---------------------
    re_codes : np.ndarray
        codes for real part
    im_codes : np.ndarray
        codes for imag part
    """
    def __init__(self, re_codes, im_codes):
        self.__re_codes__ = re_codes
        self.__im_codes__ = im_codes

    def predict(self, X, omega=None):
        """
        Find codes that represent better the signal X.
        TO be implemented in each sub classes

        Paramters
        ---------
        X : np.array
            Input signal
        omega : np.array
            Possible mask

        Returns
        -------
        ire : int
            Id of the real code
        iim : int
            Id of the imaginary code
        """
        return -1, -1

    def compress(self, X, omega=None):
        """
        Predict and compress a signal using the codebook.
        TO be implemented in sub classes.

        Paramters
        ---------
        X : np.array
            Input signal
        omega : np.array
            Possible mask

        Returns
        -------
        Y : np.array
            Compressed signal
        """
        return None

    def reconstruct(self, X, omega):
        """
        Predict and reconstruct a signal using the codebook.

        Paramters
        ---------
        X : np.array
            Input signal
        omega : np.array
            Possible mask

        Returns
        -------
        Y : np.array
            Reconstructed signal
        """
        Y = self.compress(X, omega)
        Y[omega] = X[omega]
        return Y

    def sampling_pattern(self):
        """
        Computes an optimal (or close to optimal) sampling pattern.

        Returns
        -------
        omega : np.array
            Ordered list of index to sample
        """
        std_re = np.std((self.__re_codes__), 0)
        std_im = np.std((self.__im_codes__), 0)

        std_im /= np.max(std_im)
        std_re /= np.max(std_re)
        omega = (std_re + std_im).argsort()[::-1]
        return omega


class ReImCodebook(Codebook):
    """
    Codebook object for complex signal.

    Methods
    -------
     * predict
     * compress
     * reconstruct

    Constructor Paramters
    ---------------------
    re_codes : np.ndarray
        codes for real part
    im_codes : np.ndarray
        codes for imag part
    """
    def __init__(self, re_codes, im_codes):
        Codebook.__init__(self, re_codes, im_codes)

    def predict(self, X, omega=None):
        """
        Find the real and imaginary codes that represent better the signal X.

        Paramters
        ---------
        X : np.array
            Input signal
        omega : np.array
            Possible mask

        Returns
        -------
        ire : int
            Id of the real code
        iim : int
            Id of the imaginary code
        """
        if omega is None:
            omega = np.arange(len(X))
        es_re = np.sqrt(np.sum((self.__re_codes__[:, omega] - X[omega].real)**2, 1))
        es_im = np.sqrt(np.sum((self.__im_codes__[:, omega] - X[omega].imag)**2, 1))
        es_re /= np.sum(es_re)
        es_im /= np.sum(es_im)
        """
        TODO
        ----
        Use phase and magnetude to increase precision such that:

            |code_re + j*code_im| ~ |X|
            angle(code_re + j*code_im) ~ angle(X)

        The phase is particuarly important.
        """


        better_re = es_re.argsort()[:10]
        better_im = es_im.argsort()[:10]

        mag_errors = np.zeros(len(better_re)*len(better_im))
        other_errors = np.zeros(len(better_re)*len(better_im))
        phs_errors = np.zeros(len(better_re)*len(better_im))
        keys = []

        x_mag = np.abs(X[omega])
        x_phs = np.angle(X[omega])

        c = 0
        for i in better_re:
            for j in better_im:
                rec = self.__re_codes__[i][omega] + 1j * \
                    self.__im_codes__[j][omega]
                mag = np.abs(rec)
                phs = np.angle(rec)

                mag_errors[c] = np.sqrt(np.sum((mag - x_mag)**2))
                other_errors[c] = es_re[i] + es_im[j]

                phs_errors[c] = np.sqrt(np.sum(((phs - x_phs) % (2*np.pi))**2))

                keys.append((i, j))
                c += 1
        errors = mag_errors/np.sum(mag_errors) + other_errors + phs_errors/np.sum(phs_errors)
        best_couple = errors.argmin()
        ire, iim = keys[best_couple]


        ire, iim = es_re.argmin(), es_im.argmin()

        return ire, iim

    def compress(self, X, omega=None):
        """
        Predict and compress a signal using the codebook.

        Paramters
        ---------
        X : np.array
            Input signal
        omega : np.array
            Possible mask

        Returns
        -------
        Y : np.array
            Compressed signal
        """
        ire, iim = self.predict(X, omega)
        Y = self.__re_codes__[ire] + 1j*self.__im_codes__[iim]
        return Y


class MagPhsCodebook(Codebook):
    """
    Codebook object for complex signal.

    Methods
    -------
     * predict
     * compress
     * reconstruct

    Constructor Paramters
    ---------------------
    re_codes : np.ndarray
        codes for real part
    im_codes : np.ndarray
        codes for imag part
    """
    def __init__(self, re_codes, im_codes):
        Codebook.__init__(self, re_codes, im_codes)

    def predict(self, X, omega=None):
        """
        Find the real and imaginary codes that represent better the signal X.

        Paramters
        ---------
        X : np.array
            Input signal
        omega : np.array
            Possible mask

        Returns
        -------
        ire : int
            Id of the real code
        iim : int
            Id of the imaginary code
        """
        if omega is None:
            omega = np.arange(len(X))
        es_mag = np.mean(np.abs(self.__re_codes__[:, omega] - np.abs(X[omega])), 1)
        es_phs = np.mean(np.abs((self.__im_codes__[:, omega] - np.angle(X[omega])) % (2 * np.pi)), 1)

        es_mag /= np.sum(es_mag)
        es_phs /= np.sum(es_phs)

        """
        TODO
        ----
        Use phase and magnetude to increase precision such that:

            |code_re + j*code_im| ~ |X|
            angle(code_re + j*code_im) ~ angle(X)

        The phase is particuarly important.
        """
        better_mag = es_mag.argsort()[:10]
        better_phs = es_phs.argsort()[:10]

        re_errors = np.zeros(len(better_phs)*len(better_mag))
        im_errors = np.zeros(len(better_phs)*len(better_mag))

        keys = []

        x_re = X[omega].real
        x_im = X[omega].imag

        c = 0
        for i in better_mag:
            for j in better_phs:
                rec = self.__re_codes__[i][omega] * np.exp( 1j * \
                    self.__im_codes__[j][omega])

                re = rec.real
                im = rec.imag

                re_errors[c] = np.mean(np.abs(re - x_re))
                im_errors[c] = np.mean(np.abs(im - x_im))
                keys.append((i, j))
                c += 1
        errors = re_errors/np.sum(re_errors) + im_errors/np.sum(im_errors)
        best_couple = errors.argmin()
        ire, iim = keys[best_couple]
        return ire, iim

    def compress(self, X, omega=None):
        """
        Predict and compress a signal using the codebook.

        Paramters
        ---------
        X : np.array
            Input signal
        omega : np.array
            Possible mask

        Returns
        -------
        Y : np.array
            Compressed signal
        """
        ire, iim = self.predict(X, omega)
        Y = self.__re_codes__[ire] * np.exp(1j*self.__im_codes__[iim])
        return Y


def divide(comulated, n):
    """
    Divides comulative energy in equi-energetic bands.

    Parameters
    ----------
    comulated : np.array
        Comulative energy
    n : int
        Number of sub-bands

    Returns
    -------
    bands : list of int
        List of indexes where to split the signal
    """
    min_size = 2
    bands = []
    thr = 1.0/n
    prev = 0
    for i, e in enumerate(comulated):
        if e >= thr: 
            if i - prev < min_size:
                bands.append(prev+min_size)
                prev += min_size
            else:
                bands.append(i)
                prev = i
            thr += 1.0/n
    # force last band to be complete
    if len(bands) == n:
        bands[-1] = len(comulated)
    else:
        bands.append(len(comulated))
    return tuple(bands)


def split(X, bands):
    """
    Splits a signal or a set of signals in sub-bands.

    Parameters
    ----------
    X : np.array or np.ndarray
        Signal or set of signals
    bands : list of int
        Positions where to split the input

    Returns
    -------
    Xsb : tuple of np.array or np.ndarray
        Tuple containing the sub-bands.
    """
    Xsb = []
    old_pos = 0
    is_set = len(X.shape) == 2
    for pos in bands:
        if is_set:
            Xsb.append(X[:, old_pos:pos])
        else:
            Xsb.append(X[old_pos:pos])
        old_pos = pos
    return tuple(Xsb)


def union(Xsb):
    """
    Merge a signal or set of signals sub-bands in a complete one.

    Paramters
    ---------
    Xsb : tuple of np.array or np.ndarray
        Tuple containing the sub-bands

    Returns
    -------
    X : np.array or np.ndarray
        Merged signal or set of signals
    """
    is_set = len(Xsb[0].shape) == 2
    if is_set:
        N = np.sum([x.shape[1] for x in Xsb])
        L = Xsb[0].shape[0]
        X = np.zeros((L, N), dtype=Xsb[0].dtype)
    else:
        N = np.sum([len(x) for x in Xsb])
        X = np.zeros(N, dtype=Xsb[0].dtype)
    old_pos = 0
    for sb in Xsb:
        pos = old_pos + (len(sb) if not is_set else sb.shape[1])
        if is_set:
            X[:, old_pos:pos] = sb
        else:
            X[old_pos:pos] = sb
        old_pos = pos
    return X


def num_samples(bands, m):
    """
    Computes number of samples for each band.

    Parameters
    ----------
    bands : list or tuple
        band definition

    Returns
    -------
    samples : list
        number of samples per band
    """
    samples = [0 for _ in range(len(bands))]
    sizes = []
    old_pos = 0
    for pos in bands:
        n = pos - old_pos
        sizes.append(n)
        old_pos = pos
    order = np.argsort(sizes)

    rest = 0
    for i in order:
        n = sizes[i]
        mm = m + rest
        if mm > n:
            rest = mm - n
            mm = n
        else:
            rest = 0
        samples[i]=mm
    return samples


def sub_sample(Xbs, omegas, m):
    """
    Sub-sampling multi band signal.

    Parameters
    ----------
    Xbs : tuple of np.array
        Multi-band signal
    omegas : list
        Sampling pattern per band
    m : int
        Approximative number of samples per band

    Returns
    -------
    Ybs : tuple of np.array
        Sub-sampled multi band signal
    """
    ords = np.argsort([len(x) for x in Xbs])
    Ybs = [None for _ in range(len(omegas))]
    rest = 0
    for i in ords:
        omega = omegas[i]
        mm = m + rest
        n = len(Xbs[i])
        if mm > n:
            rest = mm - n
            mm = n
        else:
            rest = 0
        Y = Xbs[i].copy()
        Y[omega[mm:]] = 0
        Ybs[i] = Y
    return Ybs


def gen_re_im_codebooks(Xsbs, n_codes, batch_size=300):
    """
    Generates Real/Imag codebooks for each sub-band.

    Paramters
    ---------
    Xsbs : tuple of np.ndarray
        Training in the frequency domain set divided in sub-bands (optionally
        normalized)
    n_codes : int
        Number of codes for each codebook
    batch_size : int
        Number of images per batch

    Returns
    -------
    codebooks : tuple of Codebook
        The generated codebooks for real and imaginary part
    """
    codebooks = []
    for sb in Xsbs:
        re_sb = sb.real
        re_kms = MiniBatchKMeans(n_clusters=n_codes, batch_size=batch_size)
        re_kms.fit(re_sb)
        im_sb = sb.imag
        im_kms = MiniBatchKMeans(n_clusters=n_codes, batch_size=batch_size)
        im_kms.fit(im_sb)
        codebooks.append(ReImCodebook(re_kms.cluster_centers_,
                                      im_kms.cluster_centers_))
    return tuple(codebooks)


def gen_mag_phs_codebooks(Xsbs, n_codes, batch_size=300):
    """
    Generates Magnetude/Phase codebooks for each sub-band.

    Paramters
    ---------
    Xsbs : tuple of np.ndarray
        Training in the frequency domain set divided in sub-bands (optionally
        normalized)
    n_codes : int
        Number of codes for each codebook
    batch_size : int
        Number of images per batch

    Returns
    -------
    codebooks : tuple of Codebook
        The generated codebooks for real and imaginary part
    """
    codebooks = []
    real_func = k_means_.euclidean_distances
    for sb in Xsbs:
        re_sb = np.abs(sb)

        k_means_.euclidean_distances = real_func
        re_kms = MiniBatchKMeans(n_clusters=n_codes, batch_size=batch_size)
        re_kms.fit(re_sb)
        im_sb = np.abs((np.angle(sb) + np.pi) % (2 * np.pi))

        k_means_.euclidean_distances = __ang_distance__
        im_kms = MiniBatchKMeans(n_clusters=n_codes, batch_size=batch_size)
        im_kms.fit(im_sb)
        codebooks.append(ReImCodebook(re_kms.cluster_centers_,
                                      im_kms.cluster_centers_))
    k_means_.euclidean_distances = real_func
    return tuple(codebooks)


def gen_codebooks(Xsbs, n_codes, mode='ReIm', batch_size=300):
    """
    Generates Real/Imag codebooks for each sub-band.

    Paramters
    ---------
    Xsbs : tuple of np.ndarray
        Training in the frequency domain set divided in sub-bands (optionally
        normalized)
    n_codes : int
        Number of codes for each codebook
    mode : str
        Codebook mode: 'ReIm' or 'MagPhs'
    batch_size : int
        Number of images per batch

    Returns
    -------
    codebooks : tuple of Codebook
        The generated codebooks for real and imaginary part
    """
    if mode == 'ReIm':
        return gen_re_im_codebooks(Xsbs, n_codes, batch_size)
    return gen_mag_phs_codebooks(Xsbs, n_codes, batch_size)




