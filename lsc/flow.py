# -*- coding: utf-8 -*-
"""
LSC flow blocks.

@author: Martino Ferrari
@email: manda.mgf@gmail.com
"""
import os
import os.path

import pickle
import numpy as np

import lsc
import common

from skimage.measure import compare_mse as mse
from skimage.measure import compare_ssim as ssim


def training(conf):
    """Training step.

    Parameters
    ----------
    conf : conf_loader.Conf
        Experiment parameters

    Returns
    -------
    (sort, tros) : ([int], [int])
        Sorting order, direct and inverse
    (energy, comulated, bands) : (np.array, np.array, [int])
        Average energy, comulated average energy and band splitting
    codebooks : [Codebook]
        codebooks per band
    """
    print('Loading training dataset...', end='', flush=True)
    n_bands = conf.nbands
    # matrxi to vector (m2v) and vector to matrix (v2m) functions
    m2v, v2m = conf.vect_functions()
    training, Training = common.load_dataset(conf.trainingset_path(),
                                             fformat=conf.fformat,
                                             size=conf.size())
    print(' [done]')
    """
    Extract stastics.
    """
    print('Extracting statistics and computing bands...', end='', flush=True)

    shape = training[0].shape
    n = np.prod(shape)

    mag, mag_std, phs, phs_std = common.retrive_basic_stats(Training)

    # direct sorting
    sort = np.arange(n)
    if conf.training['sort'] == 'random':
        sort = np.random.choice(n, size=n, replace=False)
    elif conf.training['sort'] == 'energy':
        sort = m2v(mag).argsort()[::-1]
    # inverse sorting
    tros = common.inv(sort)

    """
    Generate bands.
    """
    energy = common.norm(m2v(mag_std)[sort])
    comulated = common.norm(common.comulate(energy))
    bands = lsc.divide(comulated, n_bands)
    print(' [done]')

    """
    Check if codebook is cached.
    """
    if not os.path.isdir('codebooks'):
        os.mkdir('codebooks')
    codebook_path = f'codebooks/{conf.codebook_name()}'
    if os.path.isfile(codebook_path):
        print('Loading existing codebooks...' ,end='', flush=True)
        with open(codebook_path, 'rb') as f:
            codebooks = pickle.load(f)
    else:
        print('Computing codebooks...' ,end='', flush=True)
        """
        Prepare dataset for codes generation.
        """
        n_levels = conf.training['n_levels']
        n_codes = conf.training['n_codes']
        batch_size = conf.training['batch']

        normalize = m2v(mag)[sort]
        discretize = normalize/n_levels

        FlatTrn = m2v(Training)[:, sort]
        DiscTrn = np.round(FlatTrn/discretize) * discretize
        SBsTrn = lsc.split(DiscTrn, bands)

        """
        Compute codebook for each sub-band.
        """

        codebooks = lsc.gen_codebooks(SBsTrn, n_codes, mode='ReIm',
                                         batch_size=batch_size)
        with open(codebook_path, 'wb') as f:
            pickle.dump(codebooks, f)
    print(' [done]')
    return (sort, tros), (energy, comulated, bands), codebooks


def test_sampling(conf, training_result):
    """
    Sampling rate vs Reconstruction Quality test.

    Parameters
    ----------
    conf : conf_loader.Conf
        Experiment parameters
    training_result : touple
        Retrun values of function `training`

    Returns
    -------
    srange : np.array
        sampling rate used
    (bk_mse, fa_mse, rc_mse) : (np.array, np.array, np.array)
        mse for `k-best`, `f_avg` and `LSC` methods at different sampling rate
    (bk_ssim, fa_ssim, rc_ssim) : (np.array, np.array, np.array)
        ssim for `k-best`, `f_avg` and `LSC` methods at different saapling rate
    """
    # matrxi to vector (m2v) and vector to matrix (v2m) functions
    m2v, v2m = conf.vect_functions()
    (sort, tros), (energy, comulated, bands), codebooks = training_result
    n_bands = conf.nbands
    subcfg = conf.testing['reconstruction']

    # Load testing set
    testing, Testing = common.load_dataset(conf.testingset_path(),
                                           conf.fformat, conf.size())
    FlatTst = m2v(Testing)[:, sort]

    # f_avg sampling pattern
    Omega = energy.argsort()[::-1]

    # lsc sampling pattern
    Omegas = [c.sampling_pattern() for c in codebooks]

    shape = testing[0].shape
    n = np.prod(shape)
    N = len(testing)

    # sampling rate range
    srange = np.logspace(*subcfg['sampling_range'])

    # results accumulator
    bk_mse = np.zeros(len(srange))
    fa_mse = np.zeros(len(srange))
    rc_mse = np.zeros(len(srange))

    bk_ssim = np.zeros(len(srange))
    fa_ssim = np.zeros(len(srange))
    rc_ssim = np.zeros(len(srange))


    print('Sampling Rate vs Reconstruction Quality Test:')
    for i, rate in enumerate(srange):
        print(f'\r {i+1:3d}/{len(srange)}', flush=True, end='')
        M = int(round(n*rate))
        m = int(round(M/n_bands))
        ms = lsc.num_samples(bands, m)
        M = np.sum(ms)

        smalls = [omega[:y] for omega, y in zip(Omegas, ms)]

        for idx in range(N):
            reference = common.norm(testing[idx])
            X = FlatTst[idx]

            Xsbs = lsc.split(X, bands)
            Ysbs = lsc.sub_sample(Xsbs, Omegas, m)
            recovered = [codebooks[b].reconstruct(Ysbs[b], smalls[b]) for b in range(len(bands))]
            Y = v2m((lsc.union(recovered))[tros], shape)
            y = common.norm(common.pos(common.ifft2(Y).real))


            BK = X.copy()[tros]
            # BK sampling pattern
            O = np.abs(BK).argsort()[::-1]
            BK[O[M:]] = 0
            BK = v2m(BK, shape)
            bK = common.norm(common.pos(common.ifft2(BK).real))

            FA = X.copy()[tros]
            FA[Omega[M:]] = 0
            FA = v2m(FA, shape)
            fA = common.norm(common.pos(common.ifft2(FA).real))

            fa_mse[i] += mse(reference, fA) / N
            bk_mse[i] += mse(reference, bK) / N
            rc_mse[i] += mse(reference, y) / N

            fa_ssim[i] += ssim(reference, fA, gaussian_weights=True) / N
            bk_ssim[i] += ssim(reference, bK, gaussian_weights=True) / N
            rc_ssim[i] += ssim(reference, y, gaussian_weights=True) / N
    print('\t[done]')
    return srange, (bk_mse, fa_mse, rc_mse), (bk_ssim, fa_ssim, rc_ssim)


def test_sampling_visual(conf, training_result, idx):
    """
    Sampling rate vs Reconstruction Quality visual test.

    Parameters
    ----------
    conf : conf_loader.Conf
        Experiment parameters
    training_result : touple
        Retrun values of function `training`
    idx : int
        Id image to test

    Returns
    -------
    srange : np.array
        sampling rate used
    res : np.ndarray
        image containing visual results of reconstruction from `k-best`,
        `f_avg` and `LSC` at diffrent sampling rate.
    """
    (sort, tros), (energy, comulated, bands), codebooks = training_result
    n_bands = conf.nbands
    # matrxi to vector (m2v) and vector to matrix (v2m) functions
    m2v, v2m = conf.vect_functions()
    subcfg = conf.testing['reconstruction_visual']

    testing, Testing = common.load_dataset(conf.testingset_path(),
                                           conf.fformat, conf.size())
    FlatTst = m2v(Testing)[:, sort]

    # f_avg sampling pattern
    Omega = energy.argsort()[::-1]

    # lsc sampling pattern
    Omegas = [c.sampling_pattern() for c in codebooks]

    shape = testing[0].shape
    n = np.prod(shape)

    srange = np.logspace(*subcfg['sampling_range'])

    X = FlatTst[idx].copy()

    W, H = shape
    Wt = W * 3
    Ht = H * len(srange)

    res = np.zeros((Wt, Ht))

    print(f'Sampling Rate vs Reconstruction Quality Visual Test (img {idx}):')
    for i, rate in enumerate(srange):
        print(f'\r {i+1:3d}/{len(srange)}', flush=True, end='')

        M = int(round(rate*n))
        m = int(round(M/n_bands))
        ms = lsc.num_samples(bands, m)
        M = np.sum(ms)

        smalls = [omega[:y] for omega, y in zip(Omegas, ms)]

        Xsbs = lsc.split(X, bands)
        Ysbs = lsc.sub_sample(Xsbs, Omegas, m)

        recovered = [codebooks[b].reconstruct(Ysbs[b], smalls[b]) for b in range(len(bands))]
        Y = v2m((lsc.union(recovered))[tros], shape)
        y = common.norm(common.pos(common.ifft2(Y).real))


        BK = X.copy()[tros]
        O = np.abs(BK).argsort()[::-1]
        BK[O[M:]] = 0
        BK = v2m(BK, shape)
        bK = common.norm(common.pos(common.ifft2(BK).real))

        FA = X.copy()[tros]
        FA[Omega[M:]] = 0
        FA = v2m(FA, shape)
        fA = common.norm(common.pos(common.ifft2(FA).real))

        res[:W, H*i:H*(i+1)] = bK
        res[W:2*W, H*i:H*(i+1)] = fA
        res[2*W:3*W, H*i:H*(i+1)] = y
    print('\t[done]')
    return srange, res


def test_robust(conf, training_result):
    """
    Noise std vs Reconstruction Quality at fixed sampling rate test.

    Parameters
    ----------
    conf : conf_loader.Conf
        Experiment parameters
    training_result : touple
        Retrun values of function `training`

    Returns
    -------
    srange : np.array
        noise std range used
    sampling_rate : float
        sampling rate used
    (bk_ser, fa_ser, rc_ser) : (np.array, np.array, np.array)
        SER for `k-best`, `f_avg` and `LSC` methods at different noise std
    (bk_mse, fa_mse, rc_mse) : (np.array, np.array, np.array)
        mse for `k-best`, `f_avg` and `LSC` methods at different noise std
    """
    (sort, tros), (energy, comulated, bands), codebooks = training_result
    n_bands = conf.nbands
    # matrxi to vector (m2v) and vector to matrix (v2m) functions
    m2v, v2m = conf.vect_functions()
    subcfg = conf.testing['robust_reconstruction']

    # Load testing set
    testing, Testing = common.load_dataset(conf.testingset_path(),
                                           conf.fformat, conf.size())
    FlatTst = m2v(Testing)[:, sort]

    # f_avg sampling pattern
    Omega = energy.argsort()[::-1]

    # lsc sampling pattern
    Omegas = [c.sampling_pattern() for c in codebooks]

    shape = testing[0].shape
    n = np.prod(shape)
    N = len(testing)

    srange = np.logspace(*subcfg['noise_range'])
    sampling_rate = subcfg['sampling_rate']


    bk_ser = np.zeros(len(srange))
    fa_ser = np.zeros(len(srange))
    rc_ser = np.zeros(len(srange))

    bk_mse = np.zeros(len(srange))
    fa_mse = np.zeros(len(srange))
    rc_mse = np.zeros(len(srange))

    print('Robust recovering test:')
    for i, sigma in enumerate(srange):
        print(f'\r {i+1:3d}/{len(srange)}', end='', flush=True)
        M = int(round(n*sampling_rate))
        m = int(round(M/n_bands))
        ms = lsc.num_samples(bands, m)
        M = np.sum(ms)

        smalls = [omega[:y] for omega, y in zip(Omegas, ms)]

        for idx in range(N):
            reference = common.norm(testing[idx])
            X = FlatTst[idx] + common.noise(sigma, n)

            Xsbs = lsc.split(X, bands)
            Ysbs = lsc.sub_sample(Xsbs, Omegas, m)
            recovered = [codebooks[b].reconstruct(Ysbs[b], smalls[b]) for b in range(len(bands))]
            Y = v2m((lsc.union(recovered))[tros], shape)
            y = common.norm(common.pos(common.ifft2(Y).real))

            BK = X.copy()[tros]
            O = np.abs(BK).argsort()[::-1]
            BK[O[M:]] = 0
            BK = v2m(BK, shape)
            bK = common.norm(common.pos(common.ifft2(BK).real))

            FA = X.copy()[tros]
            FA[Omega[M:]] = 0
            FA = v2m(FA, shape)
            fA = common.norm(common.pos(common.ifft2(FA).real))

            fa_ser[i] += common.SER(reference, fA) / N
            bk_ser[i] += common.SER(reference, bK) / N
            rc_ser[i] += common.SER(reference, y) / N

            fa_mse[i] += mse(reference, fA) / N
            bk_mse[i] += mse(reference, bK) / N
            rc_mse[i] += mse(reference, y) / N
    print(' [done]')
    return srange, sampling_rate, (bk_ser, fa_ser, rc_ser), (bk_mse, fa_mse, rc_mse)


def test_robust_visual(conf, training_result, idx):
    """
    Noise std vs Reconstruction Quality at fixed sampling rate visual test.

    Parameters
    ----------
    conf : conf_loader.Conf
        Experiment parameters
    training_result : touple
        Retrun values of function `training`
    idx : int
        Id of image to test

    Returns
    -------
    srange : np.array
        noise std range used
    sampling_rate : float
        sampling rate used
    (bk_ser, fa_ser, rc_ser) : (np.array, np.array, np.array)
        SER for `k-best`, `f_avg` and `LSC` methods at different noise std
    (bk_mse, fa_mse, rc_mse) : (np.array, np.array, np.array)
        mse for `k-best`, `f_avg` and `LSC` methods at different noise std
    """
    (sort, tros), (energy, comulated, bands), codebooks = training_result
    n_bands = conf.nbands
    # matrxi to vector (m2v) and vector to matrix (v2m) functions
    m2v, v2m = conf.vect_functions()
    subcfg = conf.testing['robust_reconstruction_visual']

    testing, Testing = common.load_dataset(conf.testingset_path(),
                                           conf.fformat, conf.size())
    FlatTst = m2v(Testing)[:, sort]

    # f_avg sampling pattern
    Omega = energy.argsort()[::-1]

    # lsc sampling pattern
    Omegas = [c.sampling_pattern() for c in codebooks]

    shape = testing[0].shape
    n = np.prod(shape)

    srange = np.logspace(*subcfg['noise_range'])
    sampling_rate = subcfg['sampling_rate']

    W, H = shape
    Wt = W * 3
    Ht = H * len(srange)

    res = np.zeros((Wt, Ht))

    print(f'Robust Reconstruction Quality Visual Test (img {idx}):')
    for i, sigma in enumerate(srange):
        print(f'\r {i+1:3d}/{len(srange)}', flush=True, end='')
        X = FlatTst[idx] + common.noise(sigma, n)

        M = int(round(sampling_rate*n))
        m = int(round(M/n_bands))
        ms = lsc.num_samples(bands, m)
        M = np.sum(ms)

        smalls = [omega[:y] for omega, y in zip(Omegas, ms)]

        Xsbs = lsc.split(X, bands)
        Ysbs = lsc.sub_sample(Xsbs, Omegas, m)

        recovered = [codebooks[b].reconstruct(Ysbs[b], smalls[b]) for b in range(len(bands))]
        Y = v2m((lsc.union(recovered))[tros], shape)
        y = common.norm(common.pos(common.ifft2(Y).real))


        BK = X.copy()[tros]
        O = np.abs(BK).argsort()[::-1]
        BK[O[M:]] = 0
        BK = v2m(BK, shape)
        bK = common.norm(common.pos(common.ifft2(BK).real))

        FA = X.copy()[tros]
        FA[Omega[M:]] = 0
        FA = v2m(FA, shape)
        fA = common.norm(common.pos(common.ifft2(FA).real))

        res[:W, H*i:H*(i+1)] = bK
        res[W:2*W, H*i:H*(i+1)] = fA
        res[2*W:3*W, H*i:H*(i+1)] = y
    print('\t[done]')
    return srange, res


def test_robust_sampling(conf, training_result):
    """
    Sampling rate vs Reconstruction Quality at fixed noise std.

    Parameters
    ----------
    conf : conf_loader.Conf
        Experiment parameters
    training_result : touple
        Retrun values of function `training`

    Returns
    -------
    srange : np.array
        sampling rate range used
    sampling_rate : float
        sampling rate used
    (bk_ser, fa_ser, rc_ser) : (np.array, np.array, np.array)
        SER for `k-best`, `f_avg` and `LSC` methods
    (bk_mse, fa_mse, rc_mse) : (np.array, np.array, np.array)
        mse for `k-best`, `f_avg` and `LSC` methods
    """
    (sort, tros), (energy, comulated, bands), codebooks = training_result
    n_bands = conf.nbands
    # matrxi to vector (m2v) and vector to matrix (v2m) functions
    m2v, v2m = conf.vect_functions()
    subcfg = conf.testing['robust_sampling']

    # Load testing set
    testing, Testing = common.load_dataset(conf.testingset_path(),
                                           conf.fformat, conf.size())
    FlatTst = m2v(Testing)[:, sort]

    # f_avg sampling pattern
    Omega = energy.argsort()[::-1]

    # lsc sampling pattern
    Omegas = [c.sampling_pattern() for c in codebooks]

    shape = testing[0].shape
    n = np.prod(shape)
    N = len(testing)

    srange = np.logspace(*subcfg['sampling_range'])
    sigma = subcfg['noise_rate']

    bk_ser = np.zeros(len(srange))
    fa_ser = np.zeros(len(srange))
    rc_ser = np.zeros(len(srange))

    bk_mse = np.zeros(len(srange))
    fa_mse = np.zeros(len(srange))
    rc_mse = np.zeros(len(srange))

    print('Sampling rate at fixed noise test:')
    for i, rate in enumerate(srange):
        print(f'\r {i+1:3d}/{len(srange)}', flush=True, end='')
        M = int(round(n*rate))
        m = int(round(M/n_bands))
        ms = lsc.num_samples(bands, m)
        M = np.sum(ms)
        smalls = [omega[:y] for omega, y in zip(Omegas, ms)]

        for idx in range(N):
            reference = common.norm(testing[idx])
            X = FlatTst[idx] + common.noise(sigma, n)

            Xsbs = lsc.split(X, bands)
            Ysbs = lsc.sub_sample(Xsbs, Omegas, m)
            recovered = [codebooks[b].reconstruct(Ysbs[b], smalls[b]) for b in range(len(bands))]
            Y = v2m((lsc.union(recovered))[tros], shape)
            y = common.norm(common.pos(common.ifft2(Y).real))


            BK = X.copy()[tros]
            O = np.abs(BK).argsort()[::-1]
            BK[O[M:]] = 0
            BK = v2m(BK, shape)
            bK = common.norm(common.pos(common.ifft2(BK).real))

            FA = X.copy()[tros]
            FA[Omega[M:]] = 0
            FA = v2m(FA, shape)
            fA = common.norm(common.pos(common.ifft2(FA).real))

            fa_ser[i] += common.SER(reference, fA) / N
            bk_ser[i] += common.SER(reference, bK) / N
            rc_ser[i] += common.SER(reference, y) / N

            fa_mse[i] += mse(reference, fA) / N
            bk_mse[i] += mse(reference, bK) / N
            rc_mse[i] += mse(reference, y) / N
    print(' [done]')
    return srange, sigma, (bk_ser, fa_ser, rc_ser), (bk_mse, fa_mse, rc_mse)
