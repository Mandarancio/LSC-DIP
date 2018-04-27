#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi band reconstruction
=========================

The processing steps are the following: 
    1. Load training data-set (load_dataset())
    2. Extract statistic data on the training set 
    3. Chose a sorting strategy
    4. Sub-bands division
    5. Codebook generation
 
@author: Martino Ferrari
"""
#%%
import numpy as np
import pandas as pd

from skimage.measure import compare_mse as mse
from skimage.measure import compare_ssim as ssim

import skimage.io as iio

import common

import pyfftw 

import matplotlib.pyplot as plt

# import seaborn as sns

# Some lambadas and namespaces
fft = pyfftw.interfaces.scipy_fftpack
fft2 = lambda x : fft.fftshift(fft.fft2(x))
ifft2 = lambda x : fft.ifft2(fft.ifftshift(x))
dct2 = lambda x : fft.dct(fft.dct(x.T, norm='ortho').T, norm='ortho')
idct2 = lambda x : fft.idct(fft.idct(x.T, norm='ortho').T, norm='ortho')
comulate = lambda x : np.array([np.sum(x[:i]) for i in range(len(x))])


#%%
    
if __name__ == "__main__":
    #%%
    """
    Multi band reconstruction
    -------------------------
    
    The processing steps are the following:
        1. Load training data-set (load_dataset_)
        2. Extract statistic data on the training set (retrive_basic_stats_)
        3. Chose a sorting strategy 
        4. Sub-bands division (divide_)
        5. Codebook generation (gen_codebooks_)
    """
    setname = 'INRIA'
    n_levels = 100
   
    size = None
    if setname in ['SDO', 'SDO-Hard', 'INRIA']:
        size = (256, 256)
    fformat = '.jpg' if setname in ['CELEBa', 'INRIA'] else '.png'
    print(fformat)
    training, Training = common.load_dataset(f'../Clean/sets/{setname}-training/', 
                                             fformat=fformat, size=size)
    
    #%%
    """
    Extract stastics.
    """
    
    shape = training[0].shape
    N = np.prod(shape)
    L = len(Training)
    n_bands = 7
            
    # For plotting pourpouse
    freq_x = fft.fftshift(fft.fftfreq(shape[1]))
    freq_y = fft.fftshift(fft.fftfreq(shape[0]))
    freq_x, freq_y = np.meshgrid(freq_x, freq_y)
    
    mag, mag_std, phs, phs_std = common.retrive_basic_stats(Training)
    Omega = mag.reshape(-1).argsort()[::-1]
    
    """
    3D plot average magnetude and average phase.
    """
    
    Z = phs
    common.surf3D(freq_x, freq_y, Z)
    plt.xlabel('horizontal frequency')
    plt.ylabel('vertical frequency')
    plt.title('Average phase')
    
    Z = np.log10(mag)
    common.surf3D(freq_x, freq_y, Z)
    plt.xlabel('horizontal frequency')
    plt.ylabel('vertical frequency')
    plt.title('Average magnetude')
    
    # no wierd sorting
    sort = np.arange(N)
    tros = common.inv(sort)
    
    energy = common.norm(mag_std.reshape(-1)[sort])
    comulated = common.norm(comulate(energy))
    bands = common.divide(comulated, n_bands)
    #%%
    """
    Plot the normalized average energy, comulative energy and the sub-bands.
    
    The energy in this case is just the magnetude STD.
    """    
    
    smooth_energy = []
    smooth_freq = []
    smooth_comulated = []
    for i in range(0, N, 100):
        smooth_freq.append(i)
        smooth_energy.append(np.max(energy[i:i+100]))
        smooth_comulated.append(np.max(comulated[i:i+100]))
        
    plt.figure()
    plt.plot(smooth_freq, smooth_energy, label='energy')
    plt.plot(smooth_freq, comulated[::100], label='comulative energy')
    plt.axvline(x=0, ls='--', color='k', lw=1.0, alpha=0.5)
    for band in bands:
        plt.axvline(x=band, ls='--', color='k', lw=1.0, alpha=0.5)    
    
    plt.xlabel('index')
    plt.ylabel('energy')
    plt.grid()
    plt.legend()
    plt.title('normalized flat average energy')
    
    df = pd.DataFrame({
            'energy' : energy,
            'comulated' : comulated
            })
    df.to_csv(f'energy_{setname}_{n_bands}_{n_levels}.csv', index_label='feature')
    np.savetxt(f'bands_{setname}_{n_bands}_{n_levels}.csv', bands, header='bands', delimiter=',')
    #%%
    """
    Flatten the frequency domain dataset and normalize it and discretize it.
    Then divide the result in the computed sub-bands.
    """
    
    
    normalize = mag.reshape(-1)[sort]
    discretize = normalize/n_levels
    
    FlatTrn = Training.reshape((L, -1))[:, sort]
    DiscTrn = np.round(FlatTrn/discretize) * discretize
    SBsTrn = common.split(DiscTrn, bands)
    
    """
    Compute codebook for each sub-band.
    """
    
    n_codes = 500 if setname == 'CELEBa' else 250
    n_codes = 100 if setname == 'DENIS' else n_codes
    batch_size = 100 if setname == 'CELEBa' else 100
    codebooks = common.gen_codebooks(SBsTrn, n_codes, mode='ReIm', 
                                     batch_size=batch_size)
    
        
    #%%
    testing, Testing = common.load_dataset(f'../Clean/sets/{setname}-testing/', 
                                           fformat, size)
    FlatTst = Testing.reshape((len(Testing), -1))[:, sort]
    
    #%%
    """
    Testing compression on training data.
    """
    idx = 7
    sig = FlatTst[idx]
    base = testing[idx]
    sig_sbs = common.split(sig, bands)
    
    plt.figure()
    plt.title('orginal')
    plt.imshow(base)

    """
    Testing reconstruction on training data.
    
    Computing sampling pattern per band.
    """
    omegas = [c.sampling_pattern() for c in codebooks]

    
    M = int(round(N*0.01))
    m = int(round(M/n_bands))
    ms = common.num_samples(bands, m)
    M = np.sum(ms)

    Ysbs = common.sub_sample(sig_sbs, omegas, m)
    subs = (common.union(Ysbs))[tros].reshape(shape) 
    subs = common.norm(np.abs(ifft2(subs)))
     
    smalls = [omega[:y] for omega, y in zip(omegas, ms)]
    recovered = [codebooks[i].reconstruct(Ysbs[i], smalls[i]) for i in range(len(bands))]
    Y = (common.union(recovered))[tros].reshape(shape)
    y = common.norm(np.abs(ifft2(Y)))
    
    BK = FlatTst[idx].copy()[tros]
    O = np.abs(BK).argsort()[::-1]
    BK[O[M:]] = 0
    BK = BK.reshape(shape)
    bK = common.norm(np.abs(ifft2(BK)))    

    FA = FlatTst[idx].copy()[tros]
    FA[Omega[M:]] = 0
    FA = FA.reshape(shape)
    fA = common.norm(np.abs(ifft2(FA)))

    plt.figure()
    
    plt.subplot(2, 2, 1)
    plt.title(f'Input\nmse: {mse(base, subs):0.3e}\nssim: {ssim(subs, base, gaussian_weights=True):0.3e}')
    plt.axis('off')
    plt.imshow(subs)
    
    plt.subplot(2, 2, 2)    
    plt.title(f'Recovered\nmse: {mse(y, base):0.3e}\nssim: {ssim(y, base, gaussian_weights=True):0.3e}')
    plt.axis('off')
    plt.imshow(y)
    
    plt.subplot(2, 2, 3)
    plt.title(f'F_Avg\nmse: {mse(fA, base):0.3e}\nssim: {ssim(fA, base, gaussian_weights=True):0.3e}')
    plt.imshow(fA)
    plt.axis('off')
    
    plt.subplot(2, 2, 4)    
    plt.title(f'Best-K\nmse: {mse(bK, base):0.3e}\nssim: {ssim(bK, base, gaussian_weights=True):0.3e}')
    plt.axis('off')
    plt.imshow(bK)
    
    plt.subplots_adjust(0, 0, 1, 0.85, 0, 0.4)

    Ymag = np.abs(Y)
    Yphs = np.angle(Y)
    
    X = fft2(base)
    Xmag = np.abs(X)
    Xphs = np.angle(X)
    
    plt.figure()
    plt.imshow(np.abs(Xphs-Yphs))
    plt.colorbar()
    
    
    plt.figure()
    plt.imshow(np.log10(np.abs(Xmag-Ymag)+1))
    plt.colorbar()
    plt.show()
    
    
    
    #%%
    """
    Test the performance at different sampling rates.
    """
    
    
    L_tst = len(testing)
    
    srange = np.logspace(-2, 0, 20)[:-1]

    bk_mse = np.zeros(len(srange))
    fa_mse = np.zeros(len(srange))
    rc_mse = np.zeros(len(srange))
    
    bk_ssim = np.zeros(len(srange))
    fa_ssim = np.zeros(len(srange))
    rc_ssim = np.zeros(len(srange))
    
    # idx = 12
    factor = L_tst
    print('starting testing')
    for i, rate in enumerate(srange):
        M = int(round(N*rate))
        m = int(round(M/n_bands))
        ms = common.num_samples(bands, m)
        M = np.sum(ms)
        
        smalls = [omega[:y] for omega, y in zip(omegas, ms)]

        for idx in range(L_tst):
            reference = common.norm(testing[idx])
            X = FlatTst[idx]
                        
            Xsbs = common.split(X, bands)
            Ysbs = common.sub_sample(Xsbs, omegas, m)
            recovered = [codebooks[b].reconstruct(Ysbs[b], smalls[b]) for b in range(len(bands))]
            Y = (common.union(recovered))[tros].reshape(shape)
            y = common.norm(common.pos(ifft2(Y).real))
            
             
            BK = X.copy()[tros]
            O = np.abs(BK).argsort()[::-1]
            BK[O[M:]] = 0
            BK = BK.reshape(shape)
            bK = common.norm(common.pos(ifft2(BK).real))    
        
            FA = X.copy()[tros]
            FA[Omega[M:]] = 0
            FA = FA.reshape(shape)
            fA = common.norm(common.pos(ifft2(FA).real))
            
            fa_mse[i] += mse(reference, fA)
            bk_mse[i] += mse(reference, bK)
            rc_mse[i] += mse(reference, y) 
            
            fa_ssim[i] += ssim(reference, fA, gaussian_weights=True)
            bk_ssim[i] += ssim(reference, bK, gaussian_weights=True) 
            rc_ssim[i] += ssim(reference, y, gaussian_weights=True) 
    #%%
    srange = np.logspace(-2, 0, 20)[:-1]

    res = {'fa_mse': fa_mse,
           'bk_mse': bk_mse,
           'rc_mse': rc_mse,
           'fa_ssim': fa_ssim,
           'bk_ssim': bk_ssim,
           'rc_ssim': rc_ssim,
           'sampling_rate': srange
           }
    df = pd.DataFrame(res)
    df.to_csv(f'res_{setname}_{n_codes}_{n_bands}_{n_levels}_mbr.csv', index=False)
    
    
    plt.figure()
    plt.semilogx(srange, fa_mse/factor, 's-', label='f_avg')
    plt.semilogx(srange, bk_mse/factor, 's-', label='Best-K')
    plt.semilogx(srange, rc_mse/factor, 's-', label='MB_rec')
    
    plt.xlabel('Sampling rate')
    plt.ylabel('mse')
    plt.grid(which='both')
    plt.legend()
    plt.savefig(f'overall_{setname}_{n_codes}_{n_bands}_{n_levels}_mse.eps')
    
    plt.figure()
    plt.semilogx(srange, fa_ssim/factor, 's-', label='f_avg')
    plt.semilogx(srange, bk_ssim/factor, 's-', label='Best-K')
    plt.semilogx(srange, rc_ssim/factor, 's-', label='MB_rec')
    
    plt.xlabel('Sampling rate')
    plt.ylabel('ssim')
    plt.grid(which='both')
    plt.legend()
    plt.savefig(f'overall_{setname}_{n_codes}_{n_bands}_{n_levels}_ssim.eps')    
    
    #%%
    idx = 42
    
    X = FlatTst[idx].copy()
    x = testing[idx].copy()
    
    srange = np.logspace(-3, -1, 20)
    W, H = shape
    Wt = W * 4
    Ht = H * len(srange)
    
    res = np.zeros((Wt, Ht))
    
    for i, rate in enumerate(srange):
        M = int(round(rate*N))       
        m = int(round(M/n_bands))
        ms = common.num_samples(bands, m)
        M = np.sum(ms)
        
        smalls = [omega[:y] for omega, y in zip(omegas, ms)]
        
        Xsbs = common.split(X, bands)
        Ysbs = common.sub_sample(Xsbs, omegas, m)
        subs = (common.union(Ysbs))[tros].reshape(shape) 
        subs = common.norm(np.abs(ifft2(subs)))
    
        recovered = [codebooks[b].reconstruct(Ysbs[b], smalls[b]) for b in range(len(bands))]
        Y = (common.union(recovered))[tros].reshape(shape)
        y = common.norm(common.pos(ifft2(Y).real))
        
        
        BK = X.copy()[tros]
        O = np.abs(BK).argsort()[::-1]
        BK[O[M:]] = 0
        BK = BK.reshape(shape)
        bK = common.norm(common.pos(ifft2(BK).real))    
    
        FA = X.copy()[tros]
        FA[Omega[M:]] = 0
        FA = FA.reshape(shape)
        fA = common.norm(common.pos(ifft2(FA).real))
        
        res[:W, H*i:H*(i+1)] = bK
        res[W:2*W, H*i:H*(i+1)] = fA
        res[2*W:3*W, H*i:H*(i+1)] = subs
        res[3*W:4*W, H*i:H*(i+1)] = y
    
    plt.figure()

    iio.imsave(f'reconstruction_{setname}_{n_codes}_{n_bands}_{n_levels}_{idx}.png', res)
    plt.imshow(res)
    plt.show()
    
     
