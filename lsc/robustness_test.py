#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 12:03:57 2018

@author: Martino Ferrari
@email: manda.mgf@gmail.com
"""

#%%
import numpy as np

import common

import pyfftw

import matplotlib.pyplot as plt

import sys

# import seaborn as sns

# Some lambadas and namespaces
fft = pyfftw.interfaces.scipy_fftpack
fft2 = lambda x: fft.fftshift(fft.fft2(x))
ifft2 = lambda x: fft.ifft2(fft.ifftshift(x))
dct2 = lambda x: fft.dct(fft.dct(x.T, norm='ortho').T, norm='ortho')
idct2 = lambda x: fft.idct(fft.idct(x.T, norm='ortho').T, norm='ortho')
comulate = lambda x: np.array([np.sum(x[:i]) for i in range(len(x))])


def noise(sigma, shape):
    return np.random.normal(0, sigma, shape) * np.exp(1j * np.random.rand(shape) * 2 * np.pi)


def SER(x, xrec):
    return 10 * np.log10( np.sum(x**2) / np.sum((x - xrec) ** 2))

#%%


setname = 'SDO'
if len(sys.argv) > 1:
    setname = sys.argv[1]
n_levels = 100

size = None
if setname in ['SDO', 'SDO-Hard']:
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
n_bands = 5


mag, mag_std, phs, phs_std = common.retrive_basic_stats(Training)
Omega = mag.reshape(-1).argsort()[::-1]



# no wierd sorting
sort = np.arange(N)
tros = common.inv(sort)

energy = common.norm(mag_std.reshape(-1)[sort])
comulated = common.norm(comulate(energy))
bands = common.divide(comulated, n_bands)

#%%


plt.figure()
plt.plot(energy, label='energy')
plt.plot(comulated, label='comulative energy')
plt.axvline(x=0, ls='--', color='k', lw=1.0, alpha=0.5)
for band in bands:
    plt.axvline(x=band, ls='--', color='k', lw=1.0, alpha=0.5)

plt.xlabel('index')
plt.ylabel('energy')
plt.grid()
plt.legend()
plt.title('normalized flat average energy')

#%%


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

sigmas = np.logspace(-2, 2, 20)

errors = np.zeros((n_bands, sigmas.size))
for X, x in zip(FlatTst, testing):
    Xsbs = common.split(X, bands)
    labels = [codebooks[b].compress(Xsbs[b], None) for b in range(n_bands)]
    for i, sigma in enumerate(sigmas):
        Z = noise(sigma, N)
        I = X + Z
        Isbs = common.split(I, bands)
        ls =  [codebooks[b].compress(Isbs[b], None) for b in range(n_bands)]
        for j, (refv, comv) in enumerate(zip(labels, ls)):
            errors[j, i] += np.mean(np.abs(refv - comv)**2)

#%%

norm = np.zeros(n_bands)
oi = 0
en = mag.reshape(-1)
for i, b in enumerate(bands):
    norm[i] = np.mean(en[oi:b]**2)
    oi = b

np.savetxt(f'{setname}_sigma_vs_errors.csv', errors)



plt.figure()
for i in range(n_bands):
    plt.semilogx(sigmas, errors[i]/(norm[i]*len(testing)), 'x-', label=f'band {i+1}')

plt.xlabel('$\sigma_z$')
plt.ylabel('Error Rate')
plt.grid(which='both', alpha=0.5)
plt.xlim([sigmas[0], sigmas[-1]])
plt.legend()
plt.title(setname)
plt.savefig(f'noise_vs_error_rate_{setname}.eps')


#%%

srate = np.logspace(-2, 0, 20)
omegas = [c.sampling_pattern() for c in codebooks]

errors = np.zeros((n_bands, sigmas.size))
for X, x in zip(FlatTst, testing):
    Xsbs = common.split(X, bands)
    labels = [codebooks[b].predict(Xsbs[b], None) for b in range(n_bands)]
    for i, sr in enumerate(srate):
        M = int(round(N*sr))
        m = int(round(M/n_bands))
        ms = common.num_samples(bands, m)
        M = np.sum(ms)
        smalls = [omega[:y] for omega, y in zip(omegas, ms)]

        ls =  [codebooks[b].predict(Xsbs[b], smalls[b]) for b in range(n_bands)]
        for j, (refv, comv) in enumerate(zip(labels, ls)):
            errors[j, i] += refv != comv
#%%
plt.figure()

for i in range(n_bands):
    plt.semilogx(srate, errors[i]/len(testing), 'x-', label=f'band {i+1}')

np.savetxt(f'{setname}_errorrate.csv', errors/len(testing))

plt.title(setname)
plt.xlabel('Sampling rate')
plt.ylabel('Error Rate')
plt.grid(which='both', alpha=0.5)
plt.xlim([srate[0], srate[-1]])
plt.legend()
plt.savefig(f'srate_vs_error_rate_{setname}.eps')
