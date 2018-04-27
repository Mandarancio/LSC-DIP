# -*- coding: utf-8 -*-
import sys
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as iio

import common
import conf_loader
import flow

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        print('To use LSC please:\n\tlsc cofig.yml')
        sys.exit(-1)
    conf = conf_loader.load_conf(args[-1])

    results_dir = 'lsc_results/'
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    t = datetime.today()
    output_dir = results_dir + conf.dataset + '_' + t.strftime('%Y-%m-%d.%f')+'/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    conf.save_cfg(output_dir)

    training = flow.training(conf)
    _, (energy, comulated, bands), _ = training

    # display average magnitude, comulative magnitude and bands.
    plt.plot(energy)
    plt.plot(comulated)
    plt.axvline(0, ls='--', lw=1, c='k')
    for b in bands:
        plt.axvline(b, ls='--', lw=1, c='k')
    plt.show()

    names = ['$k$-best', '$f_{avg}$', 'LSC']
    testing_size = common.count_images(conf.testingset_path())

    if 'reconstruction' in conf.testing:
        srange, mses, ssims = flow.test_sampling(conf, training)

        csv = np.zeros((7, srange.size))
        title = 's.r.'
        csv[0] = srange
        for i, mse in enumerate(mses):
            csv[1+i] = mse
            title += ',mse_'+names[i]
        for i, ssim in enumerate(ssims):
            csv[4+i] = ssim
            title += ',ssim_'+names[i]
        np.savetxt(output_dir+'reconstruction.csv',
                   csv, header=title, delimiter=',', fmt='%.4e')

        plt.subplot(1, 2, 1)
        for name, mse in zip(names, mses):
            plt.semilogx(srange, mse, label=name)
        plt.xlabel('Sampling Rate (log scale)')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(alpha=0.5)

        plt.subplot(1, 2, 2)
        for name, ssim in zip(names, ssims):
            plt.semilogx(srange, ssim, label=name)
        plt.xlabel('Sampling Rate (log scale)')
        plt.ylabel('SSIM')
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()

    if 'reconstruction_visual' in conf.testing:
        if conf.testing['reconstruction_visual']['samples'] == 'random':
            idxs = [np.random.choice(testing_size)]
        else:
            idxs = conf.testing['reconstruction_visual']['samples']

        for idx in idxs:
            _, res = flow.test_sampling_visual(conf, training, idx)
            iio.imsave(f'{output_dir}reconstruction_img_{idx}.png', res)
            plt.imshow(res)
            plt.show()

    if 'robust_reconstruction' in conf.testing:
        srange, rate, mses, sers = flow.test_robust(conf, training)

        csv = np.zeros((8, srange.size))
        title = 'std,s.r.'
        csv[0] = srange
        csv[1] = [rate] * srange.size
        for i, mse in enumerate(mses):
            csv[2+i] = mse
            title += ',mse_'+names[i]
        for i, ssim in enumerate(sers):
            csv[5+i] = ssim
            title += ',ser_'+names[i]
        np.savetxt(output_dir+'robust_reconstruction.csv',
                   csv, header=title, delimiter=',', fmt='%.4e')

        plt.subplot(1, 2, 1)
        for name, mse in zip(names, mses):
            plt.semilogx(srange, mse, label=name)
        plt.xlabel('Noise STD (log scale)')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(alpha=0.5)

        plt.subplot(1, 2, 2)
        for name, ssim in zip(names, sers):
            plt.semilogx(srange, ssim, label=name)
        plt.xlabel('Noise STD (log scale)')
        plt.ylabel('SER')
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()

    if 'robust_reconstruction_visual' in conf.testing:
        if conf.testing['reconstruction_visual']['samples'] == 'random':
            idxs = [np.random.choice(testing_size)]
        else:
            idxs = conf.testing['reconstruction_visual']['samples']

        for idx in idxs:
            _, res = flow.test_robust_visual(conf, training, idx)
            iio.imsave(f'{output_dir}robust_reconstruction_img_{idx}.png', res)

            plt.imshow(res)
            plt.show()

    if 'robust_sampling' in conf.testing:
        srange, std, mses, sers = flow.test_robust_sampling(conf, training)

        csv = np.zeros((8, srange.size))
        title = 's.r.,std'
        csv[0] = srange
        csv[1] = [std] * srange.size
        for i, mse in enumerate(mses):
            csv[2+i] = mse
            title += ',mse_'+names[i]
        for i, ssim in enumerate(sers):
            csv[5+i] = ssim
            title += ',ser_'+names[i]
        np.savetxt(output_dir+'robust_sampling.csv',
                   csv, header=title, delimiter=',', fmt='%.4e')

        plt.subplot(1, 2, 1)
        for name, mse in zip(names, mses):
            plt.semilogx(srange, mse, label=name)
        plt.xlabel('Sampling Rate (log scale)')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(alpha=0.5)

        plt.subplot(1, 2, 2)
        for name, ssim in zip(names, sers):
            plt.semilogx(srange, ssim, label=name)
        plt.xlabel('Sampling Rate (log scale)')
        plt.ylabel('SER')
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()
