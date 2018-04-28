# -*- coding: utf-8 -*-
import sys
import os
from datetime import datetime

import yaml

import numpy as np

from dip_utils.inpainting_utils import optimize
from dip_utils.inpainting_utils import var_to_np, np_to_var

from skimage.transform import resize
from skimage.measure import compare_mse as mse
from skimage.measure import compare_ssim as ssim

import matplotlib.pyplot as plt

import dip
import torchmath as tm
import image_utility as iu


def costfunction():
    """NN Cost function: to minimise."""
    global net
    global net_input

    global P_omg_var
    global b_var

    global alpha
    global P_cmp_var
    global c_var

    return dip.costfunction(net, net_input, P_omg_var,
                            b_var, alpha, P_cmp_var, c_var)


if __name__ == '__main__':
    """ Load configuration """
    if not len(sys.argv) == 2:
        print("Usage: conv.py conf.yml")
        sys.exit(-1)
    cfg_path = sys.argv[1]
    conf = None

    with open(cfg_path, 'r') as f:
        conf = yaml.load(f)
    if conf is None:
        raise ValueError('Something wrong with configuration file!')
    datafolder = conf['datafolder']
    dataset = conf['dataset']
    fformat = conf['fformat']
    shape = conf['shape']
    tshape = (1, *shape) # torch shape
    num_iter = conf['max_iters']
    alpha = conf['alpha']
    srange = np.logspace(*conf['sampling_range'])
    # size of signal
    n = np.prod(shape)
    timestamp = datetime.today()

    """ Load dataset."""
    print('load training dataset...', end='', flush=True)
    training = iu.load_images(f'{datafolder}/{dataset}-training',
                              fformat=fformat)
    training = [resize(x, shape, mode='reflect') for x in training]
    print(' [done]')

    """ Load testing set."""
    print('print testing dataset...', end='', flush=True)
    testing = iu.load_images(f'{datafolder}/{dataset}-testing',
                              fformat=fformat)
    testing = [resize(x, shape, mode='reflect') for x in testing]

    # select images to test
    if conf['samples'] == 'random':
        idx_s = np.random.choice(len(testing), size=conf['nsamples'],
                                 replace=False)
    else:
        idx_s = conf['samples']
    print(' [done]')

    """ Find sampling pattern."""
    print('extracting information...')
    Training = [tm.dct2(x).reshape(-1) for x in training]
    Omega = np.var(Training, 0).argsort()[::-1]
    prior = np.mean(Training, 0).reshape(tshape)
    Prior = tm.dct3(prior)
    print(' [done]')



    bk_mse = np.zeros(srange.size)
    bk_ssim = np.zeros(srange.size)

    fa_mse = np.zeros(srange.size)
    fa_ssim = np.zeros(srange.size)

    dp_mse = np.zeros(srange.size)
    dp_ssim = np.zeros(srange.size)

    pdp_mse = np.zeros(srange.size)
    pdp_ssim = np.zeros(srange.size)


    h, w = shape
    res = np.zeros([h * 4 * len(idx_s), w * srange.size])

    """ Testing code. """
    for i, rate in enumerate(srange):
        m = int(round(rate*n))

        # compute P_omega and P_omega^C
        P_omega = np.zeros(n)
        P_omega[Omega[:m]] = 1
        P_omega = (P_omega.reshape(tshape))
        P_omg_var = np_to_var(P_omega).type(dip.dtype)
        P_compl = 1 - P_omega
        P_cmp_var = np_to_var(P_compl).type(dip.dtype)
        # Note: not in the math sense but in practice is the same
        # In this way simple multiplication is enough instead of a matrix mult

        for j, idx in enumerate(idx_s):
            print(f'\r>> {i+1:3d}/{len(srange)}\t[{j+1:3d}/{len(idx_s)}]\t',
                  end='', flush=True)
            img_np = testing[idx].reshape(tshape)

            # Observed Signal
            b = tm.dct3(img_np) * P_omega
            # Torch variable
            b_var = np_to_var(b).type(dip.dtype)

            # Prior Signal
            c = Prior * P_compl
            # torch variable
            c_var = np_to_var(c).type(dip.dtype)

            """ Compute $k$-best and $f_{avg}$."""
            X = tm.dct3(img_np).reshape(-1)
            O = np.abs(X).argsort()[::-1][m:]
            X[O] = 0
            x = iu.norm(tm.idct2(X.reshape(shape)))

            res[(4*j)*h:(4*j+1)*h, i*w:(i+1)*w] = x
            bk_mse[i] += mse(x, testing[idx])/len(idx_s)
            bk_ssim[i] += ssim(x, testing[idx])/len(idx_s)

            fa =  iu.norm(tm.idct3(b).reshape(shape))

            res[(4*j+1)*h:(4*j+2)*h, i*w:(i+1)*w] = fa
            fa_mse[i] += mse(fa, testing[idx])/len(idx_s)
            fa_ssim[i] += ssim(fa, testing[idx])/len(idx_s)


            """ Init net """
            net, net_input, params, OPTIMIZER, LR = dip.init_net(tshape)
            """ Optimize net """
            optimize(OPTIMIZER, params, costfunction, LR=LR, num_iter=num_iter)

            """ Get output """
            out = net(net_input)
            out_np = iu.norm(var_to_np(out).reshape(shape))

            res[(4*j+2)*h:(4*j+3)*h, i*w:(i+1)*w] = out_np
            dp_mse[i] += mse(out_np, testing[idx])/len(idx_s)
            dp_ssim[i] += ssim(out_np, testing[idx])/len(idx_s)

            """ Constraint to observed value """

            pout = iu.norm(tm.idct2(tm.dct2(out_np)*P_compl.reshape(shape)+
                            b.reshape(shape)))

            res[(4*j+3)*h:(4*j+4)*h, i*w:(i+1)*w] = pout
            pdp_mse[i] += mse(pout, testing[idx])/len(idx_s)
            pdp_ssim[i] += ssim(pout, testing[idx])/len(idx_s)
    print(' [done]')

    """ Crete output directory and store configuration file."""
    outdir = 'dip_results/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    resdir = outdir + dataset + '_' + timestamp.strftime('%Y-%m-%d.%f')+'/'
    if not os.path.isdir(resdir):
        os.mkdir(resdir)

    with open(resdir+'config.yml', 'w') as f:
        yaml.dump(conf, f)

    np.savetxt(resdir + 'results.csv',np.array([srange, bk_mse, fa_mse,
                                                dp_mse, pdp_mse, bk_ssim,
                                                fa_ssim, dp_ssim, pdp_ssim]).T,
                    delimiter=',', header='s.r.,mse_bk,mse_fa,mse_dp,mse_dpd,\
                    ssim_bk,ssim_fa,ssim_pd,ssim_pdp')
    iu.save_image(res, resdir + 'results.png')

    plt.semilogx(srange, bk_mse, label='$k$-best')
    plt.semilogx(srange, fa_mse, label='$f_{avg}$')
    plt.semilogx(srange, dp_mse, label='DIP - no b')
    plt.semilogx(srange, pdp_mse, label='DIP - b')
    plt.xlabel('Sampling Range (log)')
    plt.ylabel('MSE')
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()
