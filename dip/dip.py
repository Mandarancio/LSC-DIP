# -*- coding: utf-8 -*-
from torchmath import av_dct2, nmse
from dip_utils.inpainting_utils import var_to_np
from dip_utils.inpainting_utils import get_noise, get_params

import torch

from dip_models.skip import skip

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

mse = torch.nn.MSELoss().type(dtype)

def costfunction(net, net_input, p_omega, observed, alpha, p_compl, prior):
    out = net(net_input)
    d = av_dct2(out)
    total_loss = 1 / (1 + alpha) * (alpha * mse(d * p_compl, prior)
                                    + mse(d * p_omega, observed))
    total_loss.backward()
    return total_loss


def costfunction_measure_conv(net, net_input, p_omega, observed, alpha, 
                              p_compl, prior, ground_truth, i, loss, recover):
    out = net(net_input)

    d = av_dct2(out)

    total_loss = 1 / (1 + alpha) * (alpha * mse(d * p_compl, prior)
                                    + mse(d * p_omega, observed))
    
    
    out_np = var_to_np(out)
    total_loss.backward()
    recover_loss = nmse(out_np, ground_truth)

    loss[i] += total_loss.data[0]
    recover[i] += recover_loss
    i += 1

    return total_loss


def init_net(shape):
    pad = 'reflection'
    INPUT = 'noise'
    input_depth = 32
    OPTIMIZER = 'adam'
    LR = 0.01

    OPT_OVER = 'net'

    def get_new_net():
        return skip(input_depth, shape[0],
                    num_channels_down=[16, 32, 64, 128, 128],
                    num_channels_up=[16, 32, 64, 128, 128],
                    num_channels_skip=[0, 0, 0, 0, 0],
                    filter_size_down=3,
                    filter_size_up=3,
                    filter_skip_size=1,
                    upsample_mode='nearest',
                    act_fun='LeakyReLU',
                    need_sigmoid=True,
                    need_bias=True,
                    pad=pad).type(dtype)


    net_input = get_noise(input_depth,
                          INPUT,
                          shape[1:]).type(dtype).detach()

    net = get_new_net()
    p = get_params(OPT_OVER, net, net_input)

    return net, net_input, p, OPTIMIZER, LR


