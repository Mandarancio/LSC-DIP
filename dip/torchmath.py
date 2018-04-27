# -*- coding: utf-8 -*-
import numpy as np
import pyfftw

import torch
from torch.autograd import Function

fft = pyfftw.interfaces.scipy_fftpack

dct2 = lambda x: fft.dct(fft.dct(x.T, norm='ortho').T, norm='ortho')
idct2 = lambda x: fft.idct(fft.idct(x.T, norm='ortho').T, norm='ortho')
nmse = lambda x, y: np.mean((x-y)**2)


def dct3(x):
    if len(x.shape) == 2:
        return dct2(x)
    else:
        return dct3(x[0]).reshape(x.shape)


def idct3(x):
    if len(x.shape) == 2:
        return idct2(x)
    else:
        return idct3(x[0]).reshape(x.shape)


class IDCTFunction(Function):

    def forward(self, input):
        numpy_input = input.cpu().numpy()
        result = idct3(numpy_input)
        return torch.FloatTensor(result).cuda()

    def backward(self, grad_output):
        numpy_go = grad_output.cpu().numpy()
        result = dct3(numpy_go)
        return torch.FloatTensor(result).cuda()


class DCTFunction(Function):

    def forward(self, input):
        numpy_input = input.cpu().numpy()
        result = dct3(numpy_input)
        return torch.FloatTensor(result).cuda()

    def backward(self, grad_output):
        numpy_go = grad_output.cpu().numpy()
        result = idct3(numpy_go)
        return torch.FloatTensor(result).cuda()


def av_dct2(input):
    return DCTFunction()(input)


def av_idct2(input):
    return DCTFunction()(input)

