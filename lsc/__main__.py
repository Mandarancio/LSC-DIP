# -*- coding: utf-8 -*-
import sys


import matplotlib.pyplot as plt
import conf_loader
import flow

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        print('To use LSC please:\n\tlsc cofig.yml')
        sys.exit(-1)
    conf = conf_loader.load_conf(args[-1])
    
    training = flow.training(conf)
    _, (energy, comulated, bands), _ = training
    plt.plot(energy)
    plt.plot(comulated)
    plt.show()