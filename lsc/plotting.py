#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting utilities for LSC and DIP project

@author: Martino Ferrari
@email: manda.mgf@gmail.com
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def surf3D(x,y,z):
    """
    Plot surface 3D + contourf.

    Parameters
    ----------
    x : np.ndarray
      A 2D matrix
    y : np.ndarray
      A 2D matrix
    z : np.ndarray
      A 2D matrix

    Returns
    -------
    None
    """
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot_surface(x, y, z, cmap='coolwarm')
    off_x = plt.xlim()[0]
    off_y = plt.ylim()[1]
    ax.contourf(x, y, z,zdir='x', offset=off_x, cmap='coolwarm')
    ax.contourf(x, y, z, zdir='y', offset=off_y, cmap='coolwarm')
