from __future__ import division, print_function

import numpy as np
from scipy.stats import moment
import matplotlib.pyplot as plt

import liteMap
import fftTools
from flipper import *
STRING_MAP_FILE = '/Users/verag/Research/strings/strings/data/dTmap_stg_Dcmb0000.fits'
string_template =  liteMap.liteMapFromFits(STRING_MAP_FILE)
string_template.Nx, string_template.Ny = np.shape(string_template.data)
string_fovX = 7.2 #deg
string_fovY = 7.2 #deg
string_scaleX = np.deg2rad(string_fovX) / string_template.Nx
string_scaleY = np.deg2rad(string_fovY) / string_template.Ny


class Statistic(object):
    def __init__(self, function, name):
        """function takes map data in, spits out number or array
        """
        self.function = function
        self.name = name

    def __call__(self, data, **kwargs):
        return self.function(data, **kwargs)

class Moments(Statistic):
    def __init__(self, moments=[2,3,4], name='moments'):
        self.moments = moments
        self.name = name

    def __call__(self, data):
        result = []
        for m in self.moments:
            result.append(moment(data.ravel(), m))
        return np.array(result)
    
        #return np.array([moment(data.ravel(),m) for m in self.moments])
    
class Mean(Statistic):
    def __init__(self):
        self.function = np.mean
        self.name = 'mean'


class PowerSpectrum(Statistic):
    def __init__(self, name='power'):
        self.name = name

    def __call__(self, data, pixScaleX=string_scaleX, pixScaleY=string_scaleY,
                 bin_file='/Users/verag/flipper-master/params/BIN_100_LOG'):
        """
           pixScaleX, pixScaleY are in rad/pixel.

        """

        m = liteMap.liteMap()
        m.data = data
        m.Nx, m.Ny = data.shape
        m.pixScaleX = pixScaleX
        m.pixScaleY = pixScaleY
        map_fov_degX = np.rad2deg(pixScaleX) * m.Nx
        map_fov_degY = np.rad2deg(pixScaleY) * m.Ny
        m.area = map_fov_degX * map_fov_degY
        m.x0 = 0
        m.y0 = 0
        m.x1 = map_fov_degX
        m.y1 = map_fov_degY
        
        p2d = fftTools.powerFromLiteMap(m)
        ll, ll, lbin, cl, ll, ll = p2d.binInAnnuli(bin_file)        

        #note: cl is now res[1,:], lbin is res[0,:]
        res = np.array([lbin,cl])
        return res 
    
class PDF(Statistic):
    def __init__(self, name='pdf'):
        self.name = name

    def __call__(self, data,
                 bins=110, normed=True,
                 binmin=1e5,binmax=1e10,):

        bin_edges = np.linspace(binmin, binmax, bins+1)
        bin_mids = (bin_edges[1:] + bin_edges[:-1])/2
        h, b = np.histogram(data.ravel(),bin_edges,normed=normed)

        #(h,y,rest) = plt.hist(data.ravel(),bins=bins,
        #                      histtype=histtype,normed=normed,
        #                      color='k', alpha=0.2, lw=0.5);

        #x = [(y[i+1]-y[i])/2.+y[i] for i in np.arange(len(y)-1) ]

        
        
        res = np.array([bin_mids,h])
        return res
