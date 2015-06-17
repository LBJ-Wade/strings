from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
import os, os.path, glob, shutil
import sys,re
import cPickle as pickle

import logging

from .simulator import obtain_N_cmb_maps
from .statistic import Statistic
from .convolve import generate_rotated_stick_convolutions

MAPSROOT = '/data/verag/strings/maps'

def generate_gradmaps(parent_folder='.', scaleX=None, scaleY=None, **kwargs):
    logging.debug('Generating gradmaps, parent folder={}'.format(parent_folder))
    mapfiles = glob.glob('{}/map*.npy'.format(parent_folder))
    
    maps = []
    for mf in mapfiles:
        m = np.load(mf)
        gradx,grady = np.gradient(m, scaleX, scaleY)
        maps.append(np.sqrt(gradx**2 + grady**2))
    return maps

class MapSet_Group(object):
    def __init__(self, name='cmb', N=100,
                 recalc=False, strings=False,
                 calc_grad=True,calc_gradgrad=True,calc_rotstick=True,
                 **kwargs):

        if strings:
            name += '_stringy'
            
        for k,v in kwargs.items():
            name += '_{}{}'.format(k,v)

        logging.debug('Maps being created...')
        self.map = MapSet(name, N=N, recalc=recalc,
                            return_strings=strings, **kwargs)

        if calc_grad or calc_gradgrad or calc_rotstick:
            logging.debug('Grad Maps being created...')
            self.grad = MapSet('{}/grad'.format(name), N=N, recalc=recalc,
                                generator=generate_gradmaps,
                                parent_folder='{}/{}'.format(MAPSROOT,name),
                                scaleX=self.map.scaleX, scaleY=self.map.scaleY,
                                **kwargs)
        if calc_gradgrad  or calc_rotstick:
            logging.debug('GradGrad Maps being created...')
            self.gradgrad = MapSet('{}/grad/grad'.format(name), N=N, recalc=recalc,
                                   generator=generate_gradmaps,
                                   parent_folder='{}/{}/grad'.format(MAPSROOT,name),
                                   scaleX=self.grad.scaleX, scaleY=self.grad.scaleY,
                                   **kwargs)
        if calc_rotstick:
            logging.debug('(gradgrad) Rotated Stick Convolution Maps being created...')
            self.gradgrad_rotstick = MapSet('{}/grad/grad/rotstick'.format(name), N=N, recalc=recalc,
                                generator=generate_rotated_stick_convolutions,
                                parent_folder='{}/{}/grad/grad'.format(MAPSROOT,name),
                                default_edge=30,
                                **kwargs)
        
class MapSet(object):
    def __init__(self, folder='cmb', N=100,
                 recalc=False, map_fov_deg=7.2,
                 generator=obtain_N_cmb_maps,
                 default_edge=5,
                 **kwargs):
        """
        folder is a folder (under MAPSROOT)
        where N maps are stored
        as map0.npy, map1.npy, etc. 

        if files are not in folder, N maps will be
        generated according to keyword arguments
        provided, passed to generator function.

        generator function must be able to be called::

            generator(Nmaps=N, map_fov_deg=map_fov_deg, **kwargs)

        and return N maps.
        """
        
        folder = '{}/{}'.format(MAPSROOT,folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.folder = folder

        self.map_fov_deg = map_fov_deg
        self.N = N
        self.kwargs = kwargs
        self.default_edge = default_edge
        
        mapfiles = glob.glob('{}/map*.npy'.format(folder))
            
        if len(mapfiles) >= N and not recalc:
            self.maps = [np.load(f) for f in mapfiles[:N]]
            kwargs = self.load_kwargs()
            self.kwargs = kwargs
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
                        
            #self.maps = obtain_N_cmb_maps(Nmaps=N, map_fov_deg=map_fov_deg,
            #                              **kwargs)
            self.maps = generator(Nmaps=N, map_fov_deg=map_fov_deg, **kwargs)
            self.save_maps()
            self.write_kwargs(**kwargs)

        Nx, Ny = self.maps[0].shape
        self.scaleX = np.deg2rad(map_fov_deg / Nx)
        self.scaleY = np.deg2rad(map_fov_deg / Ny)

        self.statistics = {}
        self.N = len(self.maps)

    def generate(self):
        return obtain_N_cmb_maps(Nmaps=self.N, map_fov_deg=self.map_fov_deg,
                                 **self.kwargs)
    
    def __getitem__(self, i):
        return self.maps[i]

    def edged(self, i, edge=None):
        if edge is None:
            edge = self.default_edge
        return self[i][edge:-edge, edge:-edge]
    
    def plot_map(self,i=0, figsize=(6.5,6.5),
                 edge=None, colorbar=True,
                 ax=None, fig=None,
                 title='',
                 **kwargs):
        
        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()
            fig = plt.gcf()
        else:
            fig = plt.gcf()
            
        cax = ax.imshow(self.edged(i, edge), **kwargs)
        ax.set_title(title)
        if colorbar:
            fig.colorbar(cax)

            
    def load_kwargs(self):
        kwargs_file = '{}/kwargs.pkl'.format(self.folder)
        return pickle.load(open(kwargs_file, 'rb'))
        
    def write_kwargs(self, **kwargs):
        kwargs_file = '{}/kwargs.pkl'.format(self.folder)
        fout = open(kwargs_file, 'wb')
        pickle.dump(kwargs, fout)
        fout.close()        
        
    def save_maps(self):
        for i,m in enumerate(self.maps):
            fname = '{}/map{}.npy'.format(self.folder,i)
            np.save(fname, m)
            
    def apply_statistic(self, s, edge=None,**kwargs):
        if not isinstance(s, Statistic):
            raise ValueError('Must pass valid Statistic object')

        s_all = []
        for i in range(self.N):
            m = self.edged(i, edge)
            s_all.append(s(m,**kwargs))
        s_all = np.array(s_all)

        self.statistics[s.name] = s_all

                        
class Stringy_MapSet(MapSet):
    def __init__(self):
        pass
        
class Grad_MapSet(MapSet):
    def __init__(self, folder, parent_folder, recalc=False, **kwargs):
        self.parent_folder = parent_folder
        self.folder = '{}/{}/grad'.format(MAPSROOT, folder)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        
        mapset = MapSet(folder, recalc=recalc, **kwargs)
        self.scaleX = mapset.scaleX
        self.scaleY = mapset.scaleY
        self.N = mapset.N
        mapfiles = glob.glob('{}/map*.npy'.format(self.folder))
                             
        if len(mapfiles) == mapset.N and not recalc:
            self.maps = [np.load(f) for f in mapfiles[:mapset.N]]
            self.kwargs = self.load_kwargs()
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

            self.maps = []
            for m in mapset.maps:
                gradx,grady = np.gradient(m,self.scaleX,self.scaleY)
                self.maps.append(np.sqrt(gradx**2 + grady**2))
            self.save_maps()
            self.write_kwargs(**kwargs)
            self.kwargs = kwargs

        self.statistics = {}

    def generate(self, recalc=False):
        """doesn't work yet; that's why __init__ is re-written instead of using MapSet.__init__..."""
        
        mapset = MapSet(self.parent_folder, recalc=recalc, **self.kwargs)
        maps = []
        for m in mapset.maps:
            gradx,grady = np.gradient(m,self.scaleX,self.scaleY)
            maps.append(np.sqrt(gradx**2 + grady**2))
        return maps
    
class GradGrad_MapSet(Grad_MapSet):
    def __init__(self, folder, **kwargs):
        Grad_MapSet.__init__(self, '{}/grad'.format(folder), **kwargs)
