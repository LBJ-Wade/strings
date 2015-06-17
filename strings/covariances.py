import matplotlib.pyplot as plt
import numpy as np
import os, os.path, glob, shutil
import sys,re
import cPickle as pickle

import logging

from .simulator import obtain_N_cmb_maps
from .statistic import Statistic
from .statistic import PDF, PowerSpectrum, Moments
from .convolve import generate_rotated_stick_convolutions
from .mapset import MapSet_Group

SPECTRA_FILE = '/data/verag/strings/inputs/lensedCls.dat'
STATS_ROOT = '/data/verag/strings/stats'
pdf_defaults={
    'binmin':{'gradgrad':0,
              'gradgrad_rotstick':3e8,
              'map':-500},
    'binmax':{'gradgrad':1e10,
              'gradgrad_rotstick':7e8,
              'map':500}}


        
def compute_largemap_stats(statnumber, whichmap='gradgrad_rotstick',
                           statname='pdf',
                            map_fov_deg=72., fwhm=1.4, noise=16.,
                            Nx=10240, Ny=10240,
                            Gmu=0., strings=False, string_file_num=0,
                            name='large_cmb',
                            stats_kwargs=None,
                            restag=None, returnres=False,saveres=True):

    """fwhm is in arcmin
    """
    calc_grad = False
    calc_gradgrad = False
    calc_rotstick = False
    if whichmap=='grad':
        calc_grad = True
    if whichmap=='gradgrad':
        calc_gradgrad = True
    if whichmap=='gradgrad_rotstick':
        calc_rotstick = True

    name += '{}'.format(statnumber)
    msg = MapSet_Group(N=1,
                        calc_grad=calc_grad, calc_gradgrad=calc_gradgrad,
                        calc_rotstick=calc_rotstick,
                        name=name, strings=strings,
                        recalc=True,
                        noise=noise,
                        fwhm=fwhm, string_file_num=string_file_num,
                        map_fov_deg=map_fov_deg,
                        Gmu=Gmu, Nx=Nx, Ny=Ny)

    newmap = getattr(msg, whichmap)

    if stats_kwargs is None:
        stats_kwargs = {}

    if statname=='pdf':
        for prop in ['binmin', 'binmax']:
            if prop not in stats_kwargs:
                stats_kwargs[prop] = pdf_defaults[prop][whichmap]
        if 'bins' not in stats_kwargs:
            stats_kwargs['bins'] = 110
        if 'normed' not in stats_kwargs:
            stats_kwargs['normed'] = False#True
            
        stat = PDF(**stats_kwargs)
        
    newmap.apply_statistic(stat)
    res = newmap.statistics[statname][0]
    print newmap.statistics[statname][0]
    
    del msg, newmap
    
    if saveres:
        if restag is None:
            restag = '{:.1f}deg2_fwhm{:.1f}_{:.0f}uK'.format(map_fov_deg,fwhm,noise)
            if strings:
                restag += '_Gmu{:.1e}_stringfile{}'.format(Gmu,string_file_num)

        resfile = STATS_ROOT + '/{}{}_{}.npy'.format(statname,statnumber,restag)        
        np.save(resfile, res)

    if returnres:
        return res
    

    
def compute_pdfs(Nmaps=20):
    for i in np.arange(Nmaps):
        print 'calculating for {}/{}...'.format(i+1,Nmaps)
        compute_largemap_stats(i, whichmap='gradgrad_rotstick',
                           statname='pdf',
                            map_fov_deg=72., fwhm=1.4, noise=16.,
                            Nx=10240, Ny=10240,
                            Gmu=0., strings=False, string_file_num=0,
                            name='large_cmb',
                            stats_kwargs=None,
                            restag=None, returnres=False,saveres=True)
