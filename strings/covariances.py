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
    

    
def compute_pdfs(Nstart=0, Nmaps=1, Nstringfiles=100, strings=True, Gmu=1.4e-7):
    if strings:
        Nx = 1024
        Ny = 1024
        map_fov_deg = 7.2
    else:
        Nx = 10240
        Ny = 10240
        map_fov_deg = 72.

    Nend = Nstart + Nmaps
    count = 0
    for j,num in enumerate(np.arange(Nstart,Nend)):
        for i in np.arange(Nstringfiles):
            count += 1
            print 'calculating for {} (map{},string{})/{}...'.format(count,j,i,Nmaps*Nstringfiles)
            compute_largemap_stats(num, whichmap='gradgrad_rotstick',
                                   statname='pdf',
                                   map_fov_deg=map_fov_deg, fwhm=1.4, noise=16.,
                                   Nx=Nx, Ny=Ny,
                                   Gmu=Gmu, strings=strings, string_file_num=i,
                                   name='large_stringy',
                                   stats_kwargs=None,
                                   restag=None, returnres=False,saveres=True)



def sigma_Gmu(Gmu, h1, h2, h1S, h2S): 
    covh = np.zeros((2,2))
    covh_inv = np.zeros((2,2))
    covh[0,0] = (h1 * h1).mean() - h1.mean() * h1.mean()
    covh[1,1] = (h2 * h2).mean() - h2.mean() * h2.mean()
    covh[0,1] = (h1 * h2).mean() - h1.mean() * h2.mean()
    covh[1,0] = covh[0,1]
    det_covh = covh[0,0]*covh[1,1] - covh[0,1]**2

    covh_inv[0,0] = covh[1,1]/det_covh
    covh_inv[1,1] = covh[0,0]/det_covh
    covh_inv[0,1] = -covh[0,1]/det_covh
    covh_inv[1,0] = covh[1,0]/det_covh

    sigma2_inv = h1S.mean()**2 * covh_inv[0,0] + h2S.mean()**2 * covh_inv[1,1] + 2.*h1S.mean()*h2S.mean() * covh_inv[0,1]
    res = 1./sigma2_inv**0.5
    print res/Gmu
    return res/Gmu
    
