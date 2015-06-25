import numpy as np
import matplotlib.pyplot as plt
import logging
#import healpy as hp
from flipper import *
import utils
reload(utils)
import liteMap
reload(liteMap)
import fftTools
reload(fftTools)
from tests import *


flipper_root = '/home/verag/flipper-master/'

STRING_MAP_FILE = '/data/verag/strings/inputs/dTmap_stg_Dcmb0000.fits' #'/Users/verag/Research/strings/strings/data/dTmap_stg_Dcmb0000.fits'
SPECTRA_FILE = '/data/verag/strings/inputs/lensedCls.dat' #'/Users/verag/Research/strings/strings/data/lensedCls.dat'


def obtain_N_cmb_maps(fwhm=1.4, string_file_num=0,
                        spectra_file=SPECTRA_FILE, map_fov_deg=7.2,
                        return_strings=False,
                        beam_smear=True, Nmaps=1, Gmu=0.,
                        noise=16, Nx=None, Ny=None):

    if string_file_num < 10:
        string_map_file = '/data/verag/strings/inputs/dTmap_stg_Dcmb000{}.fits'.format(string_file_num) # this can be 0-9
    else:
        string_map_file = '/data/verag/strings/inputs/dTmap_stg_Dcmb00{}.fits'.format(string_file_num) # this can be 0-9

    maps = []
    maps_stringy = []
    
    Tcmb_uK = 2.72548*1e6
    deg2rad = np.pi/180.
    
    string_map =  liteMap.liteMapFromFits(string_map_file)
    string_map.data *= Gmu * Tcmb_uK
    string_map.area = map_fov_deg**2.
    string_map.Nx, string_map.Ny = np.shape(string_map.data)
    string_map.x0 = 0.
    string_map.y0 = 0.
    string_map.x1 = map_fov_deg
    string_map.y1 = map_fov_deg
    string_map.pixScaleX = deg2rad * map_fov_deg / string_map.Nx #this is in rad
    string_map.pixScaleY = deg2rad * map_fov_deg / string_map.Ny #this is in rad

    if Gmu>0:
        if beam_smear:
            string_map = string_map.convolveWithGaussian(fwhm=fwhm)

    cmb_map = string_map.copy()
    if Nx is not None:
        cmb_map.Nx = Nx
    if Ny is not None:
        cmb_map.Ny = Ny
    cmb_map.data = np.zeros((cmb_map.Nx,cmb_map.Ny))
    cmb_map.pixScaleX = deg2rad * map_fov_deg / cmb_map.Nx #this is in rad
    cmb_map.pixScaleY = deg2rad * map_fov_deg / cmb_map.Ny #this is in rad


    spectra = np.loadtxt(spectra_file)
    ls = spectra[:,0]
    cl = spectra[:,1]/(ls*(ls + 1))*2.*np.pi

    for i in np.arange(Nmaps):
        logging.info('{} of {}'.format(i+1, Nmaps))
        cmb_map.fillWithGaussianRandomField(ls,cl)
        if beam_smear:
            cmb_map = cmb_map.convolveWithGaussian(fwhm=fwhm)
        noise_map = np.random.standard_normal(np.shape(cmb_map.data)) * (noise)
        
        cmb = cmb_map.data + noise_map
        maps.append(cmb - cmb.mean())

        if Gmu>0:
            stringycmb = cmb_map.data + noise_map + string_map.data
            maps_stringy.append(stringycmb - stringycmb.mean())

    if Gmu>0 and return_strings:
        return maps_stringy
    else:
        return maps


def obtain_1string_map(fwhm=1.4, string_map_file=STRING_MAP_FILE,
                        map_fov_deg=7.2,
                        beam_smear=True, Gmu=1.4e-7):

    
    Tcmb_uK = 2.72548*1e6
    deg2rad = np.pi/180.
    
    string_map =  liteMap.liteMapFromFits(string_map_file)
    string_map.data *= Gmu * Tcmb_uK
    string_map.area = map_fov_deg**2.
    string_map.Nx, string_map.Ny = np.shape(string_map.data)
    string_map.x0 = 0.
    string_map.y0 = 0.
    string_map.x1 = map_fov_deg
    string_map.y1 = map_fov_deg
    string_map.pixScaleX = deg2rad * map_fov_deg / string_map.Nx #this is in rad
    string_map.pixScaleY = deg2rad * map_fov_deg / string_map.Ny #this is in rad
    
    if beam_smear:
        string_map = string_map.convolveWithGaussian(fwhm=fwhm)
    
    gradx,grady = np.gradient(string_map.data,string_map.pixScaleX,string_map.pixScaleY)
    gradmap = np.sqrt(gradx**2 + grady**2)
    gradx,grady = np.gradient(gradmap,string_map.pixScaleX,string_map.pixScaleY)
    gradgradmap = np.sqrt(gradx**2 + grady**2)

    return string_map.data, gradmap, gradgradmap
