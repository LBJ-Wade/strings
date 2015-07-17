import logging
import glob
import numpy as np
import scipy
import utils #this is from flipper
import pylab as pl


def rotated_stick_convolve(mdata, n_angles=20, width=3, length=80,
                           scale=None, mask_percentage=0.,masking_edge=80,
                           **kwargs):
    """ 
    this works only if scaleX is close to scaleY
    takes length and width in pixels. 
    note: first try was width=1arcmin, length=20arcmin.
    note2: second try (in pixels): width=10, length=100.
    note3: third try (in pixels), for grad: width=5, length=80.
    """
    
    #arcmin2pix = 2.
    if np.isclose(mask_percentage,0.):
        mdata_masked = mdata.copy()
    else:
        mdataedged = mdata[masking_edge:-masking_edge,masking_edge:-masking_edge]
        cutoff = np.percentile(mdataedged.ravel(),100-mask_percentage)
        #print cutoff
        maskpoints = mdata/mdata
        maskpoints[np.where(mdata > cutoff)] = 0.
        #pl.figure()
        #pl.imshow(maskpoints)
        #pl.colorbar()
        #pl.title('mask points')
        maskadd = mdata.copy()*0.
        maskadd[np.where(mdata > cutoff)] = mdata.mean()
        #pl.figure()
        #pl.imshow(maskadd)
        #pl.colorbar()
        #pl.title('mask add')
        mdata_masked = mdata*maskpoints + maskadd
        #pl.figure()
        #pl.imshow(mdata_masked[masking_edge:-masking_edge,masking_edge:-masking_edge])
        #pl.colorbar()
        
    position_angles = np.linspace(0,180,n_angles)
    conv_all = []

    for pa in position_angles:
        #sk = utils.stickKern(length*arcmin2pix, width*arcmin2pix, position_angle=pa, enhancement=1)
        sk = utils.stickKern(length, width, position_angle=pa, enhancement=1)
        convolved = scipy.signal.fftconvolve(mdata_masked, sk, mode='same')
        conv_all.append(convolved)

    return np.array(conv_all)
    

def generate_rotated_stick_convolutions_old(parent_folder=None, combine_method='max', **kwargs):
                                        
    mapfiles = glob.glob('{}/map*.npy'.format(parent_folder))
    maps = []
    for i,mf in enumerate(mapfiles):
        logging.info('{} of {}'.format(i+1, len(mapfiles)))
        m = np.load(mf)
        mapstack = rotated_stick_convolve(m, **kwargs)
        if combine_method=='max':
            res = mapstack.max(axis=0)
        maps.append(res)
    return maps

def generate_rotated_stick_convolutions(parent_folder=None, 
                                        combine_method='max',
                                        mask_percentage=0.,
                                        masking_edge=80,
                                        **kwargs):
                                        
    mapfiles = glob.glob('{}/map*.npy'.format(parent_folder))
    maps = []
    for i,mf in enumerate(mapfiles):
        logging.info('{} of {}'.format(i+1, len(mapfiles)))
        m = np.load(mf)
        mapstack = rotated_stick_convolve(m, 
                                          mask_percentage=mask_percentage,
                                          masking_edge=masking_edge,
                                          **kwargs)
        
        if combine_method=='max':
            res = mapstack.max(axis=0)
        if combine_method=='angle':
            n_angles = len(mapstack)
            position_angles = np.linspace(0,180,n_angles)
            sin_pa = np.sin(position_angles)
            res = sin_pa[mapstack.argmax(axis=0)]

        maps.append(res)

    return maps
