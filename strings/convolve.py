import logging
import glob
import numpy as np
import scipy
import utils #this is from flipper


def rotated_stick_convolve(mdata, n_angles=20, width=1, length=20,
                           scale=None, **kwargs):
    """ Returns stack of convolved
    this works only if scaleX is close to scaleY
    """
    
    arcmin2pix = 2.

    position_angles = np.linspace(0,180,n_angles)
    conv_all = []

    for pa in position_angles:
        sk = utils.stickKern(length*arcmin2pix, width*arcmin2pix, position_angle=pa, enhancement=1)
        convolved = scipy.signal.fftconvolve(mdata, sk, mode='same')
        conv_all.append(convolved)

    return np.array(conv_all)
    

def generate_rotated_stick_convolutions(parent_folder=None, combine_method='max', **kwargs):
                                        
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
