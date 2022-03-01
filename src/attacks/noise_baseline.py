#!/usr/bin/env python3

from conf import settings

import torch
import os
import numpy as np


def noisy(image,noise_typ):
    """ 
    https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                 n is uniform noise with specified mean & variance.
    """
    
    image = image.astype('float32')
    
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 4./255. 
        sigma =  var #**0.5
        # noisy = gaussian_filter(image, sigma)
        gauss = np.random.normal(mean,sigma, size=(row,col,ch)).astype('float32')
        noisy = np.clip(image + gauss, 0, 1)

        return noisy #p.transpose( noisy, (1,2,0) )
    
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out.reshape(ch,row,col)
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy.reshape(ch,row,col)
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy #.reshape(ch,row,col)