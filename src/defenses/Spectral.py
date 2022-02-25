#!/usr/bin/env python3

"""
our proposed method
"""

import os
import torch    
import numpy as np
from tqdm import tqdm

from utils import normalize_images

def calculate_fourier_spectrum(im, typ='MFS'):
    im = im.float()
    im = im.cpu()
    im = im.data.numpy()

    # im = im[:,:100,:,:]
    # im = im[:,:100,:5,:5]


    fft = np.fft.fft2(im)

    # im = im[:,:100,:5,:5]

    if typ == 'MFS':
        fourier_spectrum = np.abs(fft)
    elif typ == 'PFS':
        fourier_spectrum = np.abs(np.angle(fft))

    # import pdb; pdb.set_trace()
    # fourier_spectrum = np.max(fourier_spectrum, axis=1)
    # fourier_spectrum = np.mean(fourier_spectrum, axis=1)        
    
    return fourier_spectrum



def calculate_fourier_spectrum_analysis(im, fr, to, typ='MFS'):
    im = im.float()
    im = im.cpu()
    im = im.data.numpy() #[3, 32, 32]

    fft = np.fft.fft2(im)
    # fft = np.max( fft, axis=0 )
    # fft = np.mean( fft, axis=0 )
    if typ == 'MFS':
        fourier_spectrum = np.abs(fft)
    elif typ == 'PFS':
        fourier_spectrum = np.abs(np.angle(fft))

    # fourier_spectrum =  np.fft.fftshift(fourier_spectrum)       
    return fourier_spectrum[:, fr:to, fr:to]


def calculate_spectra_analysis(images, fr, to, typ='MFS'):
    fs = []   

    for i in tqdm(range(len(images))):
        image = images[i]
        fourier_image = calculate_fourier_spectrum_analysis(image, fr, to, typ=typ)
        fs.append(fourier_image.flatten())
    return fs


def calculate_spectra(images, typ='MFS'):
    fs = []   
    for i in range(len(images)):
        image = images[i]
        fourier_image = calculate_fourier_spectrum(image, typ=typ)
        fs.append(fourier_image.flatten())
    return fs


def blackbox_mfs_analysis(args, images, images_advs):
    """
    Apply FFT on the input images
    mfs: magnitute
    pfs: phase

    from: only take fft from x
    to: only take fft from y
    """
    mfs = calculate_spectra_analysis(images, args.fr, args.to)
    mfs_advs = calculate_spectra_analysis(images_advs, args.fr, args.to)
    characteristics       = np.asarray(mfs, dtype=np.float32)
    characteristics_adv   = np.asarray(mfs_advs, dtype=np.float32)
    return characteristics, characteristics_adv


def blackbox_mfs_pfs(args, images, images_advs, typ='MFS'):
    """
    Apply FFT on the input images
    mfs: magnitute
    pfs: phase
    """
    mfs      = calculate_spectra(images, typ=typ)
    mfs_advs = calculate_spectra(images_advs, typ=typ)
    characteristics       = np.asarray(mfs, dtype=np.float32)
    characteristics_adv   = np.asarray(mfs_advs, dtype=np.float32)
    return characteristics, characteristics_adv

    

###Fourier Layer   
def whitebox_mfs_pfs(args, model, images, images_advs, layers, get_layer_feature_maps, activation, typ='MFS'):
    """
    Extract the feature from the layers and apply FFT on that. 
    """
    mfs = []
    mfs_advs = []

    number_images = len(images)
    for it in tqdm(range(number_images)):

        # import pdb; pdb.set_trace()
        image = images[it].unsqueeze_(0)
        adv = images_advs[it].unsqueeze_(0)

        image = normalize_images(image, args)
        adv   = normalize_images(adv, args)

        # import pdb; pdb.set_trace()
        
        inputimage = []
        inputadv = []
        if not args.net == 'cif10vgg' and not args.net == 'cif100vgg':
            if args.take_inputimage_off:
                inputimage = [image]
                inputadv = [adv]
            if args.nr == -1:
                feat_img = model(image.cuda())
                image_feature_maps = inputimage + get_layer_feature_maps(activation, layers)

                feat_adv = model(adv.cuda())
                adv_feature_maps   = inputadv   + get_layer_feature_maps(activation, layers)
            else:
                feat_img = model(image.cuda())
                image_feature_maps = inputimage + get_layer_feature_maps(activation, layers)

                feat_adv = model(adv.cuda())
                adv_feature_maps   = inputadv   + get_layer_feature_maps(activation, layers)
        else:
            image_c = image.cuda()
            adv_c = adv.cuda()
            if args.take_inputimage_off:
                inputimage = [image_c]
                inputadv   = [adv_c]
            image_feature_maps = inputimage + get_layer_feature_maps(image_c, layers)
            adv_feature_maps   = inputadv   + get_layer_feature_maps(adv_c,   layers)
        
        fourier_maps     = calculate_spectra(image_feature_maps, typ=typ)
        fourier_maps_adv = calculate_spectra(adv_feature_maps, typ=typ)
        mfs.append(np.hstack(fourier_maps))
        mfs_advs.append(np.hstack(fourier_maps_adv))

    if not args.nr == -1:
        nr_param = 1
        for i in image_feature_maps[0].shape:
            nr_param = nr_param * i
        logger.log("INFO: parameters: " +  str(image_feature_maps[0].shape) + ', '  + str(nr_param) )
        
    characteristics       = np.asarray(mfs,      dtype=np.float32)
    characteristics_adv   = np.asarray(mfs_advs, dtype=np.float32)

    return characteristics, characteristics_adv
