#!/usr/bin/env python3

import os, sys
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

from utils import (
    normalize_images,
    get_debug_info
)


# Gaussian noise scale sizes that were determined so that the average
# L-2 perturbation size is equal to that of the adversarial samples
# mnist roughly L2_difference/20
# cifar roughly L2_difference/54
# svhn roughly L2_difference/60
# be very carefully with these settings, tune to have noisy/adv have the same L2-norm
# otherwise artifact will lose its accuracy
# STDEVS = {
#     'mnist': {'fgsm': 0.264, 'bim-a': 0.111, 'bim-b': 0.184, 'cw-l2': 0.588},
#     'cifar': {'fgsm': 0.0504, 'bim-a': 0.0087, 'bim-b': 0.0439, 'cw-l2': 0.015},
#     'svhn': {'fgsm': 0.1332, 'bim-a': 0.015, 'bim-b': 0.1024, 'cw-l2': 0.0379}
# }

# fined tuned again when retrained all models with X in [-0.5, 0.5]
# https://github.com/pokaxpoka/deep_Mahalanobis_detector/blob/master/ADV_Samples.py
STDEVS = {
    'cif10':     {'apgd-ce': 0.02295211, 'apgd-cel2': 0.02295211, 'fgsm': 0.038821902, 'bim': 0.020980978, 'pgd': 0.02295211, 'std': 0.028931687, 'df': 0.006160808, 'cw': 0.0020574536},
    'cif100':    {'apgd-ce': 0.02295211, 'apgd-cel2': 0.02295211, 'fgsm': 0.015686,    'bim': 0.015686,    'pgd': 0.015686,   'std': 0.015686,    'df': 0.015686,    'cw': 0.015686},
    'cif10vgg':  {'apgd-ce': 0.02295211, 'apgd-cel2': 0.02295211, 'fgsm': 0.015686,    'bim': 0.015686,    'pgd': 0.015686,   'std': 0.015686,    'df': 0.015686,    'cw': 0.015686},
    'cif100vgg': {'apgd-ce': 0.02295211, 'apgd-cel2': 0.02295211, 'fgsm': 0.015686,    'bim': 0.015686,    'pgd': 0.015686,   'std': 0.015686,    'df': 0.015686,    'cw': 0.015686},
    'imagenet':  {'apgd-ce': 0.02295211, 'apgd-cel2': 0.02295211, 'fgsm': 0.015686,    'bim': 0.015686,    'pgd': 0.015686,   'std': 0.015686,    'df': 0.015686,    'cw': 0.015686},
}

CLIP_MIN = 0.0
CLIP_MAX = 1.0

# Set random seed
np.random.seed(0)


def get_k(args):
    #hyperparameters
    batch_size = 100
    if args.k_lid < 0:
        if args.net == 'mnist' or args.net == 'cif10' or args.net == 'cif10vgg' or args.net == 'imagenet32' or args.net == 'celebaHQ32':
            k = 10
        else: # cif100 cif100vgg imagenet imagenet64
            k = 20
            batch_size = 64
            
            if args.net == 'imagenet':
                batch_size = 32
    else:
        k = args.k_lid
        
    return k , batch_size


def lid_newest(args, model, images, images_advs, layers, get_layer_feature_maps, activation):

    k, batch_size = get_k(args)
    
    def mle_batch(data, batch, k):
        data = np.asarray(data, dtype=np.float32)
        batch = np.asarray(batch, dtype=np.float32)
        k = min(k, len(data)-1)
        f = lambda v: -k / np.sum(np.log(v/v[-1]))
        a = cdist(batch, data)
     
        a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
        a = np.apply_along_axis(f, axis=1, arr=a)
        return a

    lid_dim = len(layers)
    shape = np.shape(images[0])
    
    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(images), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch       = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_adv   = np.zeros(shape=(n_feed, lid_dim))
        batch= torch.Tensor(n_feed, shape[0], shape[1], shape[2])
        batch_adv= torch.Tensor(n_feed, shape[0], shape[1], shape[2])
        for j in range(n_feed):
            batch[j,:,:,:] = images[j]
            batch_adv[j,:,:,:] = images_advs[j]

        batch = normalize_images(batch, args)
        batch_adv = normalize_images(batch_adv, args)

        if not args.net == 'cif10vgg' and not args.net == 'cif100vgg':
            feat_img = model(batch.cuda())
            X_act = get_layer_feature_maps(activation, layers)

            feat_adv = model(batch_adv.cuda())
            X_adv_act = get_layer_feature_maps(activation, layers)
        else:
            X_act = get_layer_feature_maps(batch.cuda(), layers)
            X_adv_act = get_layer_feature_maps(batch_adv.cuda(), layers)

        for i in range(lid_dim):
            X_act[i]       = np.asarray(X_act[i].cpu().detach().numpy()    , dtype=np.float32).reshape((n_feed, -1))
            X_adv_act[i]   = np.asarray(X_adv_act[i].cpu().detach().numpy(), dtype=np.float32).reshape((n_feed, -1))
            # Maximum likelihood estimation of Local Intrinsic Dimensionality (LID)
            lid_batch[:, i]       = mle_batch(X_act[i], X_act[i]      , k=k)
            lid_batch_adv[:, i]   = mle_batch(X_act[i], X_adv_act[i]  , k=k)
        return lid_batch, lid_batch_adv

    lids = []
    lids_adv = []
    n_batches = int(np.ceil(len(images) / float(batch_size)))
    
    for i_batch in tqdm(range(n_batches)):
        lid_batch, lid_batch_adv = estimate(i_batch)
        lids.extend(lid_batch)
        lids_adv.extend(lid_batch_adv)

    characteristics     = np.asarray(lids,     dtype=np.float32)
    characteristics_adv = np.asarray(lids_adv, dtype=np.float32)

    return characteristics, characteristics_adv


def lid(args, model, images, images_advs, layers, get_layer_feature_maps, activation):

    act_layers = layers
    k, batch_size = get_k(args)
    
    def mle_batch(data, batch, k):
        data  = np.asarray(data,  dtype=np.float32)
        batch = np.asarray(batch, dtype=np.float32)
        k = min(k, len(data)-1)
        
        f = lambda v: - k / np.sum(np.log(v/v[-1]))
        a = cdist(batch, data)
        a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
        a = np.apply_along_axis(f, axis=1, arr=a)
        return a

    lid_dim = len(act_layers)
    shape = np.shape(images[0])
    
    
    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(images), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch       = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_adv   = np.zeros(shape=(n_feed, lid_dim))
        batch= torch.Tensor(n_feed, shape[0], shape[1], shape[2])
        batch_adv= torch.Tensor(n_feed, shape[0], shape[1], shape[2])
        for j in range(n_feed):
            batch[j,:,:,:] = images[j]
            batch_adv[j,:,:,:] = images_advs[j]
        # batch = cifar_normalize(batch)
        # batch_adv = cifar_normalize(batch_adv)
        # X_act       = get_layer_feature_maps(batch.to(device), act_layers)
        # X_adv_act   = get_layer_feature_maps(batch_adv.to(device), act_layers)

        batch = normalize_images(batch, args)
        batch_adv = normalize_images(batch_adv, args)

        if not args.net == 'cif10vgg' and not args.net == 'cif100vgg':
            feat_img = model(batch.cuda())
            X_act = get_layer_feature_maps(activation, layers)

            feat_adv = model(batch_adv.cuda())
            X_adv_act = get_layer_feature_maps(activation, layers)
        else:
            X_act     = get_layer_feature_maps(batch.cuda(), layers)
            X_adv_act = get_layer_feature_maps(batch_adv.cuda(), layers)
        
        for i in range(lid_dim):
            X_act[i]       = np.asarray(X_act[i].cpu().detach().numpy()    , dtype=np.float32).reshape((n_feed, -1))
            X_adv_act[i]   = np.asarray(X_adv_act[i].cpu().detach().numpy(), dtype=np.float32).reshape((n_feed, -1))
            
            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            lid_batch[:, i]       = mle_batch(X_act[i], X_act[i]      , k=k)
            lid_batch_adv[:, i]   = mle_batch(X_act[i], X_adv_act[i]  , k=k)
        return lid_batch, lid_batch_adv

    lids = []
    lids_adv = []
    n_batches = int(np.ceil(len(images) / float(batch_size)))
    
    for i_batch in tqdm(range(n_batches)):
        lid_batch, lid_batch_adv = estimate(i_batch)
        lids.extend(lid_batch)
        lids_adv.extend(lid_batch_adv)

    characteristics       = np.asarray(lids, dtype=np.float32)
    characteristics_adv   = np.asarray(lids_adv, dtype=np.float32)

    return characteristics, characteristics_adv


def lidnoise(args, model, images, images_advs, layers, get_layer_feature_maps, activation):
    
    act_layers = layers
    k, batch_size = get_k(args)
    # import pdb; pdb.set_trace()
    
    def mle_batch(data, batch, k):
        
        data  = np.asarray(data,  dtype=np.float32)
        batch = np.asarray(batch, dtype=np.float32)
        k = min(k, len(data)-1)
        
        f  = lambda v: - k / np.sum( np.log(v/v[-1]) )
        f2 = lambda v: - np.log( v / v[-1] )
                
        dist = cdist(batch, data)
        dist = np.apply_along_axis(np.sort, axis=1, arr=dist)[:,1:k+1]
        sol = np.apply_along_axis(f, axis=1, arr=dist)
        
        # import pdb; pdb.set_trace()
        a2 = np.apply_along_axis(f2, axis=1, arr=dist)
        
        # a2 = np.zeros_like(dist)
        # for j in range(k):
            # for i in range(dist.shape[0]):
                # a2[i,j] =  dist[]
        
        return sol, a2

    lid_dim = len(act_layers)
    shape = np.shape(images[0])
    
    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(images), (i_batch + 1) * batch_size)
        n_feed = end - start
        
        lid_batch       = np.zeros(shape=(1))
        lid_batch_noise = np.zeros(shape=(1))
        lid_batch_adv   = np.zeros(shape=(1))
        
        lid_batch_k       = np.zeros(shape=(n_feed, k, lid_dim))
        lid_batch_noise_k = np.zeros(shape=(1))
        lid_batch_adv_k   = np.zeros(shape=(n_feed, k, lid_dim))
        
        batch       = torch.Tensor(n_feed, shape[0], shape[1], shape[2])
        batch_noise = torch.Tensor(1)
        batch_adv   = torch.Tensor(n_feed, shape[0], shape[1], shape[2])
        
        for j in range(n_feed):
            batch[j,:,:,:]       = images[j]
            # batch_noise[j,:,:,:] = images[j]
            batch_adv[j,:,:,:]   = images_advs[j]
        # batch = cifar_normalize(batch)
        # batch_adv = cifar_normalize(batch_adv)
        # X_act       = get_layer_feature_maps(batch.to(device), act_layers)
        # X_adv_act   = get_layer_feature_maps(batch_adv.to(device), act_layers)
        
        # batch_noise_init = get_noisy_samples(batch_noise, dataset=args.net, attack=args.attack)

        batch       = normalize_images(batch, args)
        # batch_noise = normalize_images(batch_noise_init, args)
        batch_adv   = normalize_images(batch_adv, args)

        if not args.net == 'cif10vgg' and not args.net == 'cif100vgg':
            feat_img = model(batch.cuda())
            X_act = get_layer_feature_maps(activation, layers)

            # feat_noise  = model(batch_noise.cuda())
            # X_noise_act = get_layer_feature_maps(activation, layers)

            feat_adv = model(batch_adv.cuda())
            X_adv_act = get_layer_feature_maps(activation, layers)
        else:
            X_act       = get_layer_feature_maps(batch.cuda(), layers)
            # X_noise_act = get_layer_feature_maps(batch_noise.cuda(), layers)
            X_adv_act   = get_layer_feature_maps(batch_adv.cuda(), layers)
        
        for i in range(lid_dim):
            X_act[i]       = np.asarray( X_act[i].cpu().detach().numpy()      ,dtype=np.float32).reshape((n_feed, -1) )
            # X_noise_act[i] = np.asarray( X_noise_act[i].cpu().detach().numpy(),dtype=np.float32).reshape((n_feed, -1) )
            X_adv_act[i]   = np.asarray( X_adv_act[i].cpu().detach().numpy()  ,dtype=np.float32).reshape((n_feed, -1) )
            
            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            tmp_batch, tmp_k              = mle_batch( X_act[i], X_act[i]      , k=k )
            # tmp_batch_noise, tmp_k_noise  = mle_batch( X_act[i], X_noise_act[i], k=k )
            tmp_batch_adv, tmp_k_adv      = mle_batch( X_act[i], X_adv_act[i]  , k=k )   
            
            lid_batch_k[:, :, i]       = tmp_k
            # lid_batch_noise_k[:, :, i] = tmp_k_noise
            lid_batch_adv_k[:, :, i]   = tmp_k_adv
            
            # lid_batch[:, i]       = tmp_batch
            # lid_batch_noise[:, i] = tmp_batch_noise
            # lid_batch_adv[:, i]   = tmp_batch_adv
            
        return lid_batch, lid_batch_noise, lid_batch_adv, lid_batch_k, lid_batch_noise_k, lid_batch_adv_k

    lids = []
    lid_noise = []
    lids_adv = []
    lid_tmp_k = []
    lid_tmp_k_noise = [] 
    lid_tmp_k_adv = []
    
    n_batches = int(np.ceil(len(images) / float(batch_size)))
    
    for i_batch in tqdm(range(n_batches)):
        lid_batch, lid_batch_noise, lid_batch_adv, lid_tmp_k_batch, lid_tmp_k_noise_batch, lid_tmp_k_adv_batch = estimate(i_batch)
        lids.extend(lid_batch)
        lid_noise.extend(lid_batch_noise)
        lids_adv.extend(lid_batch_adv)
    
        lid_tmp_k.extend(lid_tmp_k_batch)
        lid_tmp_k_noise.extend(lid_tmp_k_noise_batch)
        lid_tmp_k_adv.extend(lid_tmp_k_adv_batch)
    
    characteristics         = np.asarray(lids, dtype=np.float32)
    characteristics_noise   = np.asarray(lid_noise, dtype=np.float32)
    characteristics_adv     = np.asarray(lids_adv, dtype=np.float32)
    
    char_tmp_k       =  np.asarray(lid_tmp_k, dtype=np.float32)
    char_tmp_k_noise =  np.asarray(lid_tmp_k_noise, dtype=np.float32)
    char_tmp_k_adv   =  np.asarray(lid_tmp_k_adv, dtype=np.float32)

    return characteristics, characteristics_noise, characteristics_adv, char_tmp_k, char_tmp_k_noise, char_tmp_k_adv


def lidnoise_save(args, model, images, images_advs, layers, get_layer_feature_maps, activation):
    
    act_layers = layers
    k, batch_size = get_k(args)
    # import pdb; pdb.set_trace()
    
    def mle_batch(data, batch, k):
        
        data  = np.asarray(data,  dtype=np.float32)
        batch = np.asarray(batch, dtype=np.float32)
        k = min(k, len(data)-1)
        
        f  = lambda v: - k / np.sum( np.log(v/v[-1]) )
        f2 = lambda v: - np.log( v/ v[-1] )
        f3 = lambda v: - 1 /  ( np.log(v / k)  - np.log(v[-1]))
                
        dist = cdist(batch, data)
        dist = np.apply_along_axis(np.sort, axis=1, arr=dist)[:,1:k+1]
        sol = np.apply_along_axis(f, axis=1, arr=dist)
        
        # import pdb; pdb.set_trace()
        a2 = np.apply_along_axis(f2, axis=1, arr=dist)
        
        # a2 = np.zeros_like(dist)
        # for j in range(k):
            # for i in range(dist.shape[0]):
                # a2[i,j] =  dist[]
        
        return sol, a2

    lid_dim = len(act_layers)
    shape = np.shape(images[0])
    
    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(images), (i_batch + 1) * batch_size)
        n_feed = end - start
        
        lid_batch       = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_noise = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_adv   = np.zeros(shape=(n_feed, lid_dim))
        
        lid_batch_k       = np.zeros(shape=(n_feed, k, lid_dim))
        lid_batch_noise_k = np.zeros(shape=(n_feed, k, lid_dim))
        lid_batch_adv_k   = np.zeros(shape=(n_feed, k, lid_dim))
        
        batch       = torch.Tensor(n_feed, shape[0], shape[1], shape[2])
        batch_noise = torch.Tensor(n_feed, shape[0], shape[1], shape[2])
        batch_adv   = torch.Tensor(n_feed, shape[0], shape[1], shape[2])
        
        for j in range(n_feed):
            batch[j,:,:,:]       = images[j]
            batch_noise[j,:,:,:] = images[j]
            batch_adv[j,:,:,:]   = images_advs[j]
        # batch = cifar_normalize(batch)
        # batch_adv = cifar_normalize(batch_adv)
        # X_act       = get_layer_feature_maps(batch.to(device), act_layers)
        # X_adv_act   = get_layer_feature_maps(batch_adv.to(device), act_layers)
        
        batch_noise_init = get_noisy_samples(batch_noise, dataset=args.net, attack=args.attack)

        batch       = normalize_images(batch, args)
        batch_noise = normalize_images(batch_noise_init, args)
        batch_adv   = normalize_images(batch_adv, args)

        if not args.net == 'cif10vgg' and not args.net == 'cif100vgg':
            feat_img = model(batch.cuda())
            X_act = get_layer_feature_maps(activation, layers)

            feat_noise  = model(batch_noise.cuda())
            X_noise_act = get_layer_feature_maps(activation, layers)

            feat_adv = model(batch_adv.cuda())
            X_adv_act = get_layer_feature_maps(activation, layers)
        else:
            X_act       = get_layer_feature_maps(batch.cuda(), layers)
            X_noise_act = get_layer_feature_maps(batch_noise.cuda(), layers)
            X_adv_act   = get_layer_feature_maps(batch_adv.cuda(), layers)
        
        for i in range(lid_dim):
            X_act[i]       = np.asarray( X_act[i].cpu().detach().numpy()      ,dtype=np.float32).reshape((n_feed, -1) )
            X_noise_act[i] = np.asarray( X_noise_act[i].cpu().detach().numpy(),dtype=np.float32).reshape((n_feed, -1) )
            X_adv_act[i]   = np.asarray( X_adv_act[i].cpu().detach().numpy()  ,dtype=np.float32).reshape((n_feed, -1) )
            
            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            tmp_batch, tmp_k              = mle_batch( X_act[i], X_act[i]      , k=k )
            tmp_batch_noise, tmp_k_noise  = mle_batch( X_act[i], X_noise_act[i], k=k )
            tmp_batch_adv, tmp_k_adv      = mle_batch( X_act[i], X_adv_act[i]  , k=k )   
            
            lid_batch_k[:, :, i]       = tmp_k
            lid_batch_noise_k[:, :, i] = tmp_k_noise
            lid_batch_adv_k[:, :, i]   = tmp_k_adv
            
            lid_batch[:, i]       = tmp_batch
            lid_batch_noise[:, i] = tmp_batch_noise
            lid_batch_adv[:, i]   = tmp_batch_adv
            
        return lid_batch, lid_batch_noise, lid_batch_adv, lid_batch_k, lid_batch_noise_k, lid_batch_adv_k

    lids = []
    lid_noise = []
    lids_adv = []
    lid_tmp_k = []
    lid_tmp_k_noise = [] 
    lid_tmp_k_adv = []
    
    n_batches = int(np.ceil(len(images) / float(batch_size)))
    
    for i_batch in tqdm(range(n_batches)):
        lid_batch, lid_batch_noise, lid_batch_adv, lid_tmp_k_batch, lid_tmp_k_noise_batch, lid_tmp_k_adv_batch = estimate(i_batch)
        lids.extend(lid_batch)
        lid_noise.extend(lid_batch_noise)
        lids_adv.extend(lid_batch_adv)
    
        lid_tmp_k.extend(lid_tmp_k_batch)
        lid_tmp_k_noise.extend(lid_tmp_k_noise_batch)
        lid_tmp_k_adv.extend(lid_tmp_k_adv_batch)
    
    characteristics         = np.asarray(lids, dtype=np.float32)
    characteristics_noise   = np.asarray(lid_noise, dtype=np.float32)
    characteristics_adv     = np.asarray(lids_adv, dtype=np.float32)
    
    char_tmp_k       =  np.asarray(lid_tmp_k, dtype=np.float32)
    char_tmp_k_noise =  np.asarray(lid_tmp_k_noise, dtype=np.float32)
    char_tmp_k_adv   =  np.asarray(lid_tmp_k_adv, dtype=np.float32)

    return characteristics, characteristics_noise, characteristics_adv, char_tmp_k, char_tmp_k_noise, char_tmp_k_adv


def get_noisy_samples(X_test, dataset='cif10', attack='fgsm'):
    # Add Gaussian noise to the samples
    # print(STDEVS[dataset][attack])
    X_test_noisy = np.minimum(
        np.maximum(
            X_test + np.random.normal(loc=0, scale=STDEVS[dataset][attack], size=X_test.shape).astype('f'),
            CLIP_MIN
        ),
        CLIP_MAX
    )

    return X_test_noisy