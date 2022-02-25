#!/usr/bin/env python3

import os, sys
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

from utils import normalize_images

def lid(args, model, images, images_advs, layers, get_layer_feature_maps, activation):
    #hyperparameters
    batch_size = 100
    if args.net == 'mnist' or args.net == 'cif10' or args.net == 'cif10vgg' or args.net == 'imagenet32' or args.net == 'celebaHQ32':
        k = 20
    else: # cif100 cif100vgg imagenet imagenet64
        k = 10
        batch_size = 64

    
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