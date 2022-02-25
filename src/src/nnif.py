import matplotlib
import platform
# Force matplotlib to not use any Xwindows backend.
if platform.system() == 'Linux':
    matplotlib.use('Agg')

import logging
import numpy as np
# import tensorflow as tf
import os
import pickle
from tqdm import tqdm
# from tensorflow.python.platform import flags
# from NNIF_adv_defense.models.darkon_resnet34_model import DarkonReplica
# from NNIF_adv_defense.datasets.influence_feeder import MyFeederValTest
# from NNIF_adv_defense.tools.utils import mle_batch
import sklearn.covariance
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from cleverhans.evaluation import batch_eval
from cleverhans.utils import set_log_level
import time

batch_size = 125


def get_knn_layers(X, y):
    knn = {}
    train_features = batch_eval(sess, [x], model.net.values(), [X], batch_size)
    print('Fitting knn models on all layers: {}'.format(model.net.keys()))
    for layer_index, layer in enumerate(model.net.keys()):
        if len(train_features[layer_index].shape) == 4:
            train_features[layer_index] = np.asarray(train_features[layer_index], dtype=np.float32).reshape((X.shape[0], -1, train_features[layer_index].shape[-1]))
            train_features[layer_index] = np.mean(train_features[layer_index], axis=1)
        elif len(train_features[layer_index].shape) == 2:
            pass  # leave as is
        else:
            raise AssertionError('Expecting size of 2 or 4 but got {} for {}'.format(len(train_features[layer_index].shape), layer))

        knn[layer] = NearestNeighbors(n_neighbors=X.shape[0], p=2, n_jobs=20, algorithm='brute')
        knn[layer].fit(train_features[layer_index], y)

    del train_features
    return knn


def calc_all_ranks_and_dists(X, subset, knn):
    num_output = len(model.net.keys())
    n_neighbors = knn[knn.keys()[0]].n_neighbors
    all_neighbor_ranks = -1 * np.ones((len(X), num_output, n_neighbors), dtype=np.int32)
    all_neighbor_dists = -1 * np.ones((len(X), num_output, n_neighbors), dtype=np.float32)

    features = batch_eval(sess, [x], model.net.values(), [X], batch_size)
    for layer_index, layer in enumerate(model.net.keys()):
        print('Calculating ranks and distances for subset {} for layer {}'.format(subset, layer))
        if len(features[layer_index].shape) == 4:
            features[layer_index] = np.asarray(features[layer_index], dtype=np.float32).reshape((X.shape[0], -1, features[layer_index].shape[-1]))
            features[layer_index] = np.mean(features[layer_index], axis=1)
        elif len(features[layer_index].shape) == 2:
            pass  # leave as is
        else:
            raise AssertionError('Expecting size of 2 or 4 but got {} for {}'.format(len(features[layer_index].shape), layer))

        all_neighbor_dists[:, layer_index], all_neighbor_ranks[:, layer_index] = \
            knn[layer].kneighbors(features[layer_index], return_distance=True)

    del features
    return all_neighbor_ranks, all_neighbor_dists


def append_suffix(f):
    # if with_noisy:
    #     f = f + '_noisy_{}'.format(FLAGS.with_noise)  # TODO(remove in the future. For backward compatibility)
    if FLAGS.noisy:
        f = f + '_noisy'
    if FLAGS.only_last:
        f = f + '_only_last'
    f = f + '.npy'
    return f


def find_ranks(sub_index, sorted_influence_indices, adversarial=False):

    if adversarial:
        ni = all_adv_ranks
        nd = all_adv_dists
    else:
        ni = all_normal_ranks
        nd = all_normal_dists

    num_output = len(model.net)
    ranks = -1 * np.ones((num_output, len(sorted_influence_indices)), dtype=np.int32)
    dists = -1 * np.ones((num_output, len(sorted_influence_indices)), dtype=np.float32)

    print('Finding ranks for sub_index={} (adversarial={})'.format(sub_index, adversarial))
    for target_idx in range(len(sorted_influence_indices)):  # for only some indices (say, 0:50 only)
        idx = sorted_influence_indices[target_idx]  # selecting training sample index
        for layer_index in range(num_output):
            loc_in_knn = np.where(ni[sub_index, layer_index] == idx)[0][0]
            knn_dist   = nd[sub_index, layer_index, loc_in_knn]
            ranks[layer_index, target_idx] = loc_in_knn
            dists[layer_index, target_idx] = knn_dist

    ranks_mean = np.mean(ranks, axis=1)
    dists_mean = np.mean(dists, axis=1)

    return ranks_mean, dists_mean


def get_nnif(X, subset, max_indices):
    """Returns the knn rank of every testing sample"""
    if subset == 'val':
        inds_correct = val_inds_correct
        y_sparse     = y_val_sparse
        x_preds      = x_val_preds
        x_preds_adv  = x_val_preds_adv
    else:
        inds_correct = test_inds_correct
        y_sparse     = y_test_sparse
        x_preds      = x_test_preds
        x_preds_adv  = x_test_preds_adv
    inds_correct = feeder.get_global_index(subset, inds_correct)

    # initialize knn for layers
    num_output = len(model.net)

    ranks     = -1 * np.ones((len(X), num_output, 4))
    ranks_adv = -1 * np.ones((len(X), num_output, 4))

    for i in tqdm(range(len(inds_correct))):
        global_index = inds_correct[i]
        real_label = y_sparse[i]
        pred_label = x_preds[i]
        adv_label  = x_preds_adv[i]
        assert pred_label == real_label, 'failed for i={}, global_index={}'.format(i, global_index)
        index_dir = os.path.join(model_dir, subset, '{}_index_{}'.format(subset, global_index))

        # collect pred scores:
        scores = np.load(os.path.join(index_dir, 'real', 'scores.npy'))
        sorted_indices = np.argsort(scores)
        ranks[i, :, 0], ranks[i, :, 1] = find_ranks(i, sorted_indices[-max_indices:][::-1], adversarial=False)
        ranks[i, :, 2], ranks[i, :, 3] = find_ranks(i, sorted_indices[:max_indices], adversarial=False)

        # collect adv scores:
        scores = np.load(os.path.join(index_dir, 'adv', FLAGS.attack, 'scores.npy'))
        sorted_indices = np.argsort(scores)
        ranks_adv[i, :, 0], ranks_adv[i, :, 1] = find_ranks(i, sorted_indices[-max_indices:][::-1], adversarial=True)
        ranks_adv[i, :, 2], ranks_adv[i, :, 3] = find_ranks(i, sorted_indices[:max_indices], adversarial=True)

    print("{} ranks_normal: ".format(subset), ranks.shape)
    print("{} ranks_adv: ".format(subset), ranks_adv.shape)
    assert (ranks     != -1).all()
    assert (ranks_adv != -1).all()

    return ranks, ranks_adv



