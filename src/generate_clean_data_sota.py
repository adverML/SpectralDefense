#!/usr/bin/env python3
""" Generate Clean Data

author Peter Lorenz
"""

#this script extracts the correctly classified images
print('INFO: Load modules...')
import pdb
import os, sys
import json
from conf import settings
import argparse
import datetime
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
import numpy as np

from utils import *

from datasets import smallimagenet


def generate_data_labels(logger, args, loader):

    clean_dataset = []
    clean_labels = []
    correct = 0
    total = 0
    i = 0
    acc = ((args.wanted_samples, 0, 0))

    logger.log('INFO: Classify images...')

    for images, labels in loader:
        if i == 0:
            logger.log( "INFO: tensor size: " + str(images.size()) )

        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        correct += (predicted == labels).sum().item()
        if (predicted == labels):
            # clean_dataset.append(data)
            clean_dataset.append(images.cpu().numpy())
            clean_labels.append(labels.cpu().numpy())

        i = i + 1
        if i % 500 == 0:
            acc  = (args.wanted_samples, i, 100 * correct / total)
            logger.log('INFO: Accuracy of the network on the %d test images: %d, %d %%' % acc)

        if len(clean_dataset) >= args.wanted_samples:
            break
    
    logger.log("INFO: initial accuracy: {:.2f}".format(acc[-1]))
    logger.log("INFO: output_path_dir: " + output_path_dir + ", len(clean_dataset) " + str(len(clean_dataset)) )
    
    return clean_dataset, clean_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_nr",         default=1,    type=int, help="Which run should be taken?")

    parser.add_argument("--net",            default='cif10rn34sota',    help=settings.HELP_NET)
    parser.add_argument("--img_size",       default=32,   type=int, help=settings.HELP_IMG_SIZE)
    parser.add_argument("--num_classes",    default=10,   type=int, help=settings.HELP_NUM_CLASSES)
    parser.add_argument("--batch_size",     default=1   , type=int, help=settings.HELP_BATCH_SIZE)
    parser.add_argument("--nf",             default=3   , type=int, help="Number of folds!")
    parser.add_argument("--wanted_samples", default=2000, type=int, help=settings.HELP_WANTED_SAMPLES)
    parser.add_argument("--shuffle_off",     action='store_false',  help="Switch shuffle data off")

    parser.add_argument('--net_normalization', action='store_false', help=settings.HELP_NET_NORMALIZATION)
    
    args = parser.parse_args()


    if not args.batch_size == 1:
        get_debug_info(msg='Err: Batch size must be always 1!')
        assert True

    # output_path_dir = create_dir_clean_data(args, root='./data/clean_data/')
    # logger.log('INFO: Load model...')

    model, preprocessing = load_model(args)
    model.cuda()
    model.eval()

    # logger.log('INFO: Load dataset...')
    train_loader = load_train_set(args, shuffle=True, preprocessing=None)
    test_loader  = load_test_set(args, shuffle=True, preprocessing=None) # Data Normalizations; No Net Normaliztion

    for i in range(1, args.nf + 1):
        output_path_dir = '/home/lorenzp/adversialml/src/src/submodules/adversarial-detection/expts/numpy_data/cifar10/fold_{}/CleanData'.format(i)
        make_dir(output_path_dir)
        save_args_to_file(args, output_path_dir)
        logger = Logger(output_path_dir + os.sep + 'log.txt')
        log_header(logger, args, output_path_dir, sys)

        clean_tr_data, clean_tr_labels = generate_data_labels(logger, args, train_loader)

        np.save( os.path.join(output_path_dir, 'data_tr_clean.npy' ) , np.asarray(clean_tr_data).squeeze()   )
        np.save( os.path.join(output_path_dir, 'labels_tr_clean.npy'), np.asarray(clean_tr_labels).squeeze() )

        print("saved adv. examples generated from the train data for fold:", i)

        del logger


    for i in range(1, args.nf + 1):
        output_path_dir = '/home/lorenzp/adversialml/src/src/submodules/adversarial-detection/expts/numpy_data/cifar10/fold_{}/CleanData'.format(i)
        make_dir(output_path_dir)
        save_args_to_file(args, output_path_dir)
        logger = Logger(output_path_dir + os.sep + 'log.txt')
        # log_header(logger, args, output_path_dir, sys)

        clean_te_data, clean_te_labels = generate_data_labels(logger, args, test_loader)

        np.save( os.path.join(output_path_dir, 'data_te_clean.npy' ) , np.asarray(clean_te_data).squeeze()   )
        np.save( os.path.join(output_path_dir, 'labels_te_clean.npy'), np.asarray(clean_te_labels).squeeze() )

        print("saved adv. examples generated from the test data for fold:", i)

        del logger
