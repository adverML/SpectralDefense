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

from utils import (
    Logger,
    log_header,
    create_dir_clean_data,
    save_args_to_file,
    load_model,
    get_debug_info,
    load_test_set,
    load_train_set
)



from datasets import smallimagenet


def generate_data_labels(logger, args, loader, wanted_samples, option=2):

    clean_dataset = []
    clean_labels = []
    correct = 0
    total = 0
    i = 0
    acc = ((wanted_samples, 0, 0))

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
            clean_dataset.append([images.cpu(), labels.cpu()])
            # clean_dataset.append(images.cpu().numpy())
            # clean_labels.append(labels.cpu().numpy())

        i = i + 1
        if i % 500 == 0:
            acc  = (wanted_samples, i, 100 * correct / total)
            logger.log('INFO: Accuracy of the network on the %d test images: %d, %d %%' % acc)

        if len(clean_dataset) >= wanted_samples:
            break
    
    if option == 2: 
        logger.log("INFO: initial accuracy: {:.2f}".format(acc[-1]))
    elif option == 1: 
        logger.log("INFO: initial te accuracy: {:.2f}".format(acc[-1]))
    elif option == 0: 
        logger.log("INFO: initial tr accuracy: {:.2f}".format(acc[-1]))
    else:
        logger.log("err: logger not found!")
    logger.log("INFO: output_path_dir: " + output_path_dir + ", len(clean_dataset) " + str(len(clean_dataset)) )
    # print( "len labels ", len(clean_labels))
    
    return clean_dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_nr",            default=1,    type=int, help="Which run should be taken?")

    parser.add_argument("--net",               default='cif10',        help=settings.HELP_NET)
    parser.add_argument("--img_size",          default=32,   type=int, help=settings.HELP_IMG_SIZE)
    parser.add_argument("--num_classes",       default=10,   type=int, help=settings.HELP_NUM_CLASSES)
    parser.add_argument("--batch_size",        default=1   , type=int, help=settings.HELP_BATCH_SIZE)
    parser.add_argument("--wanted_samples",   default=0, type=int, help=settings.HELP_WANTED_SAMPLES)
    parser.add_argument("--wanted_samples_tr", default=1000, type=int, help=settings.HELP_WANTED_SAMPLES)
    parser.add_argument("--wanted_samples_te", default=1000, type=int, help=settings.HELP_WANTED_SAMPLES)
    parser.add_argument("--shuffle_off",       action='store_false',  help="Switch shuffle data off")

    parser.add_argument('--net_normalization', action='store_false', help=settings.HELP_NET_NORMALIZATION)
    
    
    args = parser.parse_args()

    if not args.batch_size == 1:
        get_debug_info(msg='Err: Batch size must be always 1!')
        assert True

    output_path_dir = create_dir_clean_data(args, root='./data/clean_data/')

    save_args_to_file(args, output_path_dir)
    logger = Logger(output_path_dir + os.sep + 'log.txt')
    log_header(logger, args, output_path_dir, sys)

    logger.log('INFO: Load model...')

    model, preprocessing = load_model(args)
    model.cuda()
    model.eval()

    print("preprocessing", preprocessing)
    logger.log('INFO: Load dataset...')

    if args.wanted_samples > 0:

        args.wanted_samples_tr = 0
        args.wanted_samples_te = 0
        
        test_loader  = load_test_set(args, shuffle=args.shuffle_off, preprocessing=None) # Data Normalizations; No Net Normaliztion
        clean_data   = generate_data_labels(logger, args, test_loader, args.wanted_samples_te, option=2)

        torch.save(clean_data,   output_path_dir + os.sep + 'clean_data',   pickle_protocol=4)
        # torch.save(clean_te_labels, output_path_dir + os.sep + 'clean_labels', pickle_protocol=4)

    if args.wanted_samples_tr > 0:
        train_loader   = load_train_set(args, shuffle=args.shuffle_off, clean_data=True, preprocessing=None)
        clean_tr_data  = generate_data_labels(logger, args, train_loader, args.wanted_samples_tr, option=0)

        torch.save(clean_tr_data,   output_path_dir + os.sep + 'clean_tr_data',   pickle_protocol=4)
        # torch.save(clean_tr_labels, output_path_dir + os.sep + 'clean_tr_labels', pickle_protocol=4)

        clean_tr_data = []; #clean_tr_labels = []

    if args.wanted_samples_te > 0:
        test_loader   = load_test_set(args, shuffle=args.shuffle_off, preprocessing=None) # Data Normalizations; No Net Normaliztion
        clean_te_data = generate_data_labels(logger, args, test_loader, args.wanted_samples_te,  option=1)

        torch.save(clean_te_data,   output_path_dir + os.sep + 'clean_te_data',   pickle_protocol=4)
        # torch.save(clean_te_labels, output_path_dir + os.sep + 'clean_te_labels', pickle_protocol=4)

        clean_te_data = []; #clean_te_labels = []
