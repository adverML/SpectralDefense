#!/usr/bin/env python3
""" Generate Clean Data

author Peter Lorenz
"""

#this script extracts the correctly classified images
print('INFO: Load modules...')
import pdb
import os, sys
import json
import argparse
import datetime
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets

from conf import settings
from cfg import * 

from utils import (
    Logger,
    log_header,
    save_args_to_file,
    create_dir_clean_data,
    load_model,
    get_debug_info,
    load_test_set,
    load_train_set,
    print_args,
    args_handling
)

from custom_datasets import smallimagenet

from gen_clean_data.helper_generate_clean_data import (
    check_args_generate_clean_data
)
from gen_clean_data.generate_data_labels import (
    generate_data_labels
)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_nr",            default=1,    type=int, help="Which run should be taken?")

    parser.add_argument("--net",               default='cif10',        help=settings.HELP_NET)
    parser.add_argument("--img_size",          default=32,   type=int, help=settings.HELP_IMG_SIZE)
    parser.add_argument("--num_classes",       default=10,   type=int, help=settings.HELP_NUM_CLASSES)
    parser.add_argument("--batch_size",        default=1   , type=int, help=settings.HELP_BATCH_SIZE)
    parser.add_argument("--wanted_samples",    default=0, type=int, help=settings.HELP_WANTED_SAMPLES)
    parser.add_argument("--wanted_samples_tr", default=1000, type=int, help=settings.HELP_WANTED_SAMPLES)
    parser.add_argument("--wanted_samples_te", default=1000, type=int, help=settings.HELP_WANTED_SAMPLES)
    #parser.add_argument("--shuffle_off",       action='store_false',  help="Switch shuffle data off")
    parser.add_argument("--shuffle",          default=True,  type=lambda x: x == 'True', help="Switch shuffle data off")

    parser.add_argument('--net_normalization', action='store_false', help=settings.HELP_NET_NORMALIZATION)
    
    parser.add_argument("--load_json", default="", help="Load settings from file in json format. Command line options override values in file.")
    parser.add_argument("--save_json", default="", help="Save settings to file in json format. Ignored in json file")
    
    args = parser.parse_args()
    args = args_handling(args, parser, cfg_gen_path)
    print_args(args)

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
        
        test_loader  = load_test_set(args, shuffle=args.shuffle, preprocessing=None) # No Data Normalizations; Net Normaliztion
        clean_data   = generate_data_labels(logger, args, model, test_loader, args.wanted_samples_te, output_path_dir, option=2)

        torch.save(clean_data,   output_path_dir + os.sep + 'clean_data',   pickle_protocol=4)


    if args.wanted_samples_tr > 0:
        train_loader   = load_train_set(args, shuffle=args.shuffle, clean_data=True, preprocessing=None)
        clean_tr_data  = generate_data_labels(logger, args, model, train_loader, args.wanted_samples_tr, output_path_dir, option=0)

        torch.save(clean_tr_data,   output_path_dir + os.sep + 'clean_tr_data',   pickle_protocol=4)


        clean_tr_data = []; #clean_tr_labels = []

    if args.wanted_samples_te > 0:
        test_loader   = load_test_set(args, shuffle=args.shuffle, preprocessing=None) # No Data Normalizations; Net Normaliztion
        clean_te_data = generate_data_labels(logger, args, model, test_loader, args.wanted_samples_te, output_path_dir, option=1)

        torch.save(clean_te_data,   output_path_dir + os.sep + 'clean_te_data',   pickle_protocol=4)


        clean_te_data = []; #clean_te_labels = []
