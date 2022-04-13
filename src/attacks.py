#!/usr/bin/env python3
""" AutoAttack Foolbox

author Peter Lorenz
"""

print('Load modules...')
import os, sys
import argparse
import pdb
import torch

import numpy as np
from tqdm import tqdm

from conf import settings

from utils import (
    Logger,
    log_header,
    create_dir_extracted_characteristics, 
    save_args_to_file,
    getdevicename,
    create_dir_attacks,
    create_save_dir_path,
    create_dir_clean_data,
    epsilon_to_float,
    get_num_classes,
    load_model,
    get_debug_info
)

from attack.helper_attacks import (
    adapt_batchsize,
    check_args_attack,
    create_advs
)


if __name__ == '__main__':
    # processing the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_nr",            default=1,       type=int, help="Which run should be taken?")
    
    parser.add_argument("--attack",            default='fgsm',            help=settings.HELP_ATTACK)
    parser.add_argument("--net",               default='cif10',           help=settings.HELP_NET)
    parser.add_argument("--img_size",          default='32',    type=int, help=settings.HELP_IMG_SIZE)
    parser.add_argument("--num_classes",       default='10',    type=int, help=settings.HELP_NUM_CLASSES)
    parser.add_argument("--wanted_samples",    default='2000',  type=int, help=settings.HELP_WANTED_SAMPLES)
    parser.add_argument("--all_samples",       default='4000',  type=int, help="Samples from generate Clean data")
    parser.add_argument("--shuffle_on",        action='store_true',       help="Switch shuffle data on")

    # Only for Autoatack
    parser.add_argument('--norm',       type=str, default='Linf')
    parser.add_argument('--eps',        type=str, default='8./255.')
    parser.add_argument('--individual',           action='store_true')
    parser.add_argument('--batch_size', type=int, default=1500)
    parser.add_argument('--log_path',   type=str, default='log.txt')
    parser.add_argument('--version',    type=str, default='standard')

    parser.add_argument('--net_normalization', action='store_false', help=settings.HELP_NET_NORMALIZATION)

    args = parser.parse_args()
    args = check_args_attack(args)
    
    if args.attack == 'apgd-ce' or args.attack == 'apgd-t' or args.attack == 'fab-t' or args.attack == 'square':
        args.individual = True
        args.version = 'custom'
        
    # output data
    output_path_dir = create_dir_attacks(args, root='./data/attacks/')

    save_args_to_file(args, output_path_dir)
    logger = Logger(output_path_dir + os.sep + 'log.txt')
    log_header(logger, args, output_path_dir, sys) # './data/attacks/imagenet32/wrn_28_10/fgsm'

    device_name = getdevicename()

    #load model
    logger.log('INFO: Load model...')
    model, preprocessing = load_model(args)

    model = model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #load correctly classified data
    args.batch_size = adapt_batchsize(args, device_name)
    logger.log('INFO: batch size: ' + str(args.batch_size))

    # input data    
    clean_data_path = create_dir_clean_data(args, root='./data/clean_data/')
    logger.log('INFO: clean data path: ' + clean_data_path)
    

    images, labels, images_advs, labels_advs = create_advs(logger, args, model, output_path_dir, clean_data_path, args.wanted_samples, preprocessing, option=2)

    # create save dir 
    images_path, images_advs_path = create_save_dir_path(output_path_dir, args)
    logger.log('images_path: ' + images_path)

    # pdb.set_trace()
    torch.save(images,      images_path,      pickle_protocol=4)
    torch.save(images_advs, images_advs_path, pickle_protocol=4)

    labels_path, labels_advs_path = create_save_dir_path(output_path_dir, args, filename='labels')
    torch.save(labels,      labels_path,      pickle_protocol=4)
    torch.save(labels_advs, labels_advs_path, pickle_protocol=4)

    logger.log('INFO: Done performing attacks and adversarial examples are saved!')