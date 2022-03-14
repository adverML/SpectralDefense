#!/usr/bin/env python3

# cif10 4 titan because of extract char --> 4x45GB RAM

print('Load modules...')
import numpy as np
import pdb
import os, sys
import pickle

import torch
# import torchvision
# import torchvision.models as models_v

from models.vgg_cif10 import VGG
from models.wideresidual import WideResNet, WideBasic
from models.orig_resnet import wide_resnet50_2

import argparse

from conf import settings
from utils import (
    Logger,
    log_header,
    create_dir_extracted_characteristics, 
    save_args_to_file,
    check_args,
    create_dir_attacks,
    create_save_dir_path,
    get_num_classes,
    load_model
)

from defenses.helper_layer_extr import get_whitebox_features, test_activation, dfknn_layer
from defenses.Spectral import blackbox_mfs_analysis, blackbox_mfs_pfs, whitebox_mfs_pfs


def extract_features(logger, args, model, input_path_dir, output_path_dir, wanted_samples, option=1):
    """
    docstring
    """
    if option == 0:
        indicator = '_tr'
    elif option == 1:
        indicator = '_te'
    elif optino == 2:
        indicator = ''

    images_path, images_advs_path = create_save_dir_path(input_path_dir, args, filename='images' + indicator)

    logger.log("INFO: images_path " + images_path)
    logger.log("INFO: images_advs " + images_advs_path)

    images =      torch.load(images_path)[:wanted_samples]
    images_advs = torch.load(images_advs_path)[:wanted_samples]

    number_images = len(images)
    logger.log("INFO: eps " + str(args.eps) + " INFO: nr_img " + str(number_images) + " INFO: Wanted Samples: " + str(wanted_samples) )

    if args.detector == 'LayerMFS' or args.detector == 'LayerPFS' or args.detector == 'LID'  or args.detector == 'Mahalanobis':
        get_layer_feature_maps, layers, model, activation = get_whitebox_features(args, logger, model)
    elif args.detector == 'DkNN':
        layers = dfknn_layer(args, model)

    ################Sections for each different detector
    ####### Fourier section
    if args.detector == 'InputMFSAnalysis':
        characteristics, characteristics_adv = blackbox_mfs_analysis(args, images, images_advs)

    elif args.detector == 'InputMFS':
        characteristics, characteristics_adv = blackbox_mfs_pfs(args, images, images_advs, typ='MFS')

    elif args.detector == 'InputPFS':
        characteristics, characteristics_adv = blackbox_mfs_pfs(args, images, images_advs, typ='PFS')

    elif args.detector == 'LayerMFS': 
        characteristics, characteristics_adv = whitebox_mfs_pfs(args, model, images, images_advs, layers, get_layer_feature_maps, activation, typ='MFS')

    elif args.detector == 'LayerPFS':
        characteristics, characteristics_adv = whitebox_mfs_pfs(args, model, images, images_advs, layers, get_layer_feature_maps, activation, typ='PFS')

    ####### LID section
    elif args.detector == 'LID':
        from defenses.Lid import lid
        characteristics, characteristics_adv = lid(args, model, images, images_advs, layers, get_layer_feature_maps, activation)

    ####### Mahalanobis section
    elif args.detector == 'Mahalanobis':
        from defenses.DeepMahalanobis import deep_mahalanobis
        characteristics, characteristics_adv = deep_mahalanobis(args, logger, model, images, images_advs, layers, get_layer_feature_maps, activation, output_path_dir)

    ####### Dknn section
    elif args.detector == 'DkNN':
        import defenses.DeepkNN as DkNN
        characteristics, characteristics_adv = DkNN.calculate(args, model, images, images_advs, layers, 0, 0)
        
    ####### Trust section
    elif args.detector == 'Trust':
        pass

    ####### LID_Class_Cond section
    elif args.detector == 'LID_Class_Cond':
        pass

    ####### ODD section https://github.com/jayaram-r/adversarial-detection
    elif args.detector == 'ODD':
        pass
    else:
        logger.log('ERR: unknown detector')

    return characteristics, characteristics_adv


if __name__ == '__main__':
    #processing the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_nr",          default=1,              type=int, help="Which run should be taken?")

    parser.add_argument("--attack"  ,        default='fgsm',          help=settings.HELP_ATTACK)
    parser.add_argument("--detector",        default='LayerMFS',      help=settings.HELP_DETECTOR)
    parser.add_argument('--take_inputimage_off', action='store_false',    help='Input Images for feature extraction. Default = True')
    parser.add_argument("--max_freq_on",     action='store_true',      help="Switch max frequency normalization on")

    parser.add_argument("--net",            default='cif10',          help=settings.HELP_NET)
    parser.add_argument("--nr",             default='-1',   type=int, help=settings.HELP_LAYER_NR)
    parser.add_argument("--wanted_samples", default='0', type=int, help=settings.HELP_WANTED_SAMPLES)
    parser.add_argument("--wanted_samples_tr", default='1000', type=int, help=settings.HELP_WANTED_SAMPLES)
    parser.add_argument("--wanted_samples_te", default='1000', type=int, help=settings.HELP_WANTED_SAMPLES)

    parser.add_argument('--img_size',       default='32',   type=int, help=settings.HELP_IMG_SIZE)
    parser.add_argument("--num_classes",    default='10',   type=int, help=settings.HELP_NUM_CLASSES)

    parser.add_argument("--shuffle_on",     action='store_true',      help="Switch shuffle data on")
    parser.add_argument('--net_normalization', action='store_true',   help=settings.HELP_NET_NORMALIZATION)

    # parser.add_argument("--eps",       default='-1',       help=settings.HELP_AA_EPSILONS) # to activate the best layers
    parser.add_argument("--eps",       default='8./255.',       help=settings.HELP_AA_EPSILONS)
    # parser.add_argument("--eps",       default='4./255.',       help=settings.HELP_AA_EPSILONS)
    # parser.add_argument("--eps",       default='2./255.',       help=settings.HELP_AA_EPSILONS)
    # parser.add_argument("--eps",       default='1./255.',       help=settings.HELP_AA_EPSILONS)
    # parser.add_argument("--eps",       default='1./255.',       help=settings.HELP_AA_EPSILONS)
    # parser.add_argument("--eps",       default='0.5/255.',       help=settings.HELP_AA_EPSILONS)

    # Frequency Analysis
    parser.add_argument("--fr", default='8',  type=int, help="InputMFS frequency analysis") 
    parser.add_argument("--to", default='24', type=int, help="InputMFS frequency analysis") 

    args = parser.parse_args()

    # max frequency
    # if args.max_freq_on or ((args.net == 'cif100' or args.net == 'cif100vgg' or args.net == 'cif100rn34') and (args.attack=='cw' or args.attack=='df')):
    #     args.max_freq_on = True

    # input data
    input_path_dir = create_dir_attacks(args, root='./data/attacks/')

    # output path dir
    output_path_dir = create_dir_extracted_characteristics(args, root='./data/extracted_characteristics/', wait_input=False)

    save_args_to_file(args, output_path_dir)
    logger = Logger(output_path_dir + os.sep + 'log.txt')
    log_header(logger, args, output_path_dir, sys) # './data/extracted_characteristics/imagenet32/wrn_28_10/std/8_255/LayerMFS'

    # check args
    args = check_args(args, logger)

    #load model
    logger.log('INFO: Loading model...')
    model, _ = load_model(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.eval()

    layer_nr = int(args.nr)
    logger.log("INFO: layer_nr " + str(layer_nr) ) 

    if args.wanted_samples > 0:

        args.wanted_samples_tr = 0
        args.wanted_samples_te = 0

        characteristics_path, characteristics_advs_path = create_save_dir_path(output_path_dir, args, filename='characteristics')
        characteristics, characteristics_adv = extract_features(logger, args, model, input_path_dir, output_path_dir, args.wanted_samples, option=2)

        logger.log('INFO: characteristics:     ' + characteristics_path)
        logger.log('INFO: characteristics_adv: ' + characteristics_advs_path)

        torch.save(characteristics,      characteristics_path, pickle_protocol=4)
        torch.save(characteristics_adv,  characteristics_advs_path, pickle_protocol=4)


    if args.wanted_samples_tr > 0:
        characteristics_path, characteristics_advs_path = create_save_dir_path(output_path_dir, args, filename='characteristics_tr')
        characteristics, characteristics_adv = extract_features(logger, args, model, input_path_dir, output_path_dir, args.wanted_samples_tr, option=0)

        logger.log('INFO: characteristics:     ' + characteristics_path)
        logger.log('INFO: characteristics_adv: ' + characteristics_advs_path)

        torch.save(characteristics,      characteristics_path, pickle_protocol=4)
        torch.save(characteristics_adv,  characteristics_advs_path, pickle_protocol=4)

        characteristics = []; characteristics_adv = []

    if args.wanted_samples_te > 0:

        characteristics_path, characteristics_advs_path = create_save_dir_path(output_path_dir, args, filename='characteristics_te')
        characteristics, characteristics_adv = extract_features(logger, args, model, input_path_dir, output_path_dir, args.wanted_samples_te, option=1)

        logger.log('INFO: characteristics:     ' + characteristics_path)
        logger.log('INFO: characteristics_adv: ' + characteristics_advs_path)

        torch.save(characteristics,      characteristics_path, pickle_protocol=4)
        torch.save(characteristics_adv,  characteristics_advs_path, pickle_protocol=4)

        characteristics = []; characteristics_adv = []



    logger.log('INFO: Done performing extracting features!')
    
