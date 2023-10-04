#!/usr/bin/env python3

print('Load modules...')
import numpy as np
import pdb
import os, sys
import pickle
import torch
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

from torch.autograd import Variable
from scipy.spatial.distance import cdist
import scipy
from tqdm import tqdm
from collections import OrderedDict

from models.vgg_cif10 import VGG
from models.wideresidual import WideResNet, WideBasic
from models.orig_resnet import wide_resnet50_2

import argparse
import sklearn
import sklearn.covariance

from conf import settings
from cfg import * 

from utils import (
    Logger,
    log_header,
    create_dir_extracted_characteristics, 
    save_args_to_file,
    create_dir_attacks,
    create_save_dir_path,
    get_num_classes,
    load_model,
    print_args,
    args_handling
)

from attack.helper_attacks import check_args_attack

from defenses.helper_layer_extr import get_whitebox_features, test_activation, dfknn_layer
from defenses.Spectral import blackbox_mfs_analysis, blackbox_mfs_pfs, whitebox_mfs_pfs

#processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_nr",          default=1,               type=int, help="Which run should be taken?")

parser.add_argument("--attack"  ,        default='fgsm',          help=settings.HELP_ATTACK)
parser.add_argument("--detector",        default='LayerMFS',      help=settings.HELP_DETECTOR)
#parser.add_argument('--take_inputimage_off', action='store_false', help='Input Images for feature extraction. Default = True')
parser.add_argument("--take_inputimage_off", default=True, type=lambda x: x == 'True', help="Input Images for feature extraction. Default = True")

parser.add_argument("--max_freq_on",     action='store_true',     help="Switch max frequency normalization on")

parser.add_argument("--net",            default='cif10',          help=settings.HELP_NET)
parser.add_argument("--nr",             default='-1',   type=int, help=settings.HELP_LAYER_NR)
parser.add_argument("--wanted_samples", default='2000', type=int, help=settings.HELP_WANTED_SAMPLES)
parser.add_argument('--img_size',       default='32',   type=int, help=settings.HELP_IMG_SIZE)
parser.add_argument("--num_classes",    default='10',   type=int, help=settings.HELP_NUM_CLASSES)

#parser.add_argument("--shuffle_on",        action='store_true', help="Switch shuffle data on")
parser.add_argument("--shuffle_on", default='False', type=lambda x: x == 'True', help="Switch shuffle data on")
parser.add_argument("--net_normalization", default=False, type=lambda x: x == 'True', help=settings.HELP_NET_NORMALIZATION)
parser.add_argument("--fixed_clean_data", default=False, type=lambda x: x == 'True', help="Fixed Clean Data")

parser.add_argument('--version',    type=str, default='standard')
# parser.add_argument("--eps",       default='-1',       help=settings.HELP_AA_EPSILONS) # to activate the best layers
parser.add_argument("--eps",       default='8./255.',    help=settings.HELP_AA_EPSILONS)
# parser.add_argument("--eps",       default='4./255.',  help=settings.HELP_AA_EPSILONS)
# parser.add_argument("--eps",       default='2./255.',  help=settings.HELP_AA_EPSILONS)
# parser.add_argument("--eps",       default='1./255.',  help=settings.HELP_AA_EPSILONS)
# parser.add_argument("--eps",       default='1./255.',  help=settings.HELP_AA_EPSILONS)
# parser.add_argument("--eps",       default='0.5/255.', help=settings.HELP_AA_EPSILONS)

parser.add_argument("--k_lid",       default='-1',   type=int, help="k for LID")
parser.add_argument("--k_lid_bs",    default='100',  type=int, help="k for LID")

# Frequency Analysis
parser.add_argument("--fr", default='8',  type=int, help="InputMFS frequency analysis")
parser.add_argument("--to", default='24', type=int, help="InputMFS frequency analysis")

parser.add_argument("--device", default='cpu', help="cpu vs. cuda")

parser.add_argument("--load_json", default="", help="Load settings from file in json format. Command line options override values in file.")
parser.add_argument("--save_json", default="", help="Save settings to file in json format. Ignored in json file")

args = parser.parse_args()
args = args_handling(args, parser, cfg_extract_path)
print_args(args)
args = check_args_attack(args, version=True)

# output path dir
output_path_dir = create_dir_extracted_characteristics(args, root='./data/extracted_characteristics/', wait_input=False)

save_args_to_file(args, output_path_dir)
logger = Logger(output_path_dir + os.sep + 'log.txt')
log_header(logger, args, output_path_dir, sys) 

# input data
input_path_dir = create_dir_attacks(args, root='./data/attacks/')
images_path, images_advs_path = create_save_dir_path(input_path_dir, args)

logger.log("INFO: images_path " + images_path)
logger.log("INFO: images_advs " + images_advs_path)

if args.fixed_clean_data:
    images =      torch.load(images_path)
    images_advs = torch.load(images_advs_path)
else:
    images =      torch.load(images_path)[:args.wanted_samples]
    images_advs = torch.load(images_advs_path)[:args.wanted_samples]

number_images = len(images)
logger.log("INFO: eps " + str(args.eps) + " INFO: nr_img " + str(number_images) + " INFO: Wanted Samples: " + str(args.wanted_samples) )

#load model
logger.log('INFO: Loading model...')
model, _ = load_model(args)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(args.device)
model = model.eval()

layer_nr = int(args.nr)
logger.log("INFO: layer_nr " + str(layer_nr) ) 

if args.detector in ['LayerMFS',  'LayerPFS',  'LID',  'LIDNOISE', 'FFTmultiLIDMFS', 'FFTmultiLIDPFS', 'LIDLessFeatures',  'multiLIDLessFeatures', 'Mahalanobis']:
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
    characteristics, characteristics_adv = whitebox_mfs_pfs(args, logger, model, images, images_advs, layers, get_layer_feature_maps, activation, typ='MFS')

elif args.detector == 'LayerPFS':
    characteristics, characteristics_adv = whitebox_mfs_pfs(args, logger, model, images, images_advs, layers, get_layer_feature_maps, activation, typ='PFS')

####### LID section
elif args.detector in ['LID', 'LIDLessFeatures']:
    from defenses.Lid import lid
    characteristics, characteristics_adv = lid(args, model, images, images_advs, layers, get_layer_feature_maps, activation)

####### multiLID section
elif args.detector in ['multiLID', 'multiLIDLessFeatures', 'FFTmultiLIDMFS', 'FFTmultiLIDPFS']:
    from defenses.Lid import multiLID
    characteristics, characteristics_adv = multilid(args, model, images, images_advs, layers, get_layer_feature_maps, activation)

####### LIDNOISE 
elif args.detector in ['LIDNOISE']:
    from defenses.Lid import lidnoise
    # characteristics,  characteristics_noise, characteristics_adv, lid_tmp_k, lid_tmp_k_noise, lid_tmp_k_adv = lidnoise(args, model, images, images_advs, layers, get_layer_feature_maps, activation)
    lid_tmp_k,  lid_tmp_k_adv, lid_tmp_k_adv, characteristics, lid_tmp_k_noise, characteristics_adv = lidnoise(args, model, images, images_advs, layers, get_layer_feature_maps, activation)
    
    lid_tmp_k_path, lid_tmp_k_advs_path = create_save_dir_path(output_path_dir, args, filename='lid_tmp_k')
    noise_path, _                       = create_save_dir_path(output_path_dir, args, filename='lid_tmp_k_noise')
    characteristics_noise_path, _       = create_save_dir_path(output_path_dir, args, filename='characteristics_noise')
    
    torch.save(lid_tmp_k,        lid_tmp_k_path,       pickle_protocol=4)
    torch.save(lid_tmp_k_noise,  noise_path,           pickle_protocol=4)
    torch.save(lid_tmp_k_adv,    lid_tmp_k_advs_path,  pickle_protocol=4)

####### Mahalanobis section
elif args.detector == 'Mahalanobis':
    
    from defenses.DeepMahalanobis import deep_mahalanobis
    characteristics, characteristics_adv = deep_mahalanobis(args, logger, model, images, images_advs, layers, get_layer_feature_maps, activation, output_path_dir)

else:
    logger.log('ERR: unknown detector')

# Save
logger.log("INFO: Save extracted characteristics ...")

characteristics_path, characteristics_advs_path = create_save_dir_path(output_path_dir, args, filename='characteristics')
logger.log('INFO: characteristics:     ' + characteristics_path)
logger.log('INFO: characteristics_adv: ' + characteristics_advs_path)

torch.save(characteristics,      characteristics_path, pickle_protocol=4)
torch.save(characteristics_adv,  characteristics_advs_path, pickle_protocol=4)

logger.log('INFO: Done extracting and saving characteristics!')
