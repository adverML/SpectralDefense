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

from utils import (
    Logger,
    log_header,
    create_dir_extracted_characteristics, 
    save_args_to_file,
    create_dir_attacks,
    create_save_dir_path,
    get_num_classes,
    load_model
)

from attack.helper_attacks import check_args_attack

from defenses.helper_layer_extr import get_whitebox_features, test_activation, dfknn_layer
from defenses.Spectral import blackbox_mfs_analysis, blackbox_mfs_pfs, whitebox_mfs_pfs

#processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_nr",          default=1,              type=int, help="Which run should be taken?")

parser.add_argument("--attack"  ,        default='fgsm',          help=settings.HELP_ATTACK)
parser.add_argument("--detector",        default='LayerMFS',      help=settings.HELP_DETECTOR)
parser.add_argument('--take_inputimage_off', action='store_false', help='Input Images for feature extraction. Default = True')
parser.add_argument("--max_freq_on",     action='store_true',     help="Switch max frequency normalization on")

parser.add_argument("--net",            default='cif10',          help=settings.HELP_NET)
parser.add_argument("--nr",             default='-1',   type=int, help=settings.HELP_LAYER_NR)
parser.add_argument("--wanted_samples", default='2000', type=int, help=settings.HELP_WANTED_SAMPLES)
parser.add_argument('--img_size',       default='32',   type=int, help=settings.HELP_IMG_SIZE)
parser.add_argument("--num_classes",    default='10',   type=int, help=settings.HELP_NUM_CLASSES)

parser.add_argument("--shuffle_on",        action='store_true',   help="Switch shuffle data on")
parser.add_argument('--net_normalization', action='store_true',   help=settings.HELP_NET_NORMALIZATION)
parser.add_argument("--fixed_clean_data",  action='store_true',       help="Fixed Clean Data")

parser.add_argument('--version',    type=str, default='standard')
# parser.add_argument("--eps",       default='-1',       help=settings.HELP_AA_EPSILONS) # to activate the best layers
parser.add_argument("--eps",       default='8./255.',    help=settings.HELP_AA_EPSILONS)
# parser.add_argument("--eps",       default='4./255.',  help=settings.HELP_AA_EPSILONS)
# parser.add_argument("--eps",       default='2./255.',  help=settings.HELP_AA_EPSILONS)
# parser.add_argument("--eps",       default='1./255.',  help=settings.HELP_AA_EPSILONS)
# parser.add_argument("--eps",       default='1./255.',  help=settings.HELP_AA_EPSILONS)
# parser.add_argument("--eps",       default='0.5/255.', help=settings.HELP_AA_EPSILONS)

# Frequency Analysis
parser.add_argument("--fr", default='8',  type=int, help="InputMFS frequency analysis")
parser.add_argument("--to", default='24', type=int, help="InputMFS frequency analysis")

args = parser.parse_args()
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = model.eval()

layer_nr = int(args.nr)
logger.log("INFO: layer_nr " + str(layer_nr) ) 

if args.detector in ['LayerMFS',  'LayerPFS',  'LID', 'Mahalanobis']:
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
elif args.detector in ['LID']:
    from defenses.Lid import lid
    characteristics, characteristics_adv = lid(args, model, images, images_advs, layers, get_layer_feature_maps, activation)

####### Mahalanobis section
elif args.detector == 'Mahalanobis':
    
    from defenses.DeepMahalanobis import deep_mahalanobis
    characteristics, characteristics_adv = deep_mahalanobis(args, logger, model, images, images_advs, layers, get_layer_feature_maps, activation, output_path_dir)


else:
    logger.log('ERR: unknown detector')
    

# Save
logger.log("INFO: Save extracted characteristics ...")

characteristics_path, characteristics_advs_path = create_save_dir_path(output_path_dir, args, filename='characteristics' )
logger.log('INFO: characteristics:     ' + characteristics_path)
logger.log('INFO: characteristics_adv: ' + characteristics_advs_path)

torch.save(characteristics,      characteristics_path, pickle_protocol=4)
torch.save(characteristics_adv,  characteristics_advs_path, pickle_protocol=4)

logger.log('INFO: Done extracting and saving characteristics!')
