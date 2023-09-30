#!/usr/bin/env python3
""" Detect Adversarials

author Peter Lorenz
"""
print('Load modules...')
import numpy as np
import pickle
import torch
import sys, os
import pdb

import argparse

from conf import settings
from cfg import * 
from utils import (
    Logger,
    log_header,
    create_dir_extracted_characteristics,
    create_dir_detection,
    save_args_to_file,
    create_save_dir_path,
    print_args,
    args_handling
)

from detection.helper_detection import (
    show_results, 
    split_data, 
    save_load_clf, 
    compute_time_sample
)

from attack.helper_attacks import check_args_attack

#processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_nr",         default=1,      type=int, help="Which run should be taken?")

parser.add_argument("--attack",            default='fgsm',           help=settings.HELP_ATTACK)
parser.add_argument("--detector",          default='InputMFS',       help=settings.HELP_DETECTOR)
parser.add_argument("--net",               default='cif10',          help=settings.HELP_NET)
parser.add_argument("--mode",              default='test',           help="Choose test or validation case")
parser.add_argument("--nr",                default='-1',   type=int, help=settings.HELP_LAYER_NR)
parser.add_argument("--wanted_samples",    default='1500', type=int, help=settings.HELP_WANTED_SAMPLES)
parser.add_argument("--num_classes",       default='10',   type=int, help=settings.HELP_NUM_CLASSES)
parser.add_argument("--clf",               default='LR',             help="Logistic Regression (LR) or Random Forest (RF) or Isolation Forest (IF)")
parser.add_argument("--trees",             default='300',  type=int, help=settings.HELP_NUM_CLASSES)
parser.add_argument("--num_iter",          default='100',  type=int, help="LR: Number iteration")
parser.add_argument("--kernel",            default='rbf',  type=str, help="SVC: rbf, poly, linear, sigmoid, precomputed")
parser.add_argument("--pca_features",      default='0',    type=int, help="Number of PCA features to train")
parser.add_argument("--fixed_clean_data",  action='store_true',      help="Fixed Clean Data")
parser.add_argument("--lid_k_log",         action='store_true',      help="LID expanded")
parser.add_argument("--tuning", default=None,  help="randomsearch, gridsearch")

parser.add_argument('--version',    type=str, default='standard')
# parser.add_argument("--eps",   default='-1',     help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")

parser.add_argument("--eps",    default='8./255.',     help=settings.HELP_AA_EPSILONS)
# parser.add_argument("--eps",    default='4./255.',   help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps",    default='2./255.',   help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps",    default='1./255.',   help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps",    default='1./255.',   help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps",    default='0.5/255.',  help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")

parser.add_argument("--k_lid",    default='-1',  type=int,     help="k for LID")

parser.add_argument("--load_json", default="", help="Load settings from file in json format. Command line options override values in file.")
parser.add_argument("--save_json", default="", help="Save settings to file in json format. Ignored in json file")

args = parser.parse_args()
args = args_handling(args, parser, cfg_detect_path)
print_args(args)

# check args
args = check_args_attack(args, version=True, net_normalization=False, img_size=False)

# output data
output_path_dir = create_dir_detection(args, root='./data/detection/')
save_args_to_file(args, output_path_dir)
logger = Logger(output_path_dir + os.sep + 'log.txt')
log_header(logger, args, output_path_dir, sys)

# load characteristics
logger.log('INFO: Loading characteristics...')

# input data
extracted_characteristics_path = create_dir_extracted_characteristics(args, root='./data/extracted_characteristics/', wait_input=False)

filename = 'characteristics'
if args.lid_k_log:
    filename = 'lid_tmp_k'

characteristics_path, characteristics_advs_path = create_save_dir_path(extracted_characteristics_path, args, filename=filename)


logger.log("characteristics_path:      " + str(characteristics_path) )
# logger.log("characteristics_advs_path: " + str(characteristics_path) )

# import pdb; pdb.set_trace()

s = 1
if args.detector in ['LIDNOISE', 'LIDLESSLayersFeatures'] and not args.fixed_clean_data:
    s = 2
    #characteristics     = torch.load(characteristics_path)[:args.wanted_samples * s]
    characteristics     = torch.load(characteristics_path)[:args.wanted_samples]
    characteristics_adv = torch.load(characteristics_advs_path)[:args.wanted_samples]
    
    if args.lid_k_log:
        characteristics     = characteristics.reshape((characteristics.shape[0], -1))
        characteristics_adv = characteristics_adv.reshape((characteristics_adv.shape[0], -1))
        
if args.fixed_clean_data: 
    characteristics     = torch.load(characteristics_path)
    index = np.random.choice(characteristics.shape[0], args.wanted_samples, replace=False)  
    characteristics = characteristics[index]
    characteristics_adv = torch.load(characteristics_advs_path)[:args.wanted_samples]
else:
    characteristics     = torch.load(characteristics_path)[:args.wanted_samples]
    characteristics_adv = torch.load(characteristics_advs_path)[:args.wanted_samples]
    
characteristics     = np.asarray(characteristics)
characteristics_adv = np.asarray(characteristics_adv)

if len(characteristics.shape) > 2:
    characteristics = characteristics.reshape(characteristics.shape[0], -1)
    characteristics_adv = characteristics_adv.reshape(characteristics_adv.shape[0], -1)
    
shape = np.shape(characteristics)
logger.log("shape: " + str(shape))

if shape[0] < args.wanted_samples:
    logger.log("CAUTION: The actual number is smaller as the wanted samples!")

#if args.detector in ['LIDNOISE', 'LIDLESSLayersFeatures'] and not args.lid_k_log:
#    X_train, y_train, X_test, y_test = split_data(args, logger, characteristics, characteristics_adv, noise=True, test_size=0.1, random_state=42)
#else:
#    X_train, y_train, X_test, y_test = split_data(args, logger, characteristics, characteristics_adv, noise=False, test_size=0.1, random_state=42)
X_train, y_train, X_test, y_test = split_data(args, logger, characteristics, characteristics_adv, noise=False, test_size=0.2, random_state=42)  
# X_train, y_train, X_test, y_test = split_data(args, logger, characteristics, characteristics_adv[:2000], k=shape[0], test_size=0.2, random_state=42)
# scaler  = MinMaxScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test  = scaler.transform(X_test)

if args.pca_features > 0:
    logger.log('Apply PCA decomposition. Reducing number of features from {} to {}'.format(X_train.shape[1], args.pca_features))
    from sklearn.decomposition import PCA # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html?highlight=pca#sklearn.decomposition.PCA
    pca = PCA(n_components=args.pca_features, svd_solver='auto', random_state=32)

    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test  = pca.transform(X_test)
    # # X_train = torch.from_numpy(X_train)
    # from submodules.PyTorch.TorchPCA import PCA
    # y = PCA.Decomposition(X_train.cuda(), k=1)

#train classifier
logger.log('Training classifier...')

if settings.SAVE_CLASSIFIER:
    if args.clf == 'LR':
        from detection.LogisticRegression import LR
        clf, y_hat, y_hat_pr = LR(args, logger, X_train, y_train, X_test, y_test)
    elif args.clf == 'RF':
        from detection.RandomForest import RF
        clf, y_hat, y_hat_pr = RF(args, logger, X_train, y_train, X_test, y_test)
    elif args.clf == 'IF':
        from detection.IsolationForest import IF
        clf, y_hat, y_hat_pr = IF(args, logger, X_train, y_train, X_test, y_test)
    elif args.clf == 'SVC':
        from detection.SVC import SVC
        clf, y_hat, y_hat_pr = SVC(args, logger, X_train, y_train, X_test, y_test)
    elif args.clf == 'cuSVC':
        from detection.SVC import cuSVC
        clf, y_hat, y_hat_pr = cuSVC(args, logger, X_train, y_train, X_test, y_test)
    clf = save_load_clf(args, clf,   output_path_dir=output_path_dir)
else: # load clf
    clf = save_load_clf(args, clf=0, output_path_dir=output_path_dir)

show_results(args, logger, y_test, y_hat, y_hat_pr)