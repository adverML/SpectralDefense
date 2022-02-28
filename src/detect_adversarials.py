#!/usr/bin/env python3
""" Detect Adversarials

author Peter Lorenz
"""
print('Load modules...')
import numpy as np
import pickle
import torch
import sys, os

from conf import settings
from utils import (
    Logger,
    log_header,
    create_dir_extracted_characteristics,
    create_dir_detection,
    save_args_to_file,
    create_save_dir_path,
)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import svm
import argparse

from detection.helper_detection import show_results, split_data, save_load_clf


#processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_nr",         default=1,      type=int, help="Which run should be taken?")

parser.add_argument("--attack",         default='fgsm',           help=settings.HELP_ATTACK)
parser.add_argument("--detector",       default='InputMFS',       help=settings.HELP_DETECTOR)
parser.add_argument("--net",            default='cif10',          help=settings.HELP_NET)
parser.add_argument("--mode",           default='test',           help="Choose test or validation case")
parser.add_argument("--nr",             default='-1',   type=int, help=settings.HELP_LAYER_NR)
parser.add_argument("--wanted_samples", default='1500', type=int, help=settings.HELP_WANTED_SAMPLES)
parser.add_argument("--num_classes",    default='10',   type=int, help=settings.HELP_NUM_CLASSES)
parser.add_argument("--clf",            default='LR',             help="Logistic Regression (LR) or Random Forest (RF) or Isolation Forest (IF)")
parser.add_argument("--trees",          default='300',  type=int, help=settings.HELP_NUM_CLASSES)
parser.add_argument("--num_iter",       default='100',  type=int, help="LR: Number iteration")
parser.add_argument("--pca_features",   default='1',    type=int, help="Number of PCA features to train")


# parser.add_argument("--eps",   default='-1',     help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
parser.add_argument("--eps",    default='8./255.',     help=settings.HELP_AA_EPSILONS)
# parser.add_argument("--eps",    default='4./255.',   help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps",    default='2./255.',   help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps",    default='1./255.',   help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps",    default='1./255.',   help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps",    default='0.5/255.',  help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")

args = parser.parse_args()

# output data
output_path_dir = create_dir_detection(args, root='./data/detection/')
save_args_to_file(args, output_path_dir)
logger = Logger(output_path_dir + os.sep + 'log.txt')
log_header(logger, args, output_path_dir, sys) # './data/extracted_characteristics/imagenet32/wrn_28_10/std/8_255/LayerMFS'

# load characteristics
logger.log('INFO: Loading characteristics...')


# input data
extracted_characteristics_path = create_dir_extracted_characteristics(args, root='./data/extracted_characteristics/', wait_input=False)
characteristics_path, characteristics_advs_path = create_save_dir_path(extracted_characteristics_path, args, filename='characteristics')

logger.log("characteristics_path:      " + str(characteristics_path) )
logger.log("characteristics_advs_path: " + str(characteristics_advs_path) )

characteristics =     torch.load(characteristics_path)[:args.wanted_samples]
characteristics_adv = torch.load(characteristics_advs_path)[:args.wanted_samples]

shape = np.shape(characteristics)
logger.log("shape: " + str(shape))

if shape[0] < args.wanted_samples:
    logger.log("CAUTION: The actual number is smaller as the wanted samples!")


X_train, y_train, X_test, y_test = split_data(args, logger, characteristics, characteristics_adv, k=shape[0], test_size=0.2, random_state=42)

scaler  = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)


if args.pca_features > 0:
    logger.log('Apply PCA decomposition. Reducing number of features from {} to {}'.format(X_train.shape[1], args.pca_features))
    from sklearn.decomposition import PCA # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html?highlight=pca#sklearn.decomposition.PCA
    # pca = PCA(n_components=args.pca_features)
    pca = PCA(n_components='mle', svd_solver='auto', random_state=32)

    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)


#train classifier
logger.log('Training classifier...')

if args.clf == 'LR':
    from detection.LogisticRegression import LR
    clf, y_hat, y_hat_pr = LR(args, logger, X_train, y_train, X_test, y_test)
elif args.clf == 'RF':
    from detection.RandomForest import RF
    clf, y_hat, y_hat_pr = RF(args, logger, X_train, y_train, X_test, y_test)
elif args.clf == 'IF':
    from detection.IsolationForest import IF
    clf, y_hat, y_hat_pr = IF(args, logger, X_train, y_train, X_test, y_test)


clf = save_clf(args, output_path_dir)


show_results(args, logger, y_test, y_hat, y_hat_pr)