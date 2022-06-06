#!/usr/bin/env python3
""" Attack Transfer

author Peter Lorenz
"""
print('Load modules...')
import numpy as np
import pickle
import torch
import sys 
import os

from conf import settings

from utils import (
    create_dir_detection,
    save_args_to_file,
    log_header,
    create_dir_extracted_characteristics,
    create_save_dir_path,
    Logger
)

import argparse
import copy
import pdb

from detection.helper_detection import (
    show_results, 
    show_results_attack_transfer,
    split_data, 
    save_load_clf, 
    compute_time_sample,
)


# Detect Adversarials
SAVE_CLASSIFIER = False

#processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_nr",         default=1,      type=int, help="Which run should be taken?")

parser.add_argument("--attack",         default='fgsm',           help=settings.HELP_ATTACK)
parser.add_argument("--attack_eval",    default='bim',            help=settings.HELP_ATTACK)

parser.add_argument("--detector",       default='InputMFS',       help=settings.HELP_DETECTOR)
parser.add_argument("--net",            default='cif10',          help=settings.HELP_NET)
parser.add_argument("--mode",           default='test',           help="Choose test or validation case")
parser.add_argument("--nr",             default='-1',   type=int, help=settings.HELP_LAYER_NR)
parser.add_argument("--wanted_samples", default='2000', type=int, help=settings.HELP_WANTED_SAMPLES)
parser.add_argument("--clf",            default='LR',             help="Logistic Regression (LR) or Random Forest (RF)")
parser.add_argument("--num_classes",    default='10',   type=int, help=settings.HELP_NUM_CLASSES)

parser.add_argument("--k_lid",    default='-1',  type=int,     help="k for LID")

# parser.add_argument("--eps",       default='-1',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
parser.add_argument("--eps",       default='8./255.',            help=settings.HELP_AA_EPSILONS)
# parser.add_argument("--eps",       default='4./255.',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps",       default='2./255.',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps",       default='1./255.',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps",       default='1./255.',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps",       default='0.5/255.',      help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")

parser.add_argument("--eps_to",       default='8./255.',            help=settings.HELP_AA_EPSILONS)
# parser.add_argument("--eps_to",       default='4./255.',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps_to",       default='2./255.',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps_to",       default='1./255.',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps_to",       default='1./255.',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps_to",       default='0.5/255.',      help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")

args = parser.parse_args()

# output data
from_trained_clf = create_dir_detection(args, root='./data/detection/', TRANSFER='attack_from')
print("--------------------------")
print(from_trained_clf)

output_path_dir  = create_dir_detection(args, root='./data/attack_transfer/', TRANSFER='attack')
print("--------------------------")
print(output_path_dir)


save_args_to_file(args, output_path_dir)
logger = Logger(output_path_dir + os.sep + 'log.txt')
log_header(logger, args, output_path_dir, sys) # './data/extracted_characteristics/imagenet32/wrn_28_10/std/8_255/LayerMFS'

# import pdb; pdb.set_trace()

# load characteristics
logger.log('INFO: Loading characteristics...')

# input data
extracted_characteristics_path = create_dir_extracted_characteristics(args, root='./data/extracted_characteristics/',  TRANSFER='attack', wait_input=False)
characteristics_path, characteristics_advs_path = create_save_dir_path(extracted_characteristics_path, args, filename='characteristics' )

if args.detector == 'LIDNOISE':
    characteristics_path      = characteristics_path.replace('/characteristics', '/lid_tmp_k')
    characteristics_advs_path = characteristics_advs_path.replace('/characteristics_adv', '/lid_tmp_k_adv')

logger.log("characteristics_path:      " + str(characteristics_path) )
logger.log("characteristics_advs_path: " + str(characteristics_advs_path) )

characteristics     = torch.load(characteristics_path)[:args.wanted_samples]
characteristics_adv = torch.load(characteristics_advs_path)[:args.wanted_samples]

if args.detector == 'LIDNOISE':
    # import pdb; pdb.set_trace()
    characteristics     = characteristics.reshape(     (characteristics.shape[0], -1) )
    characteristics_adv = characteristics_adv.reshape( (characteristics_adv.shape[0], -1) )

shape = np.shape(characteristics)
logger.log("shape: " + str(shape))

if shape[0] < args.wanted_samples:
    logger.log("CAUTION: The actual number is smaller as the wanted samples!")

k = shape[0]

X_train, y_train, X_test, y_test = split_data(args, logger, characteristics, characteristics_adv, noise=False, test_size=0.1, random_state=42)

# train classifier
logger.log('Training classifier...')

# save classifier
classifier_pth = from_trained_clf + os.sep + str(args.clf) + '.clf'
# if SAVE_CLASSIFIER:
#     torch.save(clf, classifier_pth)
# else:
logger.log("load clf: " + classifier_pth)
clf = torch.load(classifier_pth)

logger.log( "train score: " + str( clf.score(X_train, y_train) ) )
logger.log( "test score:  " + str( clf.score(X_test, y_test) )   )

# X_test = np.concatenate((X_train, X_test))
# y_test = np.concatenate((y_train, y_test))

logger.log('Evaluating classifier...')
y_hat =    clf.predict(X_test)
y_hat_pr = clf.predict_proba(X_test)[:, 1]

y_train_pr = clf.predict_proba(X_train)[:, 1]

show_results_attack_transfer(args, logger, y_test, y_hat, y_hat_pr, X_train, y_train, y_train_pr)

# show_results(args, logger, y_test, y_hat, y_hat_pr)