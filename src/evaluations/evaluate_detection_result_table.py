#!/usr/bin/env python3
# """ Evaluate Detection

# author Peter Lorenz
# """

import os
import glob
import pdb 
import shutil
import tqdm
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.append('./')


from conf import settings


from helper_evaluation import (
    extract_information,
    copy_var
)


# SETTINGS = {}


class SETTINGS:
    def __init__(self):
        self.TRTE = False
        self.TRAINERR = False
        # SELECTED_COL = ['asr', 'auc', 'fnr' , 'asrd']
        # SELECTED_COL = ['asr',   'auc',  'f1',  'acc', 'pre', 'tpr', 'fnr', 'asrd']
        self.SELECTED_COL = ['asr', 'auc',  'f1',  'acc', 'pre', 'tpr', 'tnr', 'fnr', 'asrd']

        self.ATTACKS_LIST = [ 'fgsm', 'bim', 'pgd', 'std', 'df', 'cw']
        # ATTACKS_LIST = [ 'apgd-cel2' ]

        self.DETECTOR_LIST_LAYERS = ['InputMFS', 'LayerMFS', 'LID', 'Mahalanobis']

        # self.DETECTOR_LIST = ['InputPFS', 'InputMFS', 'LayerPFS', 'LayerMFS', 'LID', 'Mahalanobis']
        # self.DETECTOR_LIST = ['LID', 'LIDNOISE']
        # DETECTOR_LIST = ['LID']
        # DETECTOR_LIST = ['HPF']
        self.DETECTOR_LIST = ['InputMFS', 'LayerMFS', 'LID', 'Mahalanobis']
        
        self.CLF = ['LR', 'RF']


if __name__ == "__main__":

    settings = SETTINGS()
    
    OUT_PATH = "analysis/variance/run_"
    # OUT_PATH = "analysis/variance/run_gauss_"
    # APP = '_LID_ATTACKTRANSFER'
    # APP = '_LID'
    
    # APP = '_LIDNOISE'
    # APP = '_LIDNOISE_ATTACKTRANSFER'
    
    # APP = '_LIDFeatures'
    # APP = '_apgd-ce'
    # APP = '_apgd-cel2'
    # APP = '_apgd-ce_multilid'
    # APP = '_apgd-cel2'
    APP = ''
    

    # LAYERS=True
    LAYERS=False
    CSV_FILE_PATH = []
    NR = [1]
    
    
    

    for nr in NR:
        CSV_FILE_PATH.append( extract_information(settings, root='./data', net=['cif10'],      dest='./data/detection',   nr=nr, csv_filename='cif10{}.csv'.format(APP), layers=LAYERS ) )
        # CSV_FILE_PATH.append( extract_information(root='./data', net=['cif100'],      dest='./data/detection',   nr=nr, csv_filename='cif100{}.csv'.format(APP), layers=LAYERS) )        # CSV_FILE_PATH.append( extract_information(root='./data', net=['cif10vgg'],      dest='./data/detection',  nr=nr, csv_filename='cif10vgg{}.csv'.format(APP), layers=LAYERS) )
        # CSV_FILE_PATH.append( extract_information(root='./data', net=['cif10vgg'],    dest='./data/detection',   nr=nr, csv_filename='cif10vgg{}.csv'.format(APP), layers=LAYERS) )
        # CSV_FILE_PATH.append( extract_information(root='./data', net=['cif100vgg'],   dest='./data/detection',   nr=nr, csv_filename='cif100vgg{}.csv'.format(APP), layers=LAYERS) )
        # CSV_FILE_PATH.append( extract_information(root='./data', net=['imagenet'],    dest='./data/detection',   nr=nr, csv_filename='imagenet{}.csv'.format(APP), layers=LAYERS) )



        # CSV_FILE_PATH.append( extract_information(root='./data', net=['cif10rn34'],   dest='./data/detection', nr=nr, csv_filename='cif10rn34{}.csv'.format(APP), layers=LAYERS) )
        # CSV_FILE_PATH.append( extract_information(root='./data', net=['cif100rn34'],  dest='./data/detection', nr=nr, csv_filename='cif100rn34{}.csv'.format(APP), layers=LAYERS) )

        # CSV_FILE_PATH.append( extract_information(root='./data', net=['imagenet32'],  dest='./data/detection', nr=nr, csv_filename='imagenet32{}.csv'.format(APP), layers=LAYERS, architecture='wrn_28_10') )
        # CSV_FILE_PATH.append( extract_information(root='./data', net=['imagenet64'],  dest='./data/detection', nr=nr, csv_filename='imagenet64{}.csv'.format(APP), layers=LAYERS) )
        # CSV_FILE_PATH.append( extract_information(root='./data', net=['imagenet128'], dest='./data/detection',nr=nr, csv_filename='imagenet128{}.csv'.format(APP), layers=LAYERS) )

        # CSV_FILE_PATH.append( extract_information(root='./data', net=['celebaHQ32'],  dest='./data/detection', nr=nr, csv_filename='celebaHQ32{}.csv'.format(APP), layers=LAYERS) )
        # CSV_FILE_PATH.append( extract_information(root='./data', net=['celebaHQ64'],  dest='./data/detection', nr=nr, csv_filename='celebaHQ64{}.csv'.format(APP), layers=LAYERS) )
        # CSV_FILE_PATH.append( extract_information(root='./data', net=['celebaHQ128'], dest='./data/detection', nr=nr, csv_filename='celebaHQ128{}.csv'.format(APP), layers=LAYERS) )

        # CSV_FILE_PATH.append( extract_information(root='./data', net=['cif10_rb'],    dest='./data/detection',   nr=nr, csv_filename='cif10_rb{}.csv'.format(APP), layers=LAYERS) )
        
        # architecture='wrn_28_10_25'
        # CSV_FILE_PATH.append( extract_information(root='./data', net=['imagenet32'],  dest='./data/detection', nr=nr, csv_filename='imagenet32{}.csv'.format(architecture), layers=LAYERS, architecture=architecture) )
        # architecture='wrn_28_10_50'
        # CSV_FILE_PATH.append( extract_information(root='./data', net=['imagenet32'],  dest='./data/detection', nr=nr, csv_filename='imagenet32{}.csv'.format(architecture), layers=LAYERS, architecture=architecture) )
        # architecture='wrn_28_10_75'
        # CSV_FILE_PATH.append( extract_information(root='./data', net=['imagenet32'],  dest='./data/detection', nr=nr, csv_filename='imagenet32{}.csv'.format(architecture), layers=LAYERS, architecture=architecture) )
        # architecture='wrn_28_10_100'
        # CSV_FILE_PATH.append( extract_information(root='./data', net=['imagenet32'],  dest='./data/detection', nr=nr, csv_filename='imagenet32{}.csv'.format(architecture), layers=LAYERS, architecture=architecture) )
        # architecture='wrn_28_10_250'
        # CSV_FILE_PATH.append( extract_information(root='./data', net=['imagenet32'],  dest='./data/detection', nr=nr, csv_filename='imagenet32{}.csv'.format(architecture), layers=LAYERS, architecture=architecture) )

        
        # attack_transfer
        # import pdb; pdb.set_trace()
        # CSV_FILE_PATH.append(  extract_information( root='./data', net=['cif10'],     dest='./data/attack_transfer', nr=nr, csv_filename='attack_transfer_lid_cf10{}.csv'.format(APP), layers=False, ATTACKS='attacks', DETECTION='attack_transfer' ) )
        # CSV_FILE_PATH.append(  extract_information( root='./data', net=['cif100'],    dest='./data/attack_transfer', nr=nr, csv_filename='attack_transfer_lid_cif100{}.csv'.format(APP), layers=False, ATTACKS='attacks', DETECTION='attack_transfer' ) )
        # CSV_FILE_PATH.append(  extract_information( root='./data', net=['cif10vgg'],  dest='./data/attack_transfer', nr=nr, csv_filename='attack_transfer_lid_cif10vgg{}.csv'.format(APP), layers=False, ATTACKS='attacks', DETECTION='attack_transfer' ) )
        # CSV_FILE_PATH.append(  extract_information( root='./data', net=['cif100vgg'], dest='./data/attack_transfer', nr=nr, csv_filename='attack_transfer_lid_cif100vgg{}.csv'.format(APP), layers=False, ATTACKS='attacks', DETECTION='attack_transfer' ) )
        # CSV_FILE_PATH.append(  extract_information( root='./data', net=['imagenet'],  dest='./data/attack_transfer', nr=nr, csv_filename='attack_transfer_lid_imagenet{}.csv'.format(APP), layers=False, ATTACKS='attacks', DETECTION='attack_transfer' ) )
        copy_var(CSV_FILE_PATH, OUT_PATH, nr)

    # attack_transfer
    # extract_information(root='./data', net=['cif10', 'imagenet32', 'imagenet64', 'imagenet128', 'celebaHQ32', 'celebaHQ64', 'celebaHQ128'], dest='./data/attack_transfer', run_nr=[1], csv_filename='attack_transfer.csv', layers=False, ATTACKS='attacks', DETECTION='attack_transfer')
    # extract_information(root='./data', net=['imagenet32'], dest='./data/attack_transfer', run_nr=[1], csv_filename='attack_transfer.csv', layers=False, ATTACKS='attacks', DETECTION='attack_transfer')
    # extract_information(root='./data', net=['celebaHQ32'], dest='./data/attack_transfer', run_nr=[1], csv_filename='attack_transfer.csv', layers=False, ATTACKS='attacks', DETECTION='attack_transfer')

    # extract_information(root='./data', net=['imagenet64'], dest='./data/attack_transfer', run_nr=[1], csv_filename='attack_transfer.csv', layers=False, ATTACKS='attacks', DETECTION='attack_transfer')
    # extract_information(root='./data', net=['celebaHQ64'], dest='./data/attack_transfer', run_nr=[1], csv_filename='attack_transfer.csv', layers=False, ATTACKS='attacks', DETECTION='attack_transfer')

    # extract_information(root='./data', net=['imagenet128'], dest='./data/attack_transfer', run_nr=[1], csv_filename='attack_transfer.csv', layers=False, ATTACKS='attacks', DETECTION='attack_transfer')
    # extract_information(root='./data', net=['celebaHQ128'], dest='./data/attack_transfer', run_nr=[1], csv_filename='attack_transfer.csv', layers=False, ATTACKS='attacks', DETECTION='attack_transfer')

    # extract_information(root='./data', net=['imagenet32'], dest='./data/attack_transfer', run_nr=[1], csv_filename='attack_transfer.csv', layers=False, ATTACKS='attacks', DETECTION='attack_transfer')
    # extract_information(root='./data', net=['celebaHQ32'], dest='./data/attack_transfer', run_nr=[1], csv_filename='attack_transfer.csv', layers=False, ATTACKS='attacks', DETECTION='attack_transfer')

    # extract_information( root='./data', net=['cif10', 'cif100', 'cif10vgg', 'cif100vgg', 'imagenet'], dest='./data/attack_transfer', nr=1, csv_filename='attack_transfer_lid.csv', layers=False, ATTACKS='attacks', DETECTION='attack_transfer' )

    # extract_information( root='./data', net=['cif10'], dest='./data/attack_transfer', nr=1, csv_filename='attack_transfer_lid.csv', layers=False, ATTACKS='attacks', DETECTION='attack_transfer' )
    # extract_information( root='./data', net=['cif100'], dest='./data/attack_transfer', nr=1, csv_filename='attack_transfer_lid_cif100.csv', layers=False, ATTACKS='attacks', DETECTION='attack_transfer' )

    # extract_information( root='./data', net=['cif10vgg' ], dest='./data/attack_transfer', nr=1, csv_filename='attack_transfer_lid_cif10vgg.csv', layers=False, ATTACKS='attacks', DETECTION='attack_transfer' )
    # extract_information( root='./data', net=['cif100vgg'], dest='./data/attack_transfer', nr=1, csv_filename='attack_transfer_lid_cif100vgg.csv', layers=False, ATTACKS='attacks', DETECTION='attack_transfer' )

    # extract_information( root='./data', net=['imagenet'], dest='./data/attack_transfer', nr=3, csv_filename='attack_transfer_lid_imagenet.csv', layers=False, ATTACKS='attacks', DETECTION='attack_transfer' )


    # data_transfer

    # extract_information(root='./data', net=['cif10'],       dest='./data/data_transfer', run_nr=[1],  csv_filename='data_transfer_cif10.csv',      layers=False,   ATTACKS='attacks', DETECTION='data_transfer')
    # extract_information(root='./data', net=['imagenet32'],  dest='./data/data_transfer', run_nr=[1],  csv_filename='data_transfer_imagenet32.csv', layers=False,   ATTACKS='attacks', DETECTION='data_transfer')
    # extract_information(root='./data', net=['celebaHQ32'],  dest='./data/data_transfer', run_nr=[1],  csv_filename='data_transfer_celebaHQ32.csv', layers=False,   ATTACKS='attacks', DETECTION='data_transfer')
    # extract_information(root='./data', net=['imagenet64'],  dest='./data/data_transfer', run_nr=[1],  csv_filename='data_transfer_imagenet64.csv', layers=False,   ATTACKS='attacks', DETECTION='data_transfer')
    # extract_information(root='./data', net=['celebaHQ64'],  dest='./data/data_transfer', run_nr=[1],  csv_filename='data_transfer_celebaHQ64.csv', layers=False,   ATTACKS='attacks', DETECTION='data_transfer')
    # extract_information(root='./data', net=['imagenet128'], dest='./data/data_transfer', run_nr=[1],  csv_filename='data_transfer_imagenet128.csv', layers=False,  ATTACKS='attacks', DETECTION='data_transfer')
    # extract_information(root='./data', net=['celebaHQ128'], dest='./data/data_transfer', run_nr=[1],  csv_filename='data_transfer_celebaHQ128.csv', layers=False,  ATTACKS='attacks', DETECTION='data_transfer')


    # extract_information(root='./data', net=['cif10'],       dest='./data/data_transfer', nr=1,  csv_filename='data_transfer_cif10_lid.csv',      layers=False,   ATTACKS='attacks', DETECTION='data_transfer')
    # extract_information(root='./data', net=['cif10vgg'],       dest='./data/data_transfer', nr=1,  csv_filename='data_transfer_cif10vgg_lid.csv',      layers=False,   ATTACKS='attacks', DETECTION='data_transfer')
    



# run 7 and 8

# extract_information(root='./data', net=['cif10'], dest='./data/detection',     run_nr=[7], csv_filename='eval.csv', layers=True)
# extract_information(root='./data', net=['cif10vgg'], dest='./data/detection',  run_nr=[7], csv_filename='eval.csv', layers=True)

# extract_information(root='./data', net=['cif10'], dest='./data/detection',     run_nr=[8], csv_filename='eval.csv', layers=True)
# extract_information(root='./data', net=['cif10'], dest='./data/detection',     run_nr=[8], csv_filename='eval.csv', layers=False)

# extract_information(root='./data', net=['cif10vgg'], dest='./data/detection',  run_nr=[8], csv_filename='eval.csv', layers=True)
# extract_information(root='./data', net=['cif10vgg'], dest='./data/detection',  run_nr=[8], csv_filename='eval.csv', layers=False)


# extract_information(root='./data', net=['cif10rn34'], dest='./data/detection',  run_nr=[8], csv_filename='eval.csv', layers=True)

# extract_information(root='./data', net=['cif10_rb'], dest='./data/detection',  run_nr=[7], csv_filename='eval.csv', layers=True)
# extract_information(root='./data', net=['cif10_rb'], dest='./data/detection',  run_nr=[8], csv_filename='eval.csv', layers=True)

# extract_information(root='./data', net=['imagenet'], dest='./data/detection',  run_nr=[7], csv_filename='eval.csv', layers=True)
# extract_information(root='./data', net=['imagenet'], dest='./data/detection',  run_nr=[8], csv_filename='eval.csv', layers=False)

# extract_information(root='./data', net=['imagenet32'], dest='./data/detection',  run_nr=[8], csv_filename='eval_layers.csv', layers=True)

