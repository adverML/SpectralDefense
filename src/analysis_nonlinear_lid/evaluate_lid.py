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
import pdb
import torch
# import datetime

# from conf import settings
# import csv
# import argparse


OUT_PATH = "analysis_nonlinear_lid/lid_logs"

TRTE=False
TRAINERR=False
SELECTED_COL= ['asr', 'auc', 'f1', 'acc','pre','tpr', 'fnr', 'asrd']
# ATTACKS_LIST = ['gauss', 'fgsm', 'bim', 'pgd', 'std', 'df', 'cw']
ATTACKS_LIST= ['gauss', 'fgsm', 'bim', 'pgd', 'std', 'df', 'cw']

DETECTOR_LIST_LAYERS= ['LID']
DETECTOR_LIST       = ['LID']
CLF = ['LR']


def check_eps(str_val):
    if '8./255.' in str_val:
        eps = '8_255'
    elif '4./255.' in str_val:
        eps = '4_255'     
    elif '2./255.' in str_val:
        eps = '2_255'
    elif '1./255.' in str_val:
        eps = '1_255'
    elif '0.5/255.' in str_val:
        eps = '05_255'
    elif '1./1.' in str_val:
        eps = '1_1'
    elif '1./10.' in str_val:
        eps = '1_10'
    elif '1./100.' in str_val:
        eps = '1_100'
    elif '1./1000.' in str_val:
        eps = '1_1000'

    return eps


def is_float(value):
  try:
    float(value)
    return True
  except:
    return False


def get_clean_accuracy(paths):
    result = {}
    for path in paths:
        if 'gauss' in path:
            attack_method = 'gauss'        
        elif 'fgsm' in path:
            attack_method = 'fgsm'
        elif 'bim' in path:
            attack_method = 'bim'
        elif 'pgd' in path:
            attack_method = 'pgd'
            if 'apgd-ce' in path:
                attack_method = 'apgd-ce'
            elif 'apgd-t' in path:
                attack_method = 'apgd-t'
        elif 'std' in path:
            attack_method = 'std'
            if '8_255' in path:
                eps = '8_255'
            elif '4_255' in path:
                eps = '4_255'     
            elif '2_255' in path:
                eps = '2_255'
            elif '1_255' in path:
                eps = '1_255'
            elif '05_255' in path:
                eps = '05_255'
            elif '1_1' in path:
                eps = '1_1'
            elif '1_10' in path:
                eps = '1_10'
            elif '1_100' in path:
                eps = '1_100'
            elif '1_1000' in path:
                eps = '1_1000'
        elif 'aa+' in path:
            attack_method = 'aa+' 
        elif 'fab-t+' in path:
            attack_method = 'fab-t+'
        elif 'fab+' in path:
            attack_method = 'fab+'
        elif 'square+' in path:
            attack_method = 'square+'
        elif 'fab-t' in path:
            attack_method = 'fab-t'
        elif 'square' in path:
            attack_method = 'square'
        elif 'df' in path:
            attack_method = 'df'
        elif 'cw' in path:
            attack_method = 'cw'
        else:
            raise NotImplementedError("Attack Method not implemented! {}".format(path))
        
        with open(path) as f_attack:
            lines_attack = f_attack.readlines()

        search_text =  "INFO: attack success rate:"
        search_text2 = "INFO:  attack success rate:"
        
        if TRTE:
            search_text = "INFO: te attack success rate:"

        asr_list = []
        for line in lines_attack:

            # import pdb; pdb.set_trace()
            if line.__contains__(search_text) or line.__contains__(search_text2):
                asr = float(line.strip().split(' ')[-1])
                asr_list.append(asr)            
                if attack_method  == 'std':
                    result[attack_method + '_' + eps] = asr_list[-1]
                else:
                    result[attack_method] = asr_list[-1]
                    
    return result




def sort_paths_by_layer(paths):
    # './log_evaluation/cif/cif10/run_1/data/detection/cif10/wrn_28_10_10/fgsm/LayerPFS/layer_0/LR/log.txt'
    sorted_paths = sorted(paths, key=lambda x: int(x.split('/')[-2].split('_')[-1]))

    return sorted_paths



def formatNumber(x, digits):
    formatter = formatter = '{:.' + '{}'.format(digits) + 'f}'
    x = round(x, digits)
    return formatter.format(x)




def extract_information(root='./data', net=['cif10'], dest='./data/detection', nr=1, csv_filename='eval.csv', layers=True, ATTACKS='attacks', DETECTION='extracted_characteristics', architecture='', k=5):
    print( ' Extract information! ' )

    if not architecture == '':
        architecture = '/' + architecture


    # output_path = ''
    final = []
    paths = [] 
    for net_path in net:
        if ATTACKS == 'attacks' and DETECTION == 'detection':
            in_dir_attacks = os.path.join( root, ATTACKS,   'run_' + str(nr), net_path )
            in_dir_detects = os.path.join( root, DETECTION, 'run_' + str(nr), net_path )
        elif  ATTACKS == 'attacks' and DETECTION == 'extracted_characteristics':
            in_dir_attacks = os.path.join( root, ATTACKS,   'run_' + str(nr), net_path )
            in_dir_detects = os.path.join( root, DETECTION, 'run_' + str(nr), net_path )

        
        clean_acc = get_clean_accuracy( glob.glob( in_dir_attacks + architecture + "/**/log.txt", recursive=True ) )
        print("clean accuracy: ", clean_acc)

        if layers:
            detectors = DETECTOR_LIST_LAYERS
        else:
            detectors = DETECTOR_LIST

        # paths = []  
        for det in detectors:
            for classifier in CLF:
                for att in  ATTACKS_LIST:
                    lr_paths = []
                    if DETECTION == 'extracted_characteristics':
                        if layers:
                            if att == 'std':
                                search_path = in_dir_detects + architecture + "/**/" + att + "/**/" + det + "/k_{}/layer_*/log.txt".format(k)
                            else:
                                search_path = in_dir_detects + architecture + "/**/" + att + "/" + det + "/k_{}/layer_*/log.txt".format(k)
                            
                            
                            lr_paths = sort_paths_by_layer( glob.glob( search_path, recursive=True) ) 
                        else:
                            if att == 'std':
                                search_path = in_dir_detects + architecture + "/**/" + att + "/8_255/**/" + det + "/" + classifier + "/log.txt"
                            else:
                                search_path = in_dir_detects + architecture + "/**/" + att + "/" + det + "/" + classifier + "/log.txt"

                            lr_paths = glob.glob( search_path, recursive=True)

                    elif DETECTION == 'attack_transfer':
                            if att == 'std':
                                search_path = in_dir_detects + architecture + "/**/" + att + "/**/" + det + "/" + classifier + "/**" + "/log.txt"
                            else:
                                search_path = in_dir_detects + architecture + "/**/" + att + "/" + det + "/" + classifier + "/**" + "/log.txt"

                            lr_paths = glob.glob( search_path, recursive=True)

                    elif DETECTION == 'data_transfer':
                            if att == 'std':
                                search_path = in_dir_detects + architecture + "/**/" + att + "/**/" + det + "/" + classifier + "/**" + "/log.txt"
                            else:
                                search_path = in_dir_detects + architecture + "/**/" + att + "/" + det + "/" + classifier + "/**" + "/log.txt"

                            lr_paths = glob.glob( search_path, recursive=True)
                    else:
                        print("Not known!")

                    print("lr_paths: ", lr_paths)

                    paths = paths + lr_paths 
                    # import pdb; pdb.set_trace()
                    
    print(paths)
    meta_infos = {}

    for it, path in enumerate(paths):
        splitted = path.split('/')
        tmp_path = '/'.join(path.split('/')[:-1])
        charact     = torch.load( tmp_path + os.sep + 'characteristics') 
        charact_adv = torch.load( tmp_path + os.sep + 'characteristics_adv')
        for iter in range(1):
            meta_info = {}
            meta_info['path'] =       tmp_path
            meta_info['layer'] =      splitted[-2].split('_')[-1]
            meta_info['k'] =          splitted[-3].split('_')[-1]
            meta_info['attack'] =     splitted[-5]
            meta_info['architecture'] = splitted[-6]
            meta_info['dataset'] =    splitted[-7]
            meta_info['run'] =        splitted[-8].split('_')[-1]
            

            print(  formatNumber(charact[it][0], 2) )
            meta_info['score']     = formatNumber(charact[it][0],     2)
            meta_info['score_adv'] = formatNumber(charact_adv[it][0], 2)
            meta_infos[it] = meta_info
    
    df = pd.DataFrame.from_dict(meta_infos, orient='index')
    
    df.to_pickle(os.path.join(OUT_PATH, 'pkl', str(k) + "_" + csv_filename  +  '.pkl'))
    df.to_csv(os.path.join(OUT_PATH,    'csv', str(k) + "_" + csv_filename  +  '.csv'))
    df.to_excel(os.path.join(OUT_PATH,  'xlsx',str(k) + "_" + csv_filename  +  '.xlsx'))


def copy_run_to_root(root='./data/', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=2):
    print("INFO: Copy-to-Root: ", root, net, dest, run_nr)
    
    for net_path in net:
        in_dir  = os.path.join( dest, net_path, 'run_' + str(run_nr), 'data' )
        out_dir = os.path.join( root )

        print('in_dir:  ', in_dir )
        print('out_dir: ', out_dir)

        shutil.copytree(in_dir, out_dir, dirs_exist_ok=True)


def clean_root_folders(root='./data/clean_data', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg']):
    print("INFO: delete: ", root, net)

    # import pdb; pdb.set_trace()

    print("------------------ net possibilities --------------------")
    print('cif10', 'cif10vgg', 'cif100', 'cif100vgg')
    print('imagenet')
    print('imagenet32', 'imagenet64', 'imagenet128')
    print('celebaHQ32', 'celebaHQ64', 'celebaHQ128')

    for net_path in net:
        pth = root + os.sep + net_path
        shutil.rmtree(pth)


def copy_run_dest(root='./data/clean_data', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=2):
    print("INFO: ", root, net, dest, run_nr)

    # import pdb; pdb.set_trace()

    for net_path in net:
        out_dir = os.path.join( dest, net_path, 'run_' + str(run_nr) )
        parsed_paths = []
        pth = root + os.sep + net_path + "/**/log.txt"
        files_files_paths = glob.glob(pth, recursive = True)
        
        for text_file in files_files_paths:
            tmp_path =  os.path.split(text_file)[0]
            parsed_paths.append(tmp_path)
            print("parsed paths: ", parsed_paths)

            shutil.copytree(tmp_path, out_dir + tmp_path[1:], dirs_exist_ok=True)


def copy_var(input_path, output_path, nr):
    """
    input_path = path/to/dataset.csv
    output_path = path/to/var_{nr}
    nr ... run number
    """
    print("destination: ", output_path)
    if type(input_path) == list:
        for input_p in input_path:
            shutil.copy( input_p, output_path + str(nr) )
    else:
        shutil.copy( input_path, output_path + str(nr) )




if __name__ == "__main__":

    
    # OUT_PATH = "analysis/variance/run_gauss_"
    APP = '_k'
    # APP = '_HPF'
    # APP = 'layers'
    # APP = ''

    LAYERS=True
    # LAYERS=False
    CSV_FILE_PATH = []
    # NR = [1,2,3]
    # NR = [8]
    # NR = [3]
    NR = [1]
    DETECTION='extracted_characteristics'
    
    

    root='./data'
    net=['cif10']; 
    dest='./data/extracted_characteristics'

    for nr in NR:
        architecture='wrn_28_10_10'
        extract_information( root=root, net=['cif10'], dest=dest, nr=nr, csv_filename='cif10{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=5 )
        # extract_information( root=root, net=['cif10'], dest=dest, nr=nr, csv_filename='cif10{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=10 )
        # extract_information( root=root, net=['cif10'], dest=dest, nr=nr, csv_filename='cif10{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=20 )
        # extract_information( root=root, net=['cif10'], dest=dest, nr=nr, csv_filename='cif10{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=50 )
        
        architecture='wrn_28_10_100'
        extract_information( root=root, net=['cif100'], dest=dest, nr=nr, csv_filename='cif100{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=5 )
        # extract_information( root=root, net=['cif100'], dest=dest, nr=nr, csv_filename='cif100{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=10 )
        # extract_information( root=root, net=['cif100'], dest=dest, nr=nr, csv_filename='cif100{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=20 )
        # extract_information( root=root, net=['cif100'], dest=dest, nr=nr, csv_filename='cif100{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=50 )
        
        architecture='vgg_16_0_10'
        extract_information( root=root, net=['cif10vgg'], dest=dest, nr=nr, csv_filename='cif10vgg{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=5 )
        # extract_information( root=root, net=['cif10vgg'], dest=dest, nr=nr, csv_filename='cif10vgg{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=10 )
        # extract_information( root=root, net=['cif10vgg'], dest=dest, nr=nr, csv_filename='cif10vgg{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=20 )
        # extract_information( root=root, net=['cif10vgg'], dest=dest, nr=nr, csv_filename='cif10vgg{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=50 )

        architecture='vgg_16_0_100'
        extract_information( root=root, net=['cif100vgg'], dest=dest, nr=nr, csv_filename='cif100vgg{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=5 )
        # extract_information( root=root, net=['cif100vgg'], dest=dest, nr=nr, csv_filename='cif100vgg{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=10 )
        # extract_information( root=root, net=['cif100vgg'], dest=dest, nr=nr, csv_filename='cif100vgg{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=20 )
        # extract_information( root=root, net=['cif100vgg'], dest=dest, nr=nr, csv_filename='cif100vgg{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=50 )
        
        architecture='imagenet_50_2'
        extract_information( root=root, net=['imagenet'], dest=dest, nr=nr, csv_filename='imagenet{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=5 )
        # extract_information( root=root, net=['imagenet'], dest=dest, nr=nr, csv_filename='imagenet{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=10 )
        # extract_information( root=root, net=['imagenet'], dest=dest, nr=nr, csv_filename='imagenet{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=20 )
        # extract_information( root=root, net=['imagenet'], dest=dest, nr=nr, csv_filename='imagenet{}'.format(APP), DETECTION=DETECTION, layers=LAYERS, architecture=architecture, k=50 )