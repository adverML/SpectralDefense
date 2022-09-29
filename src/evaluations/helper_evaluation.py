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

from conf import settings as sett

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
    elif '0.5' in str_val:
        eps = '05'
    elif '0.4' in str_val:
        eps = '04'
    elif '0.3' in str_val:
        eps = '03'
    elif '0.2' in str_val:
        eps = '02'
    elif '0.1' in str_val:
        eps = '01'
    elif '0.01' in str_val:
        eps = '001'    
        
    return eps


def is_float(value):
  try:
    float(value)
    return True
  except:
    return False


def get_clean_accuracy(settings, paths):
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
            if 'apgd-cel2' in path:
                attack_method = 'apgd-cel2'
                if '05' in path:
                    eps = '05'
                elif '04' in path:
                    eps = '04'
                elif '03' in path:
                    eps = '03'
                elif '02' in path:
                    eps = '02'
                elif '01' in path:
                    eps = '01'
                elif '001' in path:
                    eps = '001'
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

        search_text = "INFO: attack success rate:"
        search_text2 = "INFO:  attack success rate:"
        
        if settings.TRTE:
            search_text = "INFO: te attack success rate:"

        asr_list = []
        for line in lines_attack:

            # import pdb; pdb.set_trace()
            if line.__contains__(search_text) or line.__contains__(search_text2):
                asr = float(line.strip().split(' ')[-1])
                asr_list.append(asr)            
                if attack_method  in ['std', 'apgd-ce', 'apgd-cel2']:
                    result[attack_method + '_' + eps] = asr_list[-1]
                else:
                    result[attack_method] = asr_list[-1]
                    
    return result


def sort_paths_by_layer(paths):
    # './log_evaluation/cif/cif10/run_1/data/detection/cif10/wrn_28_10_10/fgsm/LayerPFS/layer_0/LR/log.txt'
    sorted_paths = sorted(paths, key=lambda x: int(x.split('/')[-3].split('_')[-1]))

    return sorted_paths


def extract_information(settings, root='./data', net=['cif10'], dest='./data/detection', nr=1, csv_filename='eval.csv', layers=True, ATTACKS='attacks', DETECTION='detection', architecture=''):
    print( ' Extract information! ' )
    
    if not architecture == '':
        architecture = '/' + architecture
    
    COLUMNS       = [ 'asr', 'auc',   'f1', 'acc', 'pre', 'tpr', 'tnr', 'fnr', 'asrd']
    # COLUMNS_CLEAN = [ 'auc',  'f1',  'acc', 'pre', 'tpr', 'tnr', 'fnr', 'asr', 'asrd']
    COLUMNS_CLEAN = [ 'auc', 'acc', 'pre', 'tpr', 'f1',  'tnr', 'fnr', 'asr', 'asrd']
    

    # output_path = ''
    final = []
    for net_path in net:
        if ATTACKS == 'attacks' and DETECTION == 'detection':
            in_dir_attacks = os.path.join( root, ATTACKS,   'run_' + str(nr), net_path )
            in_dir_detects = os.path.join( root, DETECTION, 'run_' + str(nr), net_path )
        else:
            in_dir_attacks = os.path.join( root, ATTACKS,   'run_' + str(nr), net_path )
            in_dir_detects = os.path.join( root, DETECTION, 'run_' + str(nr), net_path )
            

        clean_acc = get_clean_accuracy( settings, glob.glob( in_dir_attacks + architecture + "/**/log.txt", recursive=True ) )
        print("clean accuracy: ", clean_acc)

        if layers:
            detectors = settings.DETECTOR_LIST_LAYERS
        else:
            detectors = settings.DETECTOR_LIST

        paths = []  
        for det in detectors:
            for classifier in settings.CLF:
                for att in  settings.ATTACKS_LIST:
                    lr_paths = []
                    if DETECTION == 'detection':
                        if layers:
                            if att in ['std', 'apgd-ce', 'apgd-cel2']:
                                search_path = in_dir_detects + architecture + "/**/" + att + "/**/" + det + "/layer_*/" + classifier + "/log.txt"
                            else:
                                search_path = in_dir_detects + architecture + "/**/" + att + "/" + det + "/layer_*/" + classifier + "/log.txt"
                            
                            lr_paths = sort_paths_by_layer( glob.glob( search_path, recursive=True) ) 
                        else:
                            if att in ['std', 'apgd-ce', 'apgd-cel2']:
                                search_path = in_dir_detects + architecture + "/**/" + att + "/8_255/**/" + det + "/" + classifier + "/log.txt"
                            else:
                                search_path = in_dir_detects + architecture + "/**/" + att + "/" + det + "/" + classifier + "/log.txt"
                            
                            lr_paths = glob.glob( search_path, recursive=True)

                    elif DETECTION == 'attack_transfer':
                            if att in ['std', 'apgd-ce', 'apgd-cel2']:
                                search_path = in_dir_detects + architecture + "/**/" + att + "/**/" + det + "/" + classifier + "/**" + "/log.txt"
                            else:
                                search_path = in_dir_detects + architecture + "/**/" + att + "/" + det + "/" + classifier + "/**" + "/log.txt"

                            lr_paths = glob.glob( search_path, recursive=True)

                    elif DETECTION == 'data_transfer':
                            if att in ['std', 'apgd-ce', 'apgd-cel2']:
                                search_path = in_dir_detects + architecture + "/**/" + att + "/**/" + det + "/" + classifier + "/**" + "/log.txt"
                            else:
                                search_path = in_dir_detects + architecture + "/**/" + att + "/" + det + "/" + classifier + "/**" + "/log.txt"

                            lr_paths = glob.glob( search_path, recursive=True)
                    else:
                        print("Not known!")

                    print("lr_paths: ", lr_paths)
                    
                    paths = paths + lr_paths 


        index_selected = []
        asr_name = []
        train_error = []
        
        len_found = False
        
        for path in paths:
            index = []
            line_split = []
            
            print("path: ", path)
            with open(path) as f:
                lines = f.readlines()
                
            # import pdb; pdb.set_trace()

            for line in lines:
                if line.__contains__("RES"):
                    splitted_line = line.strip().split(',')[1:]                
                    # print("splitted_line: ", splitted_line)
                    if is_float( splitted_line[0] ):
                        line_split.append( [ float(item) for item in splitted_line ] )

                if line.__contains__("OUTPUT_PATH_DIR:"):
                    index_split_list = line.strip().split(' ')
                    index.append( path )

                if line.__contains__("train error:"):
                    train_error.append(line.strip().split(' ')[-1])

                if line.__contains__("'attack':"):
                    for att in settings.ATTACKS_LIST:
                        if line.find(att) != -1:
                            if att in ['std', 'apgd-ce', 'apgd-cel2']:
                                tmp_eps = check_eps(line)
                                att = att + '_' + tmp_eps
                            asr_name.append( att )                        


            print("line_split: ", line_split)
            if not len_found and len(line_split) > 0:
                len_found = True
                
                # import pdb; pdb.set_trace()
                if not (len(line_split[-1]) == len(COLUMNS)-2):
                    COLUMNS       = [ 'asr', 'auc',   'f1', 'acc', 'pre', 'tpr', 'fnr', 'asrd']
                    COLUMNS_CLEAN = [ 'auc',  'f1',  'acc', 'pre', 'tpr', 'fnr', 'asr', 'asrd']

            if len(line_split) == 0:
                line_split.append( [ -1 for _ in range(len(COLUMNS)-2) ] )
                index.append( index_split_list[1] )
            csv_line = line_split[-1]
            if not csv_line[-1] == -1:
                fnr = float(csv_line[-1]) / 100.
                # if asr_name[-1] == 'std_8_255':
                asr = np.round(clean_acc[asr_name[-1]]*100, 2)
                csv_line.append(asr)
                asrd = np.round((fnr*asr), 2)
                
                try:
                    if settings.TRAINERR and len(train_error) > 0 and float(train_error[-1]) == 0.5:
                        asrd = str(asrd) + '*'
                except:
                    print("except")
                csv_line.append(asrd)
            else:
                csv_line.append(-1)
                csv_line.append(-1)

            index_selected.append(index[-1])
            final.append(csv_line)

        output_path = os.path.join(  root, DETECTION, 'run_' + str(nr), net_path, csv_filename  )
        print("output_path: ", output_path)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        
        df = pd.DataFrame(final, columns=COLUMNS_CLEAN, index=index_selected)
        df = df[COLUMNS]
        
        df.to_csv(output_path, sep=',')
    # import pdb; pdb.set_trace()

    return output_path


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

    OUT_PATH = "analysis/variance/run_"
    # OUT_PATH = "analysis/variance/run_gauss_"
    # APP = '_LID_ATTACKTRANSFER'
    # APP = '_LID'
    
    # APP = '_LIDNOISE'
    # APP = '_LIDNOISE_ATTACKTRANSFER'
    
    # APP = '_LIDFeatures'
    # APP = '_eps'
    # APP = '_apgd-ce'
    # APP = '_apgd-cel2'
    APP = '_apgd-ce_multilid'
    # APP = '_apgd-cel2'

    
    # APP = '_lidfeat'
    
    # APP = '_HPF'
    # # APP = 'layers'
    # APP = ''
    
    # APP = 'df_cw'
    # LAYERS=True
    LAYERS=False
    CSV_FILE_PATH = []
    # NR = [1,2,3]
    # NR = [8]
    # NR = [3]
    # NR = [4]
    # NR = [1]
    NR = [20, 21, 22, 23]
    
    

    for nr in NR:
        CSV_FILE_PATH.append( extract_information(sett, root='./data', net=['cif10'],      dest='./data/detection',   nr=nr, csv_filename='cif10{}.csv'.format(APP), layers=LAYERS ) )
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

