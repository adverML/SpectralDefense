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

import matplotlib.pyplot as plt

import foolbox
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import  L2DeepFoolAttack, LinfBasicIterativeAttack, FGSM, L2CarliniWagnerAttack, FGM, PGD

from utils import *


def create_advs(args, i, train=True):
    # output data
    # output_path_dir = create_dir_attacks(args, root='./data/attacks/')
    if args.attack == 'fgsm':
        attack_mode = 'FGSM'
        foldername = 'stepsize_0.001confidence_0epsilon_0.03maxiterations_1000iterations_40maxepsilon_0.03pnorm_inf'
        output_path_dir = '/home/lorenzp/adversialml/src/src/submodules/adversarial-detection/expts/numpy_data/cifar10/fold_{}/FGSM'.format(i)
    elif  args.attack == 'std':
        attack_mode = 'AA'
        output_path_dir = '/home/lorenzp/adversialml/src/src/submodules/adversarial-detection/expts/numpy_data/cifar10/fold_{}/AA'.format(i) 
        foldername = 'stepsize_0.001confidence_0epsilon_0.03maxiterations_1000iterations_40maxepsilon_0.03pnorm_inf'
    elif  args.attack == 'df':
        attack_mode = 'DF'
        output_path_dir = '/home/lorenzp/adversialml/src/src/submodules/adversarial-detection/expts/numpy_data/cifar10/fold_{}/DF'.format(i) 
        foldername = 'stepsize_0.001confidence_0epsilon_0.03maxiterations_1000iterations_40maxepsilon_0.03norm_2'
    elif  args.attack == 'cw':
        attack_mode = 'CW'
        output_path_dir = '/home/lorenzp/adversialml/src/src/submodules/adversarial-detection/expts/numpy_data/cifar10/fold_{}/CW'.format(i) 
        foldername = 'stepsize_0.001confidence_0epsilon_0.03maxiterations_1000iterations_40maxepsilon_0.03norm_2'

    save_args_to_file(args, output_path_dir)
    logger = Logger(output_path_dir + os.sep + foldername  + os.sep + 'log.txt')
    log_header(logger, args, output_path_dir, sys) # './data/attacks/imagenet32/wrn_28_10/fgsm'

    device_name =  getdevicename()

    # check args
    args = check_args(args, logger)

    #load model
    logger.log('INFO: Load model...')
    model, preprocessing = load_model(args)

    model = model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #load correctly classified data
    batch_size = 128 
    if device_name == 'titan v' and (args.net == 'imagenet128' or args.net == 'celebaHQ128'):
        batch_size = 24
    if device_name == 'a100' and (args.net == 'imagenet128' or args.net == 'celebaHQ128'):
        batch_size = 48

    if device_name == 'titan v' and (args.attack == "apgd-ce" or args.attack == "apgd-t" or args.attack == "fab-t" or args.attack == "square"):
        batch_size = 256

    if device_name == 'titan v' and (( args.attack == "cw" or args.attack == "df") and (args.net == 'imagenet64' or args.net == 'celebaHQ64')):
        batch_size = 48
    elif args.net == 'celebaHQ256':
        batch_size = 24
    elif device_name == 'titan v' and args.net == 'imagenet':
        batch_size = 32
    elif device_name == 'a100' and args.net == 'imagenet':
        batch_size = 64


    args.batch_size = batch_size
    logger.log('INFO: batch size: ' + str(batch_size))

    # set up final lists
    images = []
    images_advs = []
    
    labels = []
    labels_advs = []

    success_counter = 0

    counter = 0
    success = []
    success_rate = 0
    # input data    
    # clean_data_path = create_dir_clean_data(args, root='./data/clean_data/')
    clean_data_path = '/home/lorenzp/adversialml/src/src/submodules/adversarial-detection/expts/numpy_data/cifar10/fold_{}/CleanData'.format(i)
    logger.log('INFO: clean data path: ' + clean_data_path)

    logger.log('INFO: Perform attacks...')

    if args.attack == 'std' or args.attack == 'apgd-ce' or args.attack == 'apgd-t' or args.attack == 'fab-t' or args.attack == 'square':
        logger.log('INFO: Load data...')
        testset = load_test_set(args,  shuffle=args.shuffle_on)

        sys.path.append("./submodules/autoattack")
        from submodules.autoattack.autoattack import AutoAttack as AutoAttack_mod
        
        adversary = AutoAttack_mod(model, norm=args.norm, eps=epsilon_to_float(args.eps), log_path=output_path_dir + os.sep + 'log.txt', version=args.version)
        if args.individual:
            adversary.attacks_to_run = [ args.attack ]

        # run attack and save images
        with torch.no_grad():
            for x_test, y_test in testset:
                if not args.individual:
                    logger.log("INFO: mode: std; not individual")
                    adv_complete, max_nr = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)
                else: 
                    logger.log("INFO: mode: individual; not std")
                    adv_complete, max_nr = adversary.run_standard_evaluation_individual(x_test, y_test, bs=args.batch_size)
                    adv_complete = adv_complete[args.attack]

                tmp_images_advs = []
                # import pdb; pdb.set_trace()
                for it, img in enumerate(adv_complete):
                    if not ( np.abs(x_test[it] - img) <= 1e-5 ).all():
                        images.append(x_test[it])
                        tmp_images_advs.append(img)
                        success_counter = success_counter + 1
                        if (success_counter % 1000) == 0:
                            get_debug_info( msg="success_counter " + str(success_counter) + " / " + str(args.wanted_samples) )

                success.append( len(tmp_images_advs) / max_nr )

                images_advs += tmp_images_advs 
                tmp_images_advs = []

                success_rate  = np.mean(success)
                if success_counter >= args.wanted_samples:
                    print( " success: {:2f}".format(success_rate) )
                    break

    elif args.attack == 'fgsm' or args.attack == 'bim' or args.attack == 'pgd' or args.attack == 'df' or args.attack == 'cw': 
        #setup depending on attack
        if args.attack == 'fgsm':
            attack = FGSM()
            epsilons = [0.3]
            if args.net == 'mnist':
                epsilons = [0.4] 
        elif args.attack == 'bim':
            attack = LinfBasicIterativeAttack()
            epsilons = [0.03]
        elif args.attack == 'pgd':
            attack = PGD()
            epsilons = [0.03]
        elif args.attack == 'df':
            attack = L2DeepFoolAttack()
            epsilons = None
        elif args.attack == 'cw':
            attack = L2CarliniWagnerAttack(steps=1000)
            epsilons = None
        elif args.attack == 'autoattack':
            logger.log("Err: Auttoattack is started from another script! attacks_autoattack.py")
            raise NotImplementedError("Err: Wrong Keyword use 'std' for 'ind' for AutoAttack!")
        else:
            logger.log('Err: unknown attack')
            raise NotImplementedError('Err: unknown attack')

        # fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        fmodel = PyTorchModel(model, bounds=(0, 1))# preprocessing=preprocessing)

        logger.log('eps: {}'.format(epsilons))

        # testset = torch.load(clean_data_path + os.sep + 'clean_data')[:args.all_samples]
        if train: 
            testset_data  = np.load(clean_data_path + os.sep + 'data_tr_clean.npy')
            testset_label = np.load(clean_data_path + os.sep + 'labels_tr_clean.npy')
        else:
            testset_data  = np.load(clean_data_path + os.sep + 'data_te_clean.npy')
            testset_label = np.load(clean_data_path + os.sep + 'labels_te_clean.npy')  

        testset = [ [ testset_data[i], testset_label[i] ] for i in range(0, len(testset_label)) ]
        
        # testset = torch.load("/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/clean_data_cif1028_10")[:args.all_samples]

        logger.log("INFO: len(testset): {}".format(len(testset)))
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=args.shuffle_on)

        for image, label in test_loader:
            image = torch.squeeze(image)
            label = torch.squeeze(label)

            if batch_size == 1:
                image = torch.unsqueeze(image, 0)
                label = torch.unsqueeze(label, 0)

            # import pdb; pdb.set_trace()

            image = image.cuda()
            label = label.cuda()

            adv, adv_clip, success = attack(fmodel, image, criterion=foolbox.criteria.Misclassification(label), epsilons=epsilons)

            outputs = model(adv[0])
            _, predicted = torch.max(outputs.data, 1)
            predicted_adv = predicted.flatten()

            if not (args.attack == 'cw' or args.attack == 'df'):
                adv_clip = adv_clip[0] # list to tensor
                success = success[0]
            for idx, suc in enumerate(success):
                counter = counter + 1
                if suc:
                    images_advs.append( adv_clip[idx].squeeze_(0) )
                    images.append( image[idx].squeeze_(0) )

                    labels.append( label[idx].cpu().item() )
                    labels_advs.append( predicted_adv[idx].cpu().item() )

                    success_counter = success_counter + 1

            if success_counter >= args.wanted_samples:
                logger.log("INFO: wanted samples reached {}".format(args.wanted_samples))
                break

    logger.log("INFO: len(testset):   {}".format( len(testset) ))
    logger.log("INFO: success_counter {}".format(success_counter))
    logger.log("INFO: images {}".format(len(images)))
    logger.log("INFO: images_advs {}".format(len(images_advs)))

    if args.attack == 'std' or args.individual:
        logger.log('INFO: attack success rate: {}'.format(success_rate) )
    else:
        logger.log('INFO: attack success rate: {}'.format(success_counter / counter ) )


    return images, labels, images_advs, labels_advs, os.path.join(output_path_dir, foldername)
    


if __name__ == '__main__':

    # processing the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_nr",         default=1,    type=int, help="Which run should be taken?")

    parser.add_argument("--attack",         default='fgsm',            help=settings.HELP_ATTACK)
    parser.add_argument("--net",            default='cif10rn34sota',   help=settings.HELP_NET)
    parser.add_argument("--img_size",       default='32',    type=int, help=settings.HELP_IMG_SIZE)
    parser.add_argument("--num_classes",    default='10',    type=int, help=settings.HELP_NUM_CLASSES)
    parser.add_argument("--wanted_samples", default='8000',  type=int, help=settings.HELP_WANTED_SAMPLES)
    parser.add_argument("--all_samples",    default='7000',  type=int, help="Samples from generate Clean data")
    parser.add_argument("--shuffle_on",     action='store_true',       help="Switch shuffle data on")
    parser.add_argument("--nf",             default=3   , type=int,    help="Number of folds!")

    # Only for Autoatack
    parser.add_argument('--norm',       type=str, default='Linf')
    parser.add_argument('--eps',        type=str, default='8./255.')
    parser.add_argument('--individual',           action='store_true')
    parser.add_argument('--batch_size', type=int, default=1500)
    parser.add_argument('--log_path',   type=str, default='log.txt')
    parser.add_argument('--version',    type=str, default='standard')

    parser.add_argument('--net_normalization', action='store_false', help=settings.HELP_NET_NORMALIZATION)

    args = parser.parse_args()

    if args.attack == 'apgd-ce' or args.attack == 'apgd-t' or args.attack == 'fab-t' or args.attack == 'square':
        args.individual = True
        args.version = 'custom'
        

    # for i in range(1, args.nf + 1):

    #     images, labels, images_advs, labels_advs, output_path_dir = create_advs(args, i, train=True)

    #     print('images_path: ' + output_path_dir)
    #     make_dir(output_path_dir)
        
    #     np.save( os.path.join(output_path_dir, 'data_tr_clean.npy' ) , images   )
    #     np.save( os.path.join(output_path_dir, 'data_tr_adv.npy' )   , images_advs )

    #     np.save( os.path.join(output_path_dir, 'labels_tr_clean.npy' ) , labels   )
    #     np.save( os.path.join(output_path_dir, 'labels_tr_adv.npy' )   , labels_advs )


    for i in range(1, args.nf + 1):

        images, labels, images_advs, labels_advs, output_path_dir = create_advs(args, i, train=False)
        
        print('images_path: ' + output_path_dir)
        make_dir(output_path_dir)
        
        np.save( os.path.join(output_path_dir, 'data_te_clean.npy' ) , images   )
        np.save( os.path.join(output_path_dir, 'data_te_adv.npy' )   , images_advs )

        np.save( os.path.join(output_path_dir, 'labels_te_clean.npy' ) , labels   )
        np.save( os.path.join(output_path_dir, 'labels_te_adv.npy' )   , labels_advs )


        args.run_nr = i
        output_path_dir = create_dir_attacks(args, root='./data/attacks/')
        images_path, images_advs_path = create_save_dir_path(output_path_dir, args)

        # import pdb; pdb.set_trace()

        torch.save(images,      images_path,      pickle_protocol=4)
        torch.save(images_advs, images_advs_path, pickle_protocol=4)

        labels_path, labels_advs_path = create_save_dir_path(output_path_dir, args, filename='labels')
        torch.save(labels,      labels_path,      pickle_protocol=4)

    # logger.log('INFO: Done performing attacks and adversarial examples are saved!')