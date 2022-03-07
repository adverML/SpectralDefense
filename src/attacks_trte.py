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

from utils import (
    Logger,
    log_header,
    create_dir_extracted_characteristics, 
    save_args_to_file,
    getdevicename,
    check_args,
    create_dir_attacks,
    create_save_dir_path,
    create_dir_clean_data,
    epsilon_to_float,
    get_num_classes,
    load_model,
    get_debug_info
)

from attack.helper_attacks import (
    adapt_batchsize
)


def create_advs(logger, args, output_path_dir, clean_data_path, wanted_samples, option=1):
    # set up final lists
    images = []
    images_advs = []
    
    labels = []
    labels_advs = []

    success_counter = 0

    counter = 0
    success = []
    success_rate = 0
    logger.log('INFO: Perform attacks...')

    if option == 0:
        indicator = 'tr'
        
    elif option == 1:
        indicator = 'te'
    elif option == 2:
        indicator = ''

    clean_path = 'clean_' + indicator + '_data' 
    dataset = torch.load(os.path.join(clean_data_path, clean_path))[:wanted_samples]
    get_debug_info( "actual len/wanted " + str(len(dataset)) + "/" + str(wanted_samples) )

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle_on)

    if args.attack == 'std' or args.attack == 'apgd-ce' or args.attack == 'apgd-t' or args.attack == 'fab-t' or args.attack == 'square':
        logger.log('INFO: Load data...')
        # dataset = load_test_set(args,  shuffle=args.shuffle_on)

        sys.path.append("./submodules/autoattack")
        from submodules.autoattack.autoattack import AutoAttack as AutoAttack_mod
        
        adversary = AutoAttack_mod(model, norm=args.norm, eps=epsilon_to_float(args.eps), log_path=output_path_dir + os.sep + 'log.txt', version=args.version)
        if args.individual:
            adversary.attacks_to_run = [ args.attack ]

        # run attack and save images
        with torch.no_grad():

            for x_test, y_test in test_loader:

                x_test = torch.squeeze(x_test).cpu()
                y_test = torch.squeeze(y_test).cpu()

                if args.batch_size == 1:
                    x_test = torch.unsqueeze(x_test, 0)
                    y_test = torch.unsqueeze(y_test, 0)

                
                # import pdb; pdb.set_trace()

                if not args.individual:
                    logger.log("INFO: mode: std; not individual")
                    x_adv, y_adv, max_nr = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size, return_labels=True)
                else: 
                    logger.log("INFO: mode: individual; not std")
                    adv_complete, max_nr = adversary.run_standard_evaluation_individual(x_test, y_test, bs=args.batch_size, return_labels=True)
                    x_adv, y_adv = adv_complete[args.attack]

                tmp_images_advs = []
                # import pdb; pdb.set_trace()
                for it, img in enumerate(x_adv):
                    if not (np.abs(x_test[it] - img) <= 1e-5).all():
                        images.append(x_test[it].cpu())                      
                        tmp_images_advs.append(img.cpu())
                        labels.append(y_test[it].cpu())
                        labels_advs.append(y_adv[it].cpu())  
                        success_counter = success_counter + 1
                        if (success_counter % 1000) == 0:
                            get_debug_info( msg="success_counter " + str(success_counter) + " / " + str(wanted_samples) )

                success.append( len(tmp_images_advs) / max_nr )

                images_advs += tmp_images_advs 
                tmp_images_advs = []

                success_rate  = np.mean(success)
                if success_counter >= wanted_samples:
                    get_debug_info( " success: {:2f}".format(success_rate) )
                    break

    elif args.attack == 'fgsm' or args.attack == 'bim' or args.attack == 'pgd' or args.attack == 'df' or args.attack == 'cw': 

        logger.log("INFO: len(dataset): {}".format(len(dataset)))

        #setup depending on attack
        if args.attack == 'fgsm':
            attack = FGSM()
            epsilons = [epsilon_to_float(args.eps)]
            if args.net == 'mnist':
                epsilons = [0.3] 
        elif args.attack == 'bim':
            attack = LinfBasicIterativeAttack()
            epsilons = [epsilon_to_float(args.eps)]
        elif args.attack == 'pgd':
            attack = PGD()
            epsilons = [epsilon_to_float(args.eps)]
        elif args.attack == 'df':
            attack = L2DeepFoolAttack()
            epsilons = None
        elif args.attack == 'cw':
            attack = L2CarliniWagnerAttack(steps=1000)
            epsilons = None
        else:
            logger.log('Err: unknown attack')
            raise NotImplementedError('Err: unknown attack')

        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        logger.log('eps: {}'.format(epsilons))

        for image, label in test_loader:

            image = torch.squeeze(image)
            label = torch.squeeze(label)

            if args.batch_size == 1:
                image = torch.unsqueeze(image, 0)
                label = torch.unsqueeze(label, 0)

            image = image.cuda()
            label = label.cuda()

            adv, adv_clip, success = attack(fmodel, image, criterion=foolbox.criteria.Misclassification(label), epsilons=epsilons)

            if not (args.attack == 'cw' or args.attack == 'df'):
                adv_clip = adv_clip[0] # list to tensor
                success = success[0]
                adv = adv[0]

            outputs = model(adv)
            _, predicted = torch.max(outputs.data, 1)
            predicted_adv = predicted.flatten()

            # import pdb; pdb.set_trace()

            for idx, suc in enumerate(success):
                counter = counter + 1
                if suc:
                    images_advs.append( adv_clip[idx].squeeze_(0).cpu() )
                    images.append( image[idx].squeeze_(0).cpu() )

                    labels.append( label[idx].cpu().item() )
                    labels_advs.append( predicted_adv[idx].cpu().item() )

                    success_counter = success_counter + 1

            if success_counter >= wanted_samples:
                logger.log("INFO: wanted samples reached {}".format(wanted_samples))
                break

    logger.log("INFO: len(dataset):   {}".format( len(dataset) ))
    logger.log("INFO: success_counter {}".format(success_counter))
    logger.log("INFO: images {}".format(len(images)))
    logger.log("INFO: images_advs {}".format(len(images_advs)))

    if args.attack == 'std' or args.individual:
        logger.log('INFO: {} attack success rate: {}'.format(indicator, success_rate) )
    else:
        logger.log('INFO: {} attack success rate: {}'.format(indicator, success_counter / counter ) )

    return images, labels, images_advs, labels_advs


if __name__ == '__main__':
    # processing the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_nr",            default=1,       type=int, help="Which run should be taken?")
    
    parser.add_argument("--attack",            default='fgsm',            help=settings.HELP_ATTACK)
    parser.add_argument("--net",               default='cif10',           help=settings.HELP_NET)
    parser.add_argument("--img_size",          default='32',    type=int, help=settings.HELP_IMG_SIZE)
    parser.add_argument("--num_classes",       default='10',    type=int, help=settings.HELP_NUM_CLASSES)
    parser.add_argument("--wanted_samples",    default='0',     type=int, help=settings.HELP_WANTED_SAMPLES)
    parser.add_argument("--wanted_samples_tr", default='1000',  type=int, help=settings.HELP_WANTED_SAMPLES)
    parser.add_argument("--wanted_samples_te", default='1000',  type=int, help=settings.HELP_WANTED_SAMPLES)
    # parser.add_argument("--all_samples",       default='4000',  type=int, help="Samples from generate Clean data")
    parser.add_argument("--shuffle_on",        action='store_true',       help="Switch shuffle data on")

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
        
    # output data
    output_path_dir = create_dir_attacks(args, root='./data/attacks/')

    save_args_to_file(args, output_path_dir)
    logger = Logger(output_path_dir + os.sep + 'log.txt')
    log_header(logger, args, output_path_dir, sys) # './data/attacks/imagenet32/wrn_28_10/fgsm'

    # check args
    args = check_args(args, logger)

    #load model
    logger.log('INFO: Load model...')
    model, preprocessing = load_model(args)

    model = model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ######################################################################################
    # load correctly classified data
    device_name = getdevicename()
    args.batch_size = adapt_batchsize(args, device_name)
    logger.log('INFO: batch size: ' + str(args.batch_size))

    # input data    
    clean_data_path = create_dir_clean_data(args, root='./data/clean_data/')
    logger.log('INFO: clean data path: ' + clean_data_path)

    if args.wanted_samples > 0:

        args.wanted_samples_tr = 0
        args.wanted_samples_te = 0

        images_path, images_advs_path = create_save_dir_path(output_path_dir, args)
        labels_path, labels_advs_path = create_save_dir_path(output_path_dir, args, filename='labels')
        
        images, labels, images_advs, labels_advs =  create_advs(logger, args, output_path_dir, clean_data_path, args.wanted_samples, option=2)

        torch.save(images,      images_path,      pickle_protocol=4)
        torch.save(images_advs, images_advs_path, pickle_protocol=4)
        torch.save(labels,      labels_path,      pickle_protocol=4)
        torch.save(labels_advs, labels_advs_path, pickle_protocol=4)

        images = []; labels = [];  images_advs = [];  labels_advs = []


    if args.wanted_samples_tr > 0:
        images, labels, images_advs, labels_advs = create_advs(logger, args, output_path_dir, clean_data_path, args.wanted_samples_tr, option=0)

        images_path, images_advs_path = create_save_dir_path(output_path_dir, args, filename='images_tr')
        labels_path, labels_advs_path = create_save_dir_path(output_path_dir, args, filename='labels_tr')

        torch.save(images,      images_path,      pickle_protocol=4)
        torch.save(images_advs, images_advs_path, pickle_protocol=4)
        torch.save(labels,      labels_path,      pickle_protocol=4)
        torch.save(labels_advs, labels_advs_path, pickle_protocol=4)

        images = []; labels = [];  images_advs = [];  labels_advs = []


    if args.wanted_samples_te > 0:

        images_path, images_advs_path = create_save_dir_path(output_path_dir, args, filename='images_te')
        labels_path, labels_advs_path = create_save_dir_path(output_path_dir, args, filename='labels_te')

        images, labels, images_advs, labels_advs = create_advs(logger, args, output_path_dir, clean_data_path, args.wanted_samples_te, option=1)

        torch.save(images,      images_path,      pickle_protocol=4)
        torch.save(images_advs, images_advs_path, pickle_protocol=4)
        torch.save(labels,      labels_path,      pickle_protocol=4)
        torch.save(labels_advs, labels_advs_path, pickle_protocol=4)

        images = []; labels = [];  images_advs = [];  labels_advs = []


    logger.log('INFO: Done performing attacks and adversarial examples are saved!')