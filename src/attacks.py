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

from attacks.helper_attack import (
    adapt_batchsize
)



if __name__ == '__main__':
    # processing the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_nr",            default=1,       type=int, help="Which run should be taken?")
    
    parser.add_argument("--attack",            default='fgsm',            help=settings.HELP_ATTACK)
    parser.add_argument("--net",               default='cif10',           help=settings.HELP_NET)
    parser.add_argument("--img_size",          default='32',    type=int, help=settings.HELP_IMG_SIZE)
    parser.add_argument("--num_classes",       default='10',    type=int, help=settings.HELP_NUM_CLASSES)
    parser.add_argument("--wanted_samples",    default='2000',  type=int, help=settings.HELP_WANTED_SAMPLES)
    parser.add_argument("--all_samples",       default='4000',  type=int, help="Samples from generate Clean data")
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
    args.batch_size = adapt_batchsize(args, device_name)
    logger.log('INFO: batch size: ' + str(batch_size))

    # input data    
    clean_data_path = create_dir_clean_data(args, root='./data/clean_data/')
    logger.log('INFO: clean data path: ' + clean_data_path)

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
                    if not (np.abs(x_test[it] - img) <= 1e-5).all():
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

        testset = torch.load(clean_data_path + os.sep + 'clean_data')[:args.all_samples]
        # testset = torch.load("/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/clean_data_cif1028_10")[:args.all_samples]

        logger.log("INFO: len(testset): {}".format(len(testset)))
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=args.shuffle_on)

        #setup depending on attack
        if args.attack == 'fgsm':
            attack = FGSM()
            epsilons = [epsilon_to_float(args.eps)]
            if args.net == 'mnist':
                epsilons = [0.4] 
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

            if batch_size == 1:
                image = torch.unsqueeze(image, 0)
                label = torch.unsqueeze(label, 0)

            print( image.shape )

            image = image.cuda()
            label = label.cuda()

            adv, adv_clip, success = attack(fmodel, image, criterion=foolbox.criteria.Misclassification(label), epsilons=epsilons)

            # adv_arr = adv_clip[0][0].cpu().detach().numpy().transpose((1,2,0))
            # plt.imsave("./pics/adv_arr.png", adv_arr)
            # import pdb; pdb.set_trace()

            # adv_clip_pred = model( adv_clip[0] )

            if not (args.attack == 'cw' or args.attack == 'df'):
                adv_clip = adv_clip[0] # list to tensor
                success = success[0]
            for idx, suc in enumerate(success):
                counter = counter + 1
                if suc:
                    # import pdb; pdb.set_trace()
                    images_advs.append( adv_clip[idx].squeeze_(0).cpu() )
                    images.append( image[idx].squeeze_(0).cpu() )
                    # labels_advs.append( torch.argmax( adv_clip_pred[idx]).cpu().item() )
                    labels.append( label[idx].cpu().item() )
                    success_counter = success_counter + 1

            if success_counter >= args.wanted_samples:
                logger.log("INFO: wanted samples reached {}".format(args.wanted_samples))
                break
    
    
    elif args.attack == 'gauss':
        # baseline accuracy
        from attack.helper_attacks imort (

        )


    logger.log("INFO: len(testset):   {}".format( len(testset) ))
    logger.log("INFO: success_counter {}".format(success_counter))
    logger.log("INFO: images {}".format(len(images)))
    logger.log("INFO: images_advs {}".format(len(images_advs)))

    if args.attack == 'std' or args.individual:
        logger.log('INFO: attack success rate: {}'.format(success_rate) )
    else:
        logger.log('INFO: attack success rate: {}'.format(success_counter / counter ) )

    # create save dir 
    images_path, images_advs_path = create_save_dir_path(output_path_dir, args)
    logger.log('images_path: ' + images_path)

    # pdb.set_trace()
    torch.save(images,      images_path,      pickle_protocol=4)
    torch.save(images_advs, images_advs_path, pickle_protocol=4)

    labels_path, labels_advs_path = create_save_dir_path(output_path_dir, args, filename='labels')
    torch.save(labels,      labels_path,      pickle_protocol=4)
    # torch.save(labels_advs, labels_advs_path, pickle_protocol=4)

    logger.log('INFO: Done performing attacks and adversarial examples are saved!')