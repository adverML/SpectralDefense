#!/usr/bin/env python3

from conf import settings

import torch
import os, sys
import numpy as np
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
    create_dir_attacks,
    create_save_dir_path,
    create_dir_clean_data,
    epsilon_to_float,
    get_num_classes,
    load_model,
    get_debug_info
)

def adapt_batchsize(args, device_name):
    get_debug_info(msg="device_name: " + device_name)
    batch_size = 128 
    
    if device_name == 'titan v' and (args.net == 'imagenet128' or args.net == 'celebaHQ128'):
        batch_size = 24
        
    if device_name == 'nvidia a100' and (args.net == 'imagenet' or args.net == 'imagenet_hierarchy' or args.net == 'restricted_imagenet' or args.net == 'imagenet128' or args.net == 'celebaHQ128'):
        batch_size = 48

    if device_name == 'titan v' and (args.attack == "apgd-ce" or args.attack == "apgd-t" or args.attack == "fab-t" or args.attack == "square"):
        batch_size = 256

    if device_name == 'titan v' and (( args.attack == "cw" or args.attack == "df") and (args.net == 'imagenet64' or args.net == 'celebaHQ64')):
        batch_size = 48
    elif args.net == 'celebaHQ256':
        batch_size = 24
    elif device_name == 'titan v' and args.net == 'imagenet':
        batch_size = 32
    
    return batch_size


def check_net_normalization(args):
    if args.net_normalization:
        if not args.attack == 'std' and not args.attack == 'apgd-ce' and not args.attack == 'apgd-t' and not args.attack == 'fab-t' and not args.attack == 'square':
            get_debug_info("Warning: Net normalization must be switched off!  Net normalization is switched off now!")
            args.net_normalization = False
            
    return args


def check_args_attack(args, net_normalization=True, num_classes=True, img_size=True):
    
    if net_normalization:
        args = check_net_normalization(args)
    
    if num_classes:
        if (args.net == 'cif10' or args.net == 'cif10vgg' or  args.net == 'cif10rb' or  args.net == 'cif10rn34' or args.net == 'cif10rn34sota')  and not args.num_classes == 10:
            args.num_classes = 10  
        elif (args.net == 'cif100' or args.net == 'cif100vgg' or  args.net == 'cif100rn34')  and not args.num_classes == 100:
            args.num_classes = 100
        elif (args.net == 'imagenet' or args.net == 'imagenet_hierarchy' or args.net == 'restricted_imagenet' or args.net == 'imagenet32' or args.net == 'imagenet64' or args.net == 'imagenet128')  and not args.num_classes == 1000:
            args.num_classes = 1000
        elif (args.net == 'celebaHQ32' or args.net == 'celebaHQ64' or args.net == 'celebaHQ128')  and not args.num_classes == 4:
            args.num_classes = 4

    if img_size:
        if (args.net == 'cif10' or args.net == 'cif10vgg' or args.net == 'cif100' or args.net == 'cif100vgg' or args.net == 'cif10rb' or args.net == 'imagenet32' or args.net == 'celebaHQ32' or args.net == 'cif10rn34sota')  and not args.img_size == 32:
            args.img_size = 32
        if (args.net == 'imagenet64' or args.net == 'celebaHQ64' )  and not args.img_size == 64:
            args.img_size =  64
        if (args.net == 'imagenet128' or args.net == 'celebaHQ128' )  and not args.img_size == 128:
            args.img_size = 128

    return args


def create_advs(logger, args, model, output_path_dir, clean_data_path, wanted_samples, preprocessing, option=2):
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
        indicator = 'tr_'
    elif option == 1:
        indicator = 'te_'
    elif option == 2:
        indicator = ''

    clean_path = 'clean_' + indicator + 'data' 
    
    dataset = torch.load(os.path.join(clean_data_path, clean_path))[:wanted_samples]
    get_debug_info( "actual len/wanted " + str(len(dataset)) + "/" + str(wanted_samples) )

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle_on)

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

            for x_test, y_test in data_loader:

                x_test = torch.squeeze(x_test).cpu()
                y_test = torch.squeeze(y_test).cpu()

                if args.batch_size == 1:
                    x_test = torch.unsqueeze(x_test, 0)
                    y_test = torch.unsqueeze(y_test, 0)


                if not args.individual:
                    logger.log("INFO: mode: std; not individual")
                    x_adv, y_adv, max_nr = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size, return_labels=True)
                else: 
                    logger.log("INFO: mode: individual; not std")
                    adv_complete, max_nr = adversary.run_standard_evaluation_individual(x_test, y_test, bs=args.batch_size, return_labels=True)
                    x_adv, y_adv = adv_complete[args.attack]

                tmp_images_advs = []
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


        for image, label in data_loader:
            
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


    elif args.attack == 'gauss':  # baseline accuracy
        from attack.noise_baseline import noisy
        
        logger.log("INFO: len(testset): {}".format(len(data_loader)))

        for image, label in data_loader:
            for itx, img in enumerate(image):
                
                img_np = img.cpu().numpy().squeeze().transpose([1,2,0])
                image_adv = torch.from_numpy( noisy(img_np, noise_typ='gauss').transpose([2, 1, 0]) )
                images.append( img.squeeze().cpu() )
                images_advs.append( image_adv.cpu() )
                labels.append( label[itx].cpu() )
                labels_advs.append( label[itx].cpu() )
                
                success_counter  = success_counter + 1
                counter = counter + 1
                
                if success_counter >= wanted_samples:
                    logger.log("INFO: wanted samples reached {}".format(wanted_samples))
                    break

    logger.log("INFO: len(dataset):   {}".format( len(dataset) ))
    logger.log("INFO: success_counter {}".format(success_counter))
    logger.log("INFO: images {}".format(len(images)))
    logger.log("INFO: images_advs {}".format(len(images_advs)))

    if args.attack == 'std' or args.individual:
        logger.log('INFO: {} attack success rate: {}'.format(indicator[:2], success_rate) )
    else:
        logger.log('INFO: {} attack success rate: {}'.format(indicator[:2], success_counter / counter ) )

    return images, labels, images_advs, labels_advs