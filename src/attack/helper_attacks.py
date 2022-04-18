#!/usr/bin/env python3

from conf import settings

import torch
import os, sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import foolbox
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa


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

AA_std  = ['std', 'apgd-ce', 'apgd-t', 'fab-t' ,'square']
AA_plus = ['aa+', 'apgd-ce+', 'apgd-dlr+', 'fab+', 'square+', 'apgd-t+', 'fab-t+']

def adapt_batchsize(args, device_name):
    get_debug_info(msg="device_name: " + device_name)
    batch_size = 128 
    
    if device_name == 'nvidia titan v' and (args.net in ['imagenet128',  'celebaHQ128']):
        batch_size = 24
        
    if device_name == 'nvidia a100' and (args.net in ['imagenet', 'imagenet_hierarchy', 'restricted_imagenet', 'imagenet128', 'celebaHQ128']):
        batch_size = 48

    if device_name == 'nvidia titan v' and (args.attack in (AA_std + AA_plus) ):
        batch_size = 256

    if device_name == 'nvidia titan v' and (( args.attack in ["cw" , "df"] ) and (args.net in ['imagenet64', 'celebaHQ64'])):
        batch_size = 48
    elif args.net == 'celebaHQ256':
        batch_size = 24
    elif device_name == 'nvidia titan v' and args.net == 'imagenet':
        batch_size = 32
    
    return batch_size


def check_net_normalization(args):
    if args.net_normalization:
        if not args.attack in (AA_std + AA_plus):
            get_debug_info("Warning: Net normalization must be switched off!  Net normalization is switched off now!")
            args.net_normalization = False
            
    return args


def check_version(args):
    if args.attack in AA_plus:
        args.version = 'plus'
        if args.attack in AA_plus[1:]:
            args.individual = True
            # args.version = 'custom'
        
    if args.version == 'standard' and (args.attack in  AA_std[1:]):
        args.individual = True
        # args.version = 'custom'
        
    return args


def check_args_attack(args, version=True, net_normalization=True, num_classes=True, img_size=True):
    
    if version:
        args = check_version(args)

    if net_normalization:
        args = check_net_normalization(args)
    
    if num_classes:
        if (args.net  in ['cif10', 'cif10vgg', 'cif10rb', 'cif10rn34', 'cif10rn34sota'])  and not args.num_classes == 10:
            args.num_classes = 10
        elif (args.net in ['cif100', 'cif100vgg', 'cif100rn34'])  and not args.num_classes == 100:
            args.num_classes = 100
        elif (args.net in ['imagenet', 'imagenet_hierarchy', 'restricted_imagenet' , 'imagenet32', 'imagenet64', 'imagenet128'])  and not args.num_classes == 1000:
            
            if args.net == 'imagenet32' and not args.num_classes == 1000:
                imagnet32_class_set = set([10, 25, 50, 75, 100, 250])
                if args.num_classes in imagnet32_class_set:
                    get_debug_info(msg='Info: Num classes dont need to be checked!')
                else:
                    args.num_classes = 1000
            else:
                args.num_classes = 1000
            
        elif (args.net  in ['celebaHQ32', 'celebaHQ64', 'celebaHQ128'])  and not args.num_classes == 4:
            args.num_classes = 4

    if img_size:
        if (args.net in ['cif10' , 'cif10vgg' , 'cif100' , 'cif100vgg' , 'cif10rb' , 'imagenet32' , 'celebaHQ32' , 'cif10rn34sota'])  and not args.img_size == 32:
            args.img_size = 32
        if (args.net in [ 'imagenet64',  'celebaHQ64'])  and not args.img_size == 64:
            args.img_size =  64
        if (args.net  in ['imagenet128', 'celebaHQ128'] )  and not args.img_size == 128:
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
    
    dataset = torch.load(os.path.join(clean_data_path, clean_path))[:args.all_samples]
    get_debug_info( "actual len/wanted " + str(len(dataset)) + "/" + str(wanted_samples) )
    tqdm_total = round(wanted_samples / args.batch_size)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle_on)
    
    if args.attack  in (AA_std + AA_plus):
        logger.log('INFO: Load data...')
        # dataset = load_test_set(args,  shuffle=args.shuffle_on)

        sys.path.append("./submodules/autoattack")
        from submodules.autoattack.autoattack import AutoAttack as AutoAttack_mod
        
        adversary = AutoAttack_mod(model, norm=args.norm, eps=epsilon_to_float(args.eps), log_path=output_path_dir + os.sep + 'log.txt', version=args.version)
        if args.individual:
            adversary.attacks_to_run = [ args.attack.replace('+', '') ]

        # run attack and save images
        with torch.no_grad():
            for it, (x_test, y_test) in enumerate(tqdm(data_loader, total=tqdm_total)):
                # for x_test, x_test in data_loader:
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
                    # import pdb; pdb.set_trace()
                    adv_complete = adversary.run_standard_evaluation_individual(x_test, y_test, bs=args.batch_size, return_labels=True)
                    x_adv, y_adv, max_nr = adv_complete[args.attack.replace('+', '')]

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

    elif args.attack in ['fgsm', 'bim', 'pgd', 'l2pgd', 'df', 'linfdf', 'cw']: 

        logger.log("INFO: len(dataset): {}".format(len(dataset)))

        #setup depending on attack
        if args.attack == 'fgsm':
            attack = fa.FGSM() #linfs
            epsilons = [epsilon_to_float(args.eps)]
            if args.net == 'mnist':
                epsilons = [0.3] 
        elif args.attack == 'bim':
            attack = fa.LinfBasicIterativeAttack()
            epsilons = [epsilon_to_float(args.eps)]
        elif args.attack == 'pgd':
            attack = fa.PGD()
            epsilons = [epsilon_to_float(args.eps)]
        elif args.attack == 'l2pgd':
            attack = fa.L2PGD()
            epsilons = [0.3]
        elif args.attack == 'df':
            attack = fa.L2DeepFoolAttack()
            epsilons = None
        elif args.attack == 'linfdf':
            attack = fa.LinfDeepFoolAttack(steps=20)
            epsilons = None
        elif args.attack == 'cw':
            attack = fa.L2CarliniWagnerAttack(steps=1000)
            epsilons = None
        else:
            logger.log('Err: unknown attack')
            raise NotImplementedError('Err: unknown attack')

        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        logger.log('eps: {}'.format(epsilons))


        for it, (image, label) in enumerate(tqdm(data_loader, total=tqdm_total)):
            
            image = torch.squeeze(image)
            label = torch.squeeze(label)

            if args.batch_size == 1:
                image = torch.unsqueeze(image, 0)
                label = torch.unsqueeze(label, 0)
            
            image = image.cuda()
            label = label.cuda()

            adv, adv_clip, success = attack(fmodel, image, criterion=foolbox.criteria.Misclassification(label), epsilons=epsilons)

            if not (args.attack in ['cw', 'df', 'linfdf']):
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


    elif args.attack in ['gauss']:  # baseline accuracy
        from attack.noise_baseline import noisy
        
        logger.log("INFO: len(testset): {}".format(len(data_loader)))

        for it, (image, label) in enumerate(tqdm(data_loader, total=tqdm_total)):
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

    if args.attack in ['aa+', 'std']  or args.individual:
        logger.log('INFO: {} attack success rate: {}'.format(indicator[:2], success_rate) )
    else:
        logger.log('INFO: {} attack success rate: {}'.format(indicator[:2], success_counter / counter ) )

    return images, labels, images_advs, labels_advs