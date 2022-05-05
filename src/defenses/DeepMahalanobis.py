#!/usr/bin/env python3

import os
import torch
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import sklearn
import sklearn.covariance

from conf import settings

from utils import (
    get_normalization,
    normalize_images,
    get_num_classes,
    load_train_set
    )

def deep_mahalanobis(args, logger, model, images, images_advs, layers, get_layer_feature_maps, activation, output_path_dir):   
    args.batch_size = 100
    
    if args.net == 'imagenet':
        args.batch_size = 50
    
    sample_mean_path      = output_path_dir + 'sample_mean_' + args.net
    sample_precision_path = output_path_dir + 'precision_'   + args.net

    mean_exists =  os.path.exists(sample_mean_path)
    prec_exists =  os.path.exists(sample_precision_path)

    
    is_sample_mean_calculated = mean_exists and prec_exists and settings.ISSAMPLEMEANCALCULATED  

    logger.log( 'INFO: is_sample_mean_calculated set {}'.format(is_sample_mean_calculated))
    logger.log( 'INFO: {}, exists? {}'.format(sample_mean_path,      mean_exists) )
    logger.log( 'INFO: {}, exists? {}'.format(sample_precision_path, prec_exists) )    

    if mean_exists and prec_exists and is_sample_mean_calculated:
        logger.log('INFO: Sample Mean will NOT be (re) calculated!')
    else:
        logger.log('INFO: Sample Mean will     be (re) calculated!') 
        is_sample_mean_calculated = False

    act_layers_mah = layers
    # if not args.net == 'imagenet':
    #     act_layers_mah = fourier_act_layers

    num_classes = get_num_classes(args)
    
    if not is_sample_mean_calculated:
        logger.log('INFO: Calculate Sample Mean and precision for Mahalanobis... using training datasets')

        mean, std     = get_normalization(args)
        preprocessing = dict(mean=mean, std=std, axis=-3)
        trainloader = load_train_set(args, preprocessing=preprocessing)

        data_iter = iter(trainloader)
        im = data_iter.next()
        feature_list=[]
        if not args.net == 'cif10vgg' and not args.net == 'cif100vgg':
            feat_img = model(im[0].cuda())
            layers  = get_layer_feature_maps(activation, act_layers_mah)
        else:
            layers = get_layer_feature_maps(im[0].cuda(), act_layers_mah)

        m = len(act_layers_mah)
        
        for i in tqdm(range(m)):
            layer = layers[i]
            n_channels=layer.shape[1]
            feature_list.append(n_channels)
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        correct, total = 0, 0
        num_output = len(feature_list)
        num_sample_per_class = np.empty(num_classes)
        num_sample_per_class.fill(0)
        list_features = []
        for i in range(num_output):
            temp_list = []
            for j in range(num_classes):
                temp_list.append(0)
            list_features.append(temp_list)

        for data, target in trainloader:
            total += data.size(0)
            data = data.cuda()
            data = Variable(data)
            with torch.no_grad():
                if not args.net == 'cif10vgg' and not args.net == 'cif100vgg':
                    # data = Variable(data)
                    feat_img = model(data)
                    out_features  = get_layer_feature_maps(activation, act_layers_mah)
                else:
                    out_features = get_layer_feature_maps(data, act_layers_mah)

            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)

            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = out[i].view(1, -1)
                        out_count += 1
                else:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        out_count += 1                
                num_sample_per_class[label] += 1
        sample_class_mean = []
        out_count = 0
        for num_feature in feature_list:
            temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
            for j in range(num_classes):
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            sample_class_mean.append(temp_list)
            out_count += 1
        precision = []
        for k in range(num_output):
            X = 0
            for i in range(num_classes):
                if i == 0:
                    X = list_features[k][i] - sample_class_mean[k][i]
                else:
                    X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
            # find inverse            
            group_lasso.fit(X.cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().cuda()
            precision.append(temp_precision)

        torch.save(sample_class_mean, sample_mean_path)
        torch.save(precision,         sample_precision_path)
        
    #load sample mean and precision
    logger.log('INFO: Loading sample mean and precision...')
    sample_mean = torch.load(sample_mean_path)
    precision   = torch.load(sample_precision_path)    
    
    if  args.net == 'mnist' or args.net == 'cif10' or args.net == 'cif10vgg' or args.net == 'imagenet32' or args.net == 'celebaHQ32':
        if args.attack == 'gauss':
            magnitude = 0.01
        elif args.attack == 'fgsm':
            magnitude = 0.0002
        elif  args.attack == 'std':
            magnitude = 0.05
        elif args.attack == 'cw':
            magnitude = 0.00001
        else:
            magnitude = 0.00005
    else:
        if args.attack == 'gauss':
            magnitude = 0.01
        elif args.attack == 'fgsm':
            magnitude = 0.005
        elif args.attack == 'cw':
            magnitude = 0.00001
        
        elif args.attack == 'df':
            magnitude = 0.0005
        else:
            magnitude = 0.01

    image_loader = torch.utils.data.DataLoader(images,      batch_size=args.batch_size, shuffle=args.shuffle_on)
    adv_loader   = torch.utils.data.DataLoader(images_advs, batch_size=args.batch_size, shuffle=args.shuffle_on)

    def get_mah(test_loader, layer_index):
        Mahalanobis = []
        for data in test_loader:
            data = normalize_images(data, args)
            data = data.cuda()
            data = Variable(data, requires_grad=True)

            if not args.net == 'cif10vgg' and not args.net == 'cif100vgg':
                feat_img = model(data)
                out_features = get_layer_feature_maps(activation, [act_layers_mah[layer_index]])[0]
            else:
                out_features = get_layer_feature_maps(data,       [act_layers_mah[layer_index]])[0]

            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            out_features = torch.mean(out_features, 2)
            gaussian_score = 0
            for i in range(num_classes):
                batch_sample_mean = sample_mean[layer_index][i]
                zero_f = out_features.cpu().data - batch_sample_mean.cpu()
                term_gau = -0.5*torch.mm(torch.mm(zero_f.cuda(),precision[layer_index].cuda()), zero_f.t().cuda()).diag()
                if i == 0:
                    gaussian_score = term_gau.view(-1,1)
                else:
                    gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
            
            # Input_processing
            sample_pred = gaussian_score.max(1)[1]
            batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
            zero_f = out_features - Variable(batch_sample_mean.cuda(), requires_grad=True)
            pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index].cuda(), requires_grad=True)), zero_f.t()).diag()
            loss = torch.mean(-pure_gau)
            loss.backward()
            gradient = torch.ge(data, 0)
            gradient = (gradient.float() - 0.5) * 2
            tempInputs = torch.add(data.data,  gradient, alpha=-magnitude)
            with torch.no_grad():
                if not args.net == 'cif10vgg' and not args.net == 'cif100vgg':
                    feat_img = model(Variable(tempInputs))
                    noise_out_features = get_layer_feature_maps(activation, [act_layers_mah[layer_index]])[0]
                else:
                    noise_out_features = get_layer_feature_maps(Variable(tempInputs), [act_layers_mah[layer_index]])[0]
                
                noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
                noise_out_features = torch.mean(noise_out_features, 2)
                noise_gaussian_score = 0
            for i in range(num_classes):
                batch_sample_mean = sample_mean[layer_index][i]
                zero_f = noise_out_features.cpu().data - batch_sample_mean.cpu()
                term_gau = -0.5*torch.mm(torch.mm(zero_f.cuda(), precision[layer_index].cuda()), zero_f.t().cuda()).diag()
                if i == 0:
                    noise_gaussian_score = term_gau.view(-1,1)
                else:
                    noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)
            noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
            Mahalanobis.extend(noise_gaussian_score.cpu().numpy())
        return Mahalanobis


    logger.log('INFO: Calculating Mahalanobis scores...')
    Mah_adv = np.zeros((len(images_advs),len(act_layers_mah)))
    Mah = np.zeros((len(images_advs),len(act_layers_mah)))
    
    for layer_index in tqdm(range(len(act_layers_mah))):
        Mah_adv[:,layer_index]=np.array(get_mah(adv_loader, layer_index))
        Mah[:,layer_index]=np.array(get_mah(image_loader, layer_index))
    
    characteristics = Mah
    characteristics_adv = Mah_adv

    return characteristics, characteristics_adv