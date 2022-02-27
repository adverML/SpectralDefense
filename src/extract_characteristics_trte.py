#!/usr/bin/env python3

# cif10 4 titan because of extract char --> 4x45GB RAM

print('Load modules...')
import numpy as np
import pdb
import os
import pickle
import torch
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

from torch.autograd import Variable
from scipy.spatial.distance import cdist
from tqdm import tqdm
from collections import OrderedDict

from models.vgg_cif10 import VGG
from models.wideresidual import WideResNet, WideBasic
from models.orig_resnet import wide_resnet50_2

import argparse
import sklearn
import sklearn.covariance

from conf import settings
from utils import *


def extract_features(logger, args, model, input_path_dir, output_path_dir, wanted_samples, option=1):
    """
    docstring
    """

    if option == 0:
        indicator = '_tr'
    elif option == 1:
        indicator = '_te'
    elif optino == 2:
        indicator = ''


    images_path, images_advs_path = create_save_dir_path(input_path_dir, args, filename='images' + indicator)

    logger.log("INFO: images_path " + images_path)
    logger.log("INFO: images_advs " + images_advs_path)

    images =      torch.load(images_path)[:wanted_samples]
    images_advs = torch.load(images_advs_path)[:wanted_samples]

    number_images = len(images)
    logger.log("INFO: eps " + str(args.eps) + " INFO: nr_img " + str(number_images) + " INFO: Wanted Samples: " + str(wanted_samples) )


    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook


    if args.net == 'cif10vgg' or args.net == 'cif100vgg':

        # indice of activation layers
        act_layers= [2,5,9,12,16,19,22,26,29,32,36,39,42]
        # fourier_act_layers = [9,16,22,29,36,42]
        #get a list of all feature maps of all layers
        model_features = model.features
        def get_layer_feature_maps(X, layers):
            X_l = []
            for i in range(len(model_features)):
                X = model_features[i](X)
                if i in layers:
                    Xc = torch.Tensor(X.cpu())
                    X_l.append(Xc.cuda())
            return X_l

        # default layer
        # layers = [2, 12, 22, 29, 36, 42]
        layers = [9, 16, 22, 29, 36, 42]
        if args.net == 'cif100vgg' and (args.attack == 'cw' or args.attack == 'df'):
            layers = [42]

        if layer_nr == 0:
            layers = [2]
        elif layer_nr == 1:
            layers = [5]
        elif layer_nr == 2:
            layers = [9]
        elif layer_nr == 3:
            layers = [12]
        elif layer_nr == 4:
            layers = [16]
        elif layer_nr == 5:
            layers = [19]
        elif layer_nr == 6:
            layers = [22]
        elif layer_nr == 7:
            layers = [26]
        elif layer_nr == 8:
            layers = [29]
        elif layer_nr == 9:
            layers = [32]
        elif layer_nr == 10:
            layers = [36]
        elif layer_nr == 11:
            layers = [39]
        elif layer_nr == 12:
            layers = [42]
        else:
            logger.log( "INFO: layer nr > 12" + ", args.nr " + str(args.nr) )
            assert True


    elif args.net == 'cif10rn34sota':

        def get_layer_feature_maps(activation_dict, act_layer_list):
            act_val_list = []
            for it in act_layer_list:
                act_val = activation_dict[it]
                act_val_list.append(act_val)
            return act_val_list


        # import pdb; pdb.set_trace()


        if not args.nr == -1:

            # model.layer1[0].conv2.register_forward_hook( get_activation('1_conv2_0') )

            model.layer1[0].conv2.register_forward_hook( get_activation('1_conv2_0') )
            model.layer1[1].conv2.register_forward_hook( get_activation('1_conv2_1') )
            model.layer1[2].conv2.register_forward_hook( get_activation('1_conv2_2') )

            model.layer2[0].conv2.register_forward_hook( get_activation('2_conv2_0') )
            model.layer2[1].conv2.register_forward_hook( get_activation('2_conv2_1') )
            model.layer2[2].conv2.register_forward_hook( get_activation('2_conv2_2') )
            model.layer2[3].conv2.register_forward_hook( get_activation('2_conv2_3') )

            model.layer3[0].conv2.register_forward_hook( get_activation('3_conv2_0') )
            model.layer3[1].conv2.register_forward_hook( get_activation('3_conv2_1') )
            model.layer3[2].conv2.register_forward_hook( get_activation('3_conv2_2') )
            model.layer3[3].conv2.register_forward_hook( get_activation('3_conv2_3') )
            model.layer3[4].conv2.register_forward_hook( get_activation('3_conv2_4') )

            model.layer4[0].conv2.register_forward_hook( get_activation('4_conv2_0') )
            model.layer4[1].conv2.register_forward_hook( get_activation('4_conv2_1') )
            model.layer4[2].conv2.register_forward_hook( get_activation('4_conv2_2') ) 

        else:
            if not (args.attack == 'df' or  args.attack == 'cw'):
                
                if not args.attack == 'fgsm':
                    # last block
                    model.layer4[0].conv2.register_forward_hook( get_activation('4_conv2_0') )
                    model.layer4[1].conv2.register_forward_hook( get_activation('4_conv2_1') )
                    model.layer4[2].conv2.register_forward_hook( get_activation('4_conv2_2') ) 
                    layers = [
                        '4_conv2_0', '4_conv2_1', '4_conv2_3'
                    ]
                else:
                    model.layer4[0].conv2.register_forward_hook( get_activation('4_conv2_0') )
                    model.layer4[1].conv2.register_forward_hook( get_activation('4_conv2_1') )
                    # model.layer4[2].conv2.register_forward_hook( get_activation('4_conv2_2') )  
                    layers = [
                        '4_conv2_0', '4_conv2_1'
                    ]
            else:
                model.layer4[0].conv2.register_forward_hook( get_activation('4_conv2_0') )
                model.layer4[1].conv2.register_forward_hook( get_activation('4_conv2_1') )
                model.layer4[2].conv2.register_forward_hook( get_activation('4_conv2_2') ) 
                layers = [
                    '4_conv2_2'
                    # 'conv5_0_relu', 'conv5_1_relu'
                ]

        if layer_nr == 0:
            layers = ['1_conv2_0']
        elif layer_nr == 1:
            layers = ['1_conv2_1']
        elif layer_nr == 2:
            layers = ['1_conv2_2']
        elif layer_nr == 3:
            layers = ['2_conv2_0']
        elif layer_nr == 4:
            layers = ['2_conv2_1']
        elif layer_nr == 5:
            layers = ['2_conv2_2']
        elif layer_nr == 6:
            layers = ['2_conv2_3']
        elif layer_nr == 7:
            layers = ['3_conv2_0']
        elif layer_nr == 8:
            layers = ['3_conv2_1']
        elif layer_nr == 9:
            layers = ['3_conv2_2']
        elif layer_nr == 10:
            layers = ['3_conv2_3']
        elif layer_nr == 11:
            layers = ['3_conv2_4']
        elif layer_nr == 12:
            layers = ['4_conv2_0']
        elif layer_nr == 13:
            layers = ['4_conv2_1']
        elif layer_nr == 14:
            layers = ['4_conv2_2']
        else:
            logger.log( "INFO: layer nr > 14" + ", args.nr " + str(args.nr) )
            assert True



    elif args.net == 'cif10rn34' or args.net == 'cif100rn34':

        # import pdb; pdb.set_trace()


        def get_layer_feature_maps(activation_dict, act_layer_list):
            act_val_list = []
            for it in act_layer_list:
                act_val = activation_dict[it]
                act_val_list.append(act_val)
            return act_val_list




        if not args.nr == -1:

            model.conv2_x[0].residual_function[2].register_forward_hook( get_activation('conv2_0_relu') )
            model.conv2_x[1].residual_function[2].register_forward_hook( get_activation('conv2_1_relu') )
            model.conv2_x[2].residual_function[2].register_forward_hook( get_activation('conv2_2_relu') )

            model.conv3_x[0].residual_function[2].register_forward_hook( get_activation('conv3_0_relu') )
            model.conv3_x[1].residual_function[2].register_forward_hook( get_activation('conv3_1_relu') )
            model.conv3_x[2].residual_function[2].register_forward_hook( get_activation('conv3_2_relu') )
            model.conv3_x[3].residual_function[2].register_forward_hook( get_activation('conv3_3_relu') )

            model.conv4_x[0].residual_function[2].register_forward_hook( get_activation('conv4_0_relu') )
            model.conv4_x[1].residual_function[2].register_forward_hook( get_activation('conv4_1_relu') )
            model.conv4_x[2].residual_function[2].register_forward_hook( get_activation('conv4_2_relu') )
            model.conv4_x[3].residual_function[2].register_forward_hook( get_activation('conv4_3_relu') )
            model.conv4_x[4].residual_function[2].register_forward_hook( get_activation('conv4_4_relu') )

            model.conv5_x[0].residual_function[2].register_forward_hook( get_activation('conv5_0_relu') )
            model.conv5_x[1].residual_function[2].register_forward_hook( get_activation('conv5_1_relu') )
            model.conv5_x[2].residual_function[2].register_forward_hook( get_activation('conv5_2_relu') )      

        else:
            if not (args.attack == 'df' or  args.attack == 'cw'):
                
                if not args.attack == 'fgsm':
                    # last block
                    model.conv5_x[0].residual_function[2].register_forward_hook( get_activation('conv5_0_relu') )
                    model.conv5_x[1].residual_function[2].register_forward_hook( get_activation('conv5_1_relu') )
                    model.conv5_x[2].residual_function[2].register_forward_hook( get_activation('conv5_2_relu') )    
                    layers = [
                        'conv5_0_relu', 'conv5_1_relu', 'conv5_2_relu'
                    ]
                else:
                    model.conv5_x[0].residual_function[2].register_forward_hook( get_activation('conv5_0_relu') )
                    model.conv5_x[1].residual_function[2].register_forward_hook( get_activation('conv5_1_relu') )
                    # model.conv5_x[2].residual_function[2].register_forward_hook( get_activation('conv5_2_relu') )    
                    layers = [
                        'conv5_0_relu', 'conv5_1_relu'
                    ]
            else:
                model.conv5_x[0].residual_function[2].register_forward_hook( get_activation('conv5_0_relu') )
                model.conv5_x[1].residual_function[2].register_forward_hook( get_activation('conv5_1_relu') )
                model.conv5_x[2].residual_function[2].register_forward_hook( get_activation('conv5_2_relu') )   
                layers = [
                    'conv5_2_relu'
                    # 'conv5_0_relu', 'conv5_1_relu'
                ]

        if layer_nr == 0:
            layers = ['conv2_0_relu']
        elif layer_nr == 1:
            layers = ['conv2_1_relu']
        elif layer_nr == 2:
            layers = ['conv2_2_relu']
        elif layer_nr == 3:
            layers = ['conv3_0_relu']
        elif layer_nr == 4:
            layers = ['conv3_1_relu']
        elif layer_nr == 5:
            layers = ['conv3_2_relu']
        elif layer_nr == 6:
            layers = ['conv3_3_relu']
        elif layer_nr == 7:
            layers = ['conv4_0_relu']
        elif layer_nr == 8:
            layers = ['conv4_1_relu']
        elif layer_nr == 9:
            layers = ['conv4_2_relu']
        elif layer_nr == 10:
            layers = ['conv4_3_relu']
        elif layer_nr == 11:
            layers = ['conv4_4_relu']
        elif layer_nr == 12:
            layers = ['conv5_0_relu']
        elif layer_nr == 13:
            layers = ['conv5_1_relu']
        elif layer_nr == 14:
            layers = ['conv5_2_relu']
        else:
            logger.log( "INFO: layer nr > 14" + ", args.nr " + str(args.nr) )
            assert True

    elif (args.net == 'mnist' or args.net == 'cif10' or args.net == 'cif100' or args.net == 'celebaHQ32' or args.net == 'imagenet32' 
        or args.net == 'celebaHQ64' or args.net == 'celebaHQ128' or args.net == 'celebaHQ256'
        or args.net == 'imagenet64' or args.net == 'imagenet128'):

        def get_layer_feature_maps(activation_dict, act_layer_list):
            act_val_list = []
            for it in act_layer_list:
                act_val = activation_dict[it]
                act_val_list.append(act_val)
            return act_val_list


        # fourier_act_layers = [ 'conv2_0_relu_1', 'conv2_0_relu_4', 'conv2_1_relu_1', 'conv2_1_relu_4', 'conv2_2_relu_1', 'conv2_2_relu_4', 'conv2_3_relu_1', 'conv2_3_relu_4']

        if not args.nr == -1:

            model.conv2[0].residual[1].register_forward_hook( get_activation('conv2_0_relu_1') )
            model.conv2[0].residual[4].register_forward_hook( get_activation('conv2_0_relu_4') )

            model.conv2[1].residual[1].register_forward_hook( get_activation('conv2_1_relu_1') )
            model.conv2[1].residual[4].register_forward_hook( get_activation('conv2_1_relu_4') )

            model.conv2[2].residual[1].register_forward_hook( get_activation('conv2_2_relu_1') )
            model.conv2[2].residual[4].register_forward_hook( get_activation('conv2_2_relu_4') )

            model.conv2[3].residual[1].register_forward_hook( get_activation('conv2_3_relu_1') )
            model.conv2[3].residual[4].register_forward_hook( get_activation('conv2_3_relu_4') )


            model.conv3[0].residual[1].register_forward_hook( get_activation('conv3_0_relu_1') )
            model.conv3[0].residual[4].register_forward_hook( get_activation('conv3_0_relu_4') )

            # 5
            model.conv3[1].residual[1].register_forward_hook( get_activation('conv3_1_relu_1') )
            model.conv3[1].residual[4].register_forward_hook( get_activation('conv3_1_relu_4') )

            model.conv3[2].residual[1].register_forward_hook( get_activation('conv3_2_relu_1') )
            model.conv3[2].residual[4].register_forward_hook( get_activation('conv3_2_relu_4') )

            # 7
            model.conv3[3].residual[1].register_forward_hook(get_activation('conv3_3_relu_1'))
            model.conv3[3].residual[4].register_forward_hook(get_activation('conv3_3_relu_4'))


            model.conv4[0].residual[1].register_forward_hook(get_activation('conv4_0_relu_1'))
            model.conv4[0].residual[4].register_forward_hook(get_activation('conv4_0_relu_4'))

            model.conv4[1].residual[1].register_forward_hook(get_activation('conv4_1_relu_1'))
            model.conv4[1].residual[4].register_forward_hook(get_activation('conv4_1_relu_4'))

            model.conv4[2].residual[1].register_forward_hook(get_activation('conv4_2_relu_1'))
            model.conv4[2].residual[4].register_forward_hook(get_activation('conv4_2_relu_4'))

            model.conv4[3].residual[1].register_forward_hook(get_activation('conv4_3_relu_1'))
            model.conv4[3].residual[4].register_forward_hook(get_activation('conv4_3_relu_4'))

            model.relu.register_forward_hook(get_activation('relu'))
        else:
            if not (args.attack == 'df' or  args.attack == 'cw'):
                # 5
                model.conv3[1].residual[1].register_forward_hook(get_activation('conv3_1_relu_1'))
                model.conv3[1].residual[4].register_forward_hook(get_activation('conv3_1_relu_4'))

                # 7
                model.conv3[3].residual[1].register_forward_hook(get_activation('conv3_3_relu_1'))
                model.conv3[3].residual[4].register_forward_hook(get_activation('conv3_3_relu_4'))
                layers = [
                    'conv3_1_relu_1', 'conv3_1_relu_4', 'conv3_3_relu_1', 'conv3_3_relu_4'
                ]
            else:
                model.conv4[3].residual[1].register_forward_hook(get_activation('conv4_3_relu_1'))
                model.conv4[3].residual[4].register_forward_hook(get_activation('conv4_3_relu_4'))
                model.relu.register_forward_hook(get_activation('relu'))

                if args.net == 'celebaHQ32' or args.net == 'celebaHQ64' or args.net == 'celebaHQ128' or args.net == 'celebaHQ256':
                    layers = [
                        'relu'
                    ]
                else: 
                    layers = [
                        # 'conv4_3_relu_1',
                        # 'conv4_3_relu_4',
                        'relu'
                    ]


        # if layer_nr == 0:
        #     layers = ['conv2_0_relu_1', 'conv2_0_relu_4']
        # elif layer_nr == 1:
        #     layers = ['conv2_1_relu_1', 'conv2_1_relu_4']
        # elif layer_nr == 2:
        #     layers = ['conv2_2_relu_1', 'conv2_2_relu_4']
        # elif layer_nr == 3:
        #     layers = ['conv2_3_relu_1', 'conv2_3_relu_4']
        # elif layer_nr == 4:
        #     layers = ['conv3_0_relu_1', 'conv3_0_relu_4']
        # elif layer_nr == 5:
        #     layers = ['conv3_1_relu_1', 'conv3_1_relu_4']
        # elif layer_nr == 6:
        #     layers = ['conv3_2_relu_1', 'conv3_2_relu_4']
        # elif layer_nr == 7:
        #     layers = ['conv3_3_relu_1', 'conv3_3_relu_4']
        # elif layer_nr == 8:
        #     layers = ['conv4_0_relu_1', 'conv4_0_relu_4']
        # elif layer_nr == 9:
        #     layers = ['conv4_1_relu_1', 'conv4_1_relu_4']
        # elif layer_nr == 10:
        #     layers = ['conv4_2_relu_1', 'conv4_2_relu_4']
        # elif layer_nr == 11:
        #     layers = ['conv4_3_relu_1', 'conv4_3_relu_4']
        # elif layer_nr == 12:
        #     layers = ['relu']
        # else:
        #     logger.log( "INFO: layer nr > 24" + ", args.nr " + str(args.nr) )
        #     assert True


        if layer_nr == 0:
            layers = ['conv2_0_relu_1']
        elif layer_nr == 1:
            layers = ['conv2_0_relu_4']
        elif layer_nr == 2:
            layers = ['conv2_1_relu_1']
        elif layer_nr == 3:
            layers = ['conv2_1_relu_4']
        elif layer_nr == 4:
            layers = ['conv2_2_relu_1']
        elif layer_nr == 5:
            layers = ['conv2_2_relu_4']
        elif layer_nr == 6:
            layers = ['conv2_3_relu_1']
        elif layer_nr == 7:
            layers = ['conv2_3_relu_4']
        elif layer_nr == 8:
            layers = ['conv3_0_relu_1']
        elif layer_nr == 9:
            layers = ['conv3_0_relu_4']
        elif layer_nr == 10:
            layers = ['conv3_1_relu_1']
        elif layer_nr == 11:
            layers = ['conv3_1_relu_4']
        elif layer_nr == 12:
            layers = ['conv3_2_relu_1']
        elif layer_nr == 13:
            layers = ['conv3_2_relu_4']
        elif layer_nr == 14:
            layers = ['conv3_3_relu_1']
        elif layer_nr == 15:
            layers = ['conv3_3_relu_4']
        elif layer_nr == 16:
            layers = ['conv4_0_relu_1']
        elif layer_nr == 17:
            layers = ['conv4_0_relu_4']
        elif layer_nr == 18:
            layers = ['conv4_1_relu_1']
        elif layer_nr == 19:
            layers = ['conv4_1_relu_4']
        elif layer_nr == 20:
            layers = ['conv4_2_relu_1']
        elif layer_nr == 21:
            layers = ['conv4_2_relu_4']
        elif layer_nr == 22:
            layers = ['conv4_3_relu_1']
        elif layer_nr == 23:
            layers = ['conv4_3_relu_4']
        elif layer_nr == 24:
            layers = ['relu']
        else:
            logger.log( "INFO: layer nr > 24" + ", args.nr " + str(args.nr) )
            assert True

    elif args.net == 'imagenet':

        # print(model)
        # import pdb; pdb.set_trace()

        def get_layer_feature_maps(activation_dict, act_layer_list):
            act_val_list = []
            for it in act_layer_list:
                act_val = activation_dict[it]
                act_val_list.append(act_val)
            return act_val_list


        if not args.nr == -1:

            model.relu.register_forward_hook( get_activation('0_relu') )

            model.layer1[0].relu.register_forward_hook( get_activation('layer_1_0_relu') )
            model.layer1[1].relu.register_forward_hook( get_activation('layer_1_1_relu') )
            model.layer1[2].relu.register_forward_hook( get_activation('layer_1_2_relu') )

            model.layer2[0].relu.register_forward_hook( get_activation('layer_2_0_relu') )
            model.layer2[1].relu.register_forward_hook( get_activation('layer_2_1_relu') )
            model.layer2[2].relu.register_forward_hook( get_activation('layer_2_2_relu') )
            model.layer2[3].relu.register_forward_hook( get_activation('layer_2_3_relu') )

            model.layer3[0].relu.register_forward_hook( get_activation('layer_3_0_relu') )
            model.layer3[1].relu.register_forward_hook( get_activation('layer_3_1_relu') )
            model.layer3[2].relu.register_forward_hook( get_activation('layer_3_2_relu') )
            model.layer3[3].relu.register_forward_hook( get_activation('layer_3_3_relu') )
            model.layer3[4].relu.register_forward_hook( get_activation('layer_3_4_relu') )
            model.layer3[5].relu.register_forward_hook( get_activation('layer_3_5_relu') )

            model.layer4[0].relu.register_forward_hook( get_activation('layer_4_0_relu') )
            model.layer4[1].relu.register_forward_hook( get_activation('layer_4_1_relu') )
            model.layer4[2].relu.register_forward_hook( get_activation('layer_4_2_relu') )
        else:
            if not (args.attack == 'df' or  args.attack == 'cw'):

                # model.layer4[0].relu.register_forward_hook( get_activation('layer_4_0_relu') )
                model.layer4[1].relu.register_forward_hook( get_activation('layer_4_1_relu') )
                model.layer4[2].relu.register_forward_hook( get_activation('layer_4_2_relu') )
                
                layers = [
                    'layer_4_1_relu'
                    # 'layer_4_0_relu', 'layer_4_1_relu', 'layer_4_2_relu'
                    # 'layer_1_0_relu', 'layer_1_1_relu', 'layer_1_2_relu', 'layer_2_0_relu'
                ]
            else:
                # model.layer4[1].relu.register_forward_hook( get_activation('layer_4_1_relu') )
                model.layer4[2].relu.register_forward_hook( get_activation('layer_4_2_relu') )
                layers = [
                    'layer_4_2_relu'
                ]

        # layers = ['layer_1_0_relu', 'layer_1_1_relu', 'layer_1_2_relu', 'layer_2_0_relu', 'layer_2_1_relu', 'layer_2_2_relu']
        # layers = ['layer_1_0_relu', 'layer_1_1_relu', 'layer_1_2_relu', 'layer_2_0_relu']
        
        if layer_nr == 0:
            layers = ['0_relu']
        elif layer_nr == 1:
            layers = ['layer_1_0_relu']
        elif layer_nr == 2:
            layers = ['layer_1_1_relu']
        elif layer_nr == 3:
            layers = ['layer_1_2_relu']
        elif layer_nr == 4:
            layers = ['layer_2_0_relu']
        elif layer_nr == 5:
            layers = ['layer_2_1_relu']
        elif layer_nr == 6:
            layers = ['layer_2_2_relu']
        elif layer_nr == 7:
            layers = ['layer_2_3_relu']
        elif layer_nr == 8:
            layers = ['layer_3_0_relu']
        elif layer_nr == 9:
            layers = ['layer_3_1_relu']
        elif layer_nr == 10:
            layers = ['layer_3_2_relu']
        elif layer_nr == 11:
            layers = ['layer_3_3_relu']
        elif layer_nr == 12:
            layers = ['layer_3_4_relu']
        elif layer_nr == 13:
            layers = ['layer_3_5_relu']
        elif layer_nr == 14:
            layers = ['layer_4_0_relu']
        elif layer_nr == 15:
            layers = ['layer_4_1_relu']
        elif layer_nr == 16:
            layers = ['layer_4_2_relu']
        else:
            logger.log( "INFO: layer nr > 16" + ", args.nr " + str(args.nr) )
            assert True


    elif args.net == 'cif10_rb':

        # print(model)
        # import pdb; pdb.set_trace()

        def get_layer_feature_maps(activation_dict, act_layer_list):
            act_val_list = []
            for it in act_layer_list:
                act_val = activation_dict[it]
                act_val_list.append(act_val)
            return act_val_list


        if not args.nr == -1:
            # 0
            model.layer[0].block[0].relu_0.register_forward_hook( get_activation('layer_0_0_relu_0') )
            model.layer[0].block[0].relu_1.register_forward_hook( get_activation('layer_0_0_relu_1') )
            # 1
            model.layer[0].block[1].relu_0.register_forward_hook( get_activation('layer_0_1_relu_0') )
            model.layer[0].block[1].relu_1.register_forward_hook( get_activation('layer_0_1_relu_1') )
            # 2
            model.layer[0].block[2].relu_0.register_forward_hook( get_activation('layer_0_2_relu_0') )
            model.layer[0].block[2].relu_1.register_forward_hook( get_activation('layer_0_2_relu_1') )
            # 3
            model.layer[0].block[3].relu_0.register_forward_hook( get_activation('layer_0_3_relu_0') )
            model.layer[0].block[3].relu_1.register_forward_hook( get_activation('layer_0_3_relu_1') )
            # 4
            model.layer[1].block[0].relu_0.register_forward_hook( get_activation('layer_1_0_relu_0') )
            model.layer[1].block[0].relu_1.register_forward_hook( get_activation('layer_1_0_relu_1') )
            # 5
            model.layer[1].block[1].relu_0.register_forward_hook( get_activation('layer_1_1_relu_0') )
            model.layer[1].block[1].relu_1.register_forward_hook( get_activation('layer_1_1_relu_1') )
            # 6
            model.layer[1].block[2].relu_0.register_forward_hook( get_activation('layer_1_2_relu_0') )
            model.layer[1].block[2].relu_1.register_forward_hook( get_activation('layer_1_2_relu_1') )
            # 7
            model.layer[1].block[3].relu_0.register_forward_hook( get_activation('layer_1_3_relu_0') )
            model.layer[1].block[3].relu_1.register_forward_hook( get_activation('layer_1_3_relu_1') )
            # 8
            model.layer[2].block[0].relu_0.register_forward_hook( get_activation('layer_2_0_relu_0') )
            model.layer[2].block[0].relu_1.register_forward_hook( get_activation('layer_2_0_relu_1') )
            # 9
            model.layer[2].block[1].relu_0.register_forward_hook( get_activation('layer_2_1_relu_0') )
            model.layer[2].block[1].relu_1.register_forward_hook( get_activation('layer_2_1_relu_1') )
            # 10
            model.layer[2].block[2].relu_0.register_forward_hook( get_activation('layer_2_2_relu_0') )
            model.layer[2].block[2].relu_1.register_forward_hook( get_activation('layer_2_2_relu_1') )
            # 11
            model.layer[2].block[3].relu_0.register_forward_hook( get_activation('layer_2_3_relu_0') )
            model.layer[2].block[3].relu_1.register_forward_hook( get_activation('layer_2_3_relu_1') )
            # 12
            model.relu.register_forward_hook( get_activation('relu') )

        else:
            if not (args.attack == 'df' or  args.attack == 'cw'):
                model.layer[0].block[1].relu_0.register_forward_hook( get_activation('layer_0_1_relu_0') )
                # model.layer[0].block[1].relu_1.register_forward_hook( get_activation('layer_0_1_relu_1') )

                layers = [
                    'layer_0_1_relu_0' #, 'layer_0_1_relu_1'
                    # 'layer_1_0_relu', 'layer_1_1_relu', 'layer_1_2_relu', 'layer_2_0_relu'
                ]
            else:
                model.layer[0].block[1].relu_0.register_forward_hook( get_activation('layer_0_1_relu_0') )
                model.layer[0].block[1].relu_1.register_forward_hook( get_activation('layer_0_1_relu_1') )
                layers = [
                    'layer_0_1_relu_0', 'layer_0_1_relu_1'
                    # 'layer_1_0_relu', 'layer_1_1_relu', 'layer_1_2_relu', 'layer_2_0_relu'
                ]

        # layers = ['layer_1_0_relu', 'layer_1_1_relu', 'layer_1_2_relu', 'layer_2_0_relu', 'layer_2_1_relu', 'layer_2_2_relu']
        # layers = ['layer_1_0_relu', 'layer_1_1_relu', 'layer_1_2_relu', 'layer_2_0_relu']
        
        if layer_nr == 0:
            layers = ['layer_0_0_relu_0', 'layer_0_0_relu_1']
        elif layer_nr == 1:
            layers = ['layer_0_1_relu_0', 'layer_0_1_relu_1']
        elif layer_nr == 2:
            layers = ['layer_0_2_relu_0', 'layer_0_2_relu_1']
        elif layer_nr == 3:
            layers = ['layer_0_3_relu_0', 'layer_0_3_relu_1']
        elif layer_nr == 4:
            layers = ['layer_1_0_relu_0', 'layer_1_0_relu_1']
        elif layer_nr == 5:
            layers = ['layer_1_1_relu_0', 'layer_1_1_relu_1']
        elif layer_nr == 6:
            layers = ['layer_1_2_relu_0', 'layer_1_2_relu_1']
        elif layer_nr == 7:
            layers = ['layer_1_3_relu_0', 'layer_1_3_relu_0']
        elif layer_nr == 8:
            layers = ['layer_2_0_relu_0', 'layer_2_0_relu_1']
        elif layer_nr == 9:
            layers = ['layer_2_1_relu_0', 'layer_2_1_relu_1']
        elif layer_nr == 10:
            layers = ['layer_2_2_relu_0', 'layer_2_2_relu_1']
        elif layer_nr == 11:
            layers = ['layer_2_3_relu_0', 'layer_2_3_relu_1']
        elif layer_nr == 12:
            layers = ['relu']
        else:
            logger.log( "INFO: layer nr > 12" + ", args.nr " + str(args.nr) )
            assert True


    logger.log('INFO: ' + str(layers))

    ################Sections for each different detector
    #######Fourier section

    def calculate_fourier_spectrum(im, typ='MFS'):
        im = im.float()
        im = im.cpu()
        im = im.data.numpy() #transorm to numpy
        fft = np.fft.fft2(im)
        if typ == 'MFS':
            fourier_spectrum = np.abs(fft)
        elif typ == 'PFS':
            fourier_spectrum = np.abs(np.angle(fft))
        # if  (args.net == 'cif100' or args.net == 'cif100vgg' or args.net == 'cif100rn34'  or args.net == 'imagenet32' or args.net == 'imagenet64' or args.net == 'imagenet128' or  args.net == 'imagenet' or args.net == 'cif10_rb') and (args.attack=='cw' or args.attack=='df'):
        # if  (args.net == 'cif100' or args.net == 'cif100vgg' or args.net == 'cif100rn34' or  args.net == 'imagenet') and (args.attack=='cw' or args.attack=='df'):
        # if  args.max_freq_on:
        #     fourier_spectrum *= 1/np.max(fourier_spectrum)
        return fourier_spectrum


    # def calculate_fourier_spectrum_analysis(im, fr, to, typ='MFS'):
    #     im = im.float()
    #     im = im.cpu()
    #     im = im.data.numpy()  # [3, 32, 32]

    #     fft = np.fft.fft2(im) # [3, 32, 32]
    #     print(fft.shape)
    #     fft = np.max( fft, axis=0 )
    #     fft = np.mean( fft, axis=0 )
    #     print(fft.shape)
    #     if typ == 'MFS':
    #         fourier_spectrum = np.abs(fft)
    #     elif typ == 'PFS':
    #         fourier_spectrum = np.abs(np.angle(fft))

    #     # fourier_spectrum =  np.log( np.fft.fftshift(fourier_spectrum)   )
    #     fourier_spectrum =  np.fft.fftshift(fourier_spectrum) 

        
    #     print(fourier_spectrum.shape)  
    #     return fourier_spectrum[fr:to, fr:to]


    # def calculate_fourier_spectrum_analysis(im, fr, to, typ='MFS'):
    #     im = im.float()
    #     im = im.cpu()
    #     im = im.data.numpy()  # [3, 32, 32]

    #     fft = np.fft.fft2(im) # [3, 32, 32]
    #     print(fft.shape)
    #     fft = np.fft.fftshift(fft)
        
    #     rectangular = fft[:, fr:to, fr:to]
    #     # rectangular = fft

    #     fft = np.fft.ifftshift(rectangular)

    #     # fft = np.fft.fft2(fft)

    #     print(fft.shape)
    #     if typ == 'MFS':
    #         fourier_spectrum = np.abs(fft)
    #     elif typ == 'PFS':
    #         fourier_spectrum = np.abs(np.angle(fft))

    #     return fourier_spectrum


    def calculate_fourier_spectrum_analysis(im, fr, to, typ='MFS'):
        im = im.float()
        im = im.cpu()
        im = im.data.numpy() #[3, 32, 32]

        fft = np.fft.fft2(im)
        # fft = np.max( fft, axis=0 )
        # fft = np.mean( fft, axis=0 )
        if typ == 'MFS':
            fourier_spectrum = np.abs(fft)
        elif typ == 'PFS':
            fourier_spectrum = np.abs(np.angle(fft))

        # fourier_spectrum =  np.fft.fftshift(fourier_spectrum)       
        return fourier_spectrum[:, fr:to, fr:to]


    # def fourier_spectrum_analysis(im, fr, to):
    #     im = im.float()
    #     im = im.cpu()
    #     im = im.data.numpy()

    #     fft = np.fft.fft2(im)
    #     fft = np.mean( fft, axis=0 )
    #     fourier_spectrum = np.abs(fft)

    #     fourier_spectrum =  np.fft.fftshift(fourier_spectrum)       
    #     return fourier_spectrum[fr:to, fr:to]


    def calculate_spectra_analysis(images, fr, to, typ='MFS'):
        fs = []   
        for i in range(len(images)):
            image = images[i]
            fourier_image = calculate_fourier_spectrum_analysis(image, fr, to, typ=typ)
            fs.append(fourier_image.flatten())
        return fs

    def calculate_spectra(images, typ='MFS'):
        fs = []   
        for i in range(len(images)):
            image = images[i]
            fourier_image = calculate_fourier_spectrum(image, typ=typ)
            fs.append(fourier_image.flatten())
        return fs



    ###Fourier Input
    if args.detector == 'InputMFSAnalysis':
        mfs = calculate_spectra_analysis(images, args.fr, args.to)
        mfs_advs = calculate_spectra_analysis(images_advs, args.fr, args.to)
        characteristics       = np.asarray(mfs, dtype=np.float32)
        characteristics_adv   = np.asarray(mfs_advs, dtype=np.float32)


    elif args.detector == 'InputMFS':
        mfs      = calculate_spectra(images)
        mfs_advs = calculate_spectra(images_advs)
        characteristics       = np.asarray(mfs, dtype=np.float32)
        characteristics_adv   = np.asarray(mfs_advs, dtype=np.float32)


    elif args.detector == 'InputPFS':
        pfs = calculate_spectra(images, typ='PFS')
        pfs_advs = calculate_spectra(images_advs, typ='PFS')
        characteristics       = np.asarray(pfs, dtype=np.float32)
        characteristics_adv   = np.asarray(pfs_advs, dtype=np.float32)

    ###Fourier Layer   
    elif args.detector == 'LayerMFS':
        mfs = []
        mfs_advs = []

        for i in tqdm(range(number_images)):
            image = images[i].unsqueeze_(0)
            adv = images_advs[i].unsqueeze_(0)

            image = normalize_images(image, args)
            adv   = normalize_images(adv, args)

            inputimage = []
            inputadv = []
            if not args.net == 'cif10vgg' and not args.net == 'cif100vgg':
                if args.take_inputimage_off:
                    inputimage = [image]
                    inputadv = [adv]
                if args.nr == -1:
                    feat_img = model(image.cuda())
                    image_feature_maps = inputimage + get_layer_feature_maps(activation, layers)

                    feat_adv = model(adv.cuda())
                    adv_feature_maps   = inputadv   + get_layer_feature_maps(activation, layers)
                else:
                    feat_img = model(image.cuda())
                    image_feature_maps = inputimage + get_layer_feature_maps(activation, layers)

                    feat_adv = model(adv.cuda())
                    adv_feature_maps   = inputadv   + get_layer_feature_maps(activation, layers)
            else:
                image_c = image.cuda()
                adv_c = adv.cuda()
                if args.take_inputimage_off:
                    inputimage = [image_c]
                    inputadv   = [adv_c]
                image_feature_maps = inputimage + get_layer_feature_maps(image_c, layers)
                adv_feature_maps   = inputadv   + get_layer_feature_maps(adv_c,   layers)

            fourier_maps     = calculate_spectra(image_feature_maps)
            fourier_maps_adv = calculate_spectra(adv_feature_maps)
            mfs.append(np.hstack(fourier_maps))
            mfs_advs.append(np.hstack(fourier_maps_adv))

        if not args.nr == -1:
            nr_param = 1
            for i in image_feature_maps[0].shape:
                nr_param = nr_param * i
            logger.log("INFO: parameters: " +  str(image_feature_maps[0].shape) + ', '  + str(nr_param) )
            
        characteristics       = np.asarray(mfs,      dtype=np.float32)
        characteristics_adv   = np.asarray(mfs_advs, dtype=np.float32)
        
    elif args.detector == 'LayerPFS':

        pfs = []
        pfs_advs = []

        print('layers: ', layers)

        layers = whitebox_layers(layers, args)

        for i in tqdm(range(number_images)):
            image = images[i].unsqueeze_(0)
            adv = images_advs[i].unsqueeze_(0)

            image_c = normalize_images(image, args)
            adv_c   = normalize_images(adv, args)

            inputimage = []
            inputadv = []
            if not args.net == 'cif10vgg' and not args.net == 'cif100vgg': #wrn 
                if  args.take_inputimage_off:
                    inputimage = [image_c]
                    inputadv = [adv_c]
                if args.nr == -1:
                    feat_img = model(image_c.cuda())
                    image_feature_maps = inputimage + get_layer_feature_maps(activation, layers)

                    feat_adv = model(adv_c.cuda())
                    adv_feature_maps   = inputadv + get_layer_feature_maps(activation, layers)

                else:
                    feat_img = model(image_c.cuda())
                    image_feature_maps = inputimage + get_layer_feature_maps(activation, layers)

                    feat_adv = model(adv_c.cuda())
                    adv_feature_maps   = inputadv + get_layer_feature_maps(activation, layers)
            else: # vgg
                image_c = image.cuda()
                adv_c = adv.cuda()
                if args.take_inputimage_off:
                    inputimage = [image_c]
                    inputadv   = [adv_c]
                image_feature_maps = inputimage + get_layer_feature_maps(image_c, layers)
                adv_feature_maps   = inputadv   + get_layer_feature_maps(adv_c,   layers)

            fourier_maps     = calculate_spectra(image_feature_maps, typ='PFS')
            fourier_maps_adv = calculate_spectra(adv_feature_maps,   typ='PFS')

            pfs.append(np.hstack(fourier_maps))
            pfs_advs.append(np.hstack(fourier_maps_adv))

        if not args.nr == -1:
            nr_param = 1
            for i in image_feature_maps[0].shape:
                nr_param = nr_param * i
            print("INFO: parameters: ", image_feature_maps[0].shape, nr_param)
            
        characteristics       = np.asarray(pfs, dtype=np.float32)
        characteristics_adv   = np.asarray(pfs_advs, dtype=np.float32)


    #######LID section
    elif args.detector == 'LID':
        
        # layers = fourier_act_layers

        #hyperparameters
        batch_size = 100
        if args.net == 'mnist' or args.net == 'cif10' or args.net == 'cif10vgg' or args.net == 'imagenet32' or args.net == 'celebaHQ32':
            k = 20
        else: # cif100 cif100vgg imagenet imagenet64
            k = 10
            batch_size = 64

        
        def mle_batch(data, batch, k):
            data = np.asarray(data, dtype=np.float32)
            batch = np.asarray(batch, dtype=np.float32)
            k = min(k, len(data)-1)
            f = lambda v: -k / np.sum(np.log(v/v[-1]))
            a = cdist(batch, data)
            a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
            a = np.apply_along_axis(f, axis=1, arr=a)
            return a

        lid_dim = len(layers)
        shape = np.shape(images[0])
        
        def estimate(i_batch):
            start = i_batch * batch_size
            end = np.minimum(len(images), (i_batch + 1) * batch_size)
            n_feed = end - start
            lid_batch       = np.zeros(shape=(n_feed, lid_dim))
            lid_batch_adv   = np.zeros(shape=(n_feed, lid_dim))
            batch= torch.Tensor(n_feed, shape[0], shape[1], shape[2])
            batch_adv= torch.Tensor(n_feed, shape[0], shape[1], shape[2])
            for j in range(n_feed):
                batch[j,:,:,:] = images[j]
                batch_adv[j,:,:,:] = images_advs[j]

            batch = normalize_images(batch, args)
            batch_adv = normalize_images(batch_adv, args)

            if not args.net == 'cif10vgg' and not args.net == 'cif100vgg':
                feat_img = model(batch.cuda())
                X_act = get_layer_feature_maps(activation, layers)

                feat_adv = model(batch_adv.cuda())
                X_adv_act = get_layer_feature_maps(activation, layers)
            else:
                X_act = get_layer_feature_maps(batch.cuda(), layers)
                X_adv_act = get_layer_feature_maps(batch_adv.cuda(), layers)


            for i in range(lid_dim):
                X_act[i]       = np.asarray(X_act[i].cpu().detach().numpy()    , dtype=np.float32).reshape((n_feed, -1))
                X_adv_act[i]   = np.asarray(X_adv_act[i].cpu().detach().numpy(), dtype=np.float32).reshape((n_feed, -1))
                # Maximum likelihood estimation of Local Intrinsic Dimensionality (LID)
                lid_batch[:, i]       = mle_batch(X_act[i], X_act[i]      , k=k)
                lid_batch_adv[:, i]   = mle_batch(X_act[i], X_adv_act[i]  , k=k)
            return lid_batch, lid_batch_adv

        lids = []
        lids_adv = []
        n_batches = int(np.ceil(len(images) / float(batch_size)))
        
        for i_batch in tqdm(range(n_batches)):
            lid_batch, lid_batch_adv = estimate(i_batch)
            lids.extend(lid_batch)
            lids_adv.extend(lid_batch_adv)

        characteristics     = np.asarray(lids,     dtype=np.float32)
        characteristics_adv = np.asarray(lids_adv, dtype=np.float32)

    ####### Mahalanobis section
    elif args.detector == 'Mahalanobis':
        args.batch_size = 100
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
            if args.attack == 'fgsm':
                magnitude = 0.0002
            elif args.attack == 'cw':
                magnitude = 0.00001
            else:
                magnitude = 0.00005
        else:
            if args.attack == 'fgsm':
                magnitude = 0.005
            elif args.attack == 'cw':
                magnitude = 0.00001
            elif args.attack == 'df':
                magnitude = 0.0005
            else:
                magnitude = 0.01

        image_loader = torch.utils.data.DataLoader(images,      batch_size=100, shuffle=args.shuffle_on)
        adv_loader   = torch.utils.data.DataLoader(images_advs, batch_size=100, shuffle=args.shuffle_on)

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


    ####### LID_Class_Cond section
    elif args.detector == 'LID_Class_Cond':
        pass

    ####### ODD section https://github.com/jayaram-r/adversarial-detection
    elif args.detector == 'ODD':
        pass

    ####### Dknn section s
    elif args.detector == 'Dknn':
        pass

    ####### Trust section
    elif args.detector == 'Trust':
        pass

    else:
        logger.log('ERR: unknown detector')


    # Save
    logger.log("INFO: Save extracted characteristics ...")

    # characteristics_path, characteristics_advs_path = create_save_dir_path(output_path_dir, args, filename='characteristics' )
    # logger.log('INFO: characteristics:     ' + characteristics_path)
    # logger.log('INFO: characteristics_adv: ' + characteristics_advs_path)

    # torch.save(characteristics,      characteristics_path, pickle_protocol=4)
    # torch.save(characteristics_adv,  characteristics_advs_path, pickle_protocol=4)

    logger.log('INFO: Done extracting and saving characteristics!')

    return characteristics, characteristics_adv

if __name__ == '__main__':
    #processing the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_nr",          default=1,              type=int, help="Which run should be taken?")

    parser.add_argument("--attack"  ,        default='fgsm',          help=settings.HELP_ATTACK)
    parser.add_argument("--detector",        default='LayerMFS',      help=settings.HELP_DETECTOR)
    parser.add_argument('--take_inputimage_off', action='store_false',    help='Input Images for feature extraction. Default = True')
    parser.add_argument("--max_freq_on",     action='store_true',      help="Switch max frequency normalization on")

    parser.add_argument("--net",            default='cif10',          help=settings.HELP_NET)
    parser.add_argument("--nr",             default='-1',   type=int, help=settings.HELP_LAYER_NR)
    parser.add_argument("--wanted_samples", default='0', type=int, help=settings.HELP_WANTED_SAMPLES)
    parser.add_argument("--wanted_samples_tr", default='1000', type=int, help=settings.HELP_WANTED_SAMPLES)
    parser.add_argument("--wanted_samples_te", default='1000', type=int, help=settings.HELP_WANTED_SAMPLES)

    parser.add_argument('--img_size',       default='32',   type=int, help=settings.HELP_IMG_SIZE)
    parser.add_argument("--num_classes",    default='10',   type=int, help=settings.HELP_NUM_CLASSES)

    parser.add_argument("--shuffle_on",     action='store_true',      help="Switch shuffle data on")
    parser.add_argument('--net_normalization', action='store_true',   help=settings.HELP_NET_NORMALIZATION)

    # parser.add_argument("--eps",       default='-1',       help=settings.HELP_AA_EPSILONS) # to activate the best layers
    parser.add_argument("--eps",       default='8./255.',       help=settings.HELP_AA_EPSILONS)
    # parser.add_argument("--eps",       default='4./255.',       help=settings.HELP_AA_EPSILONS)
    # parser.add_argument("--eps",       default='2./255.',       help=settings.HELP_AA_EPSILONS)
    # parser.add_argument("--eps",       default='1./255.',       help=settings.HELP_AA_EPSILONS)
    # parser.add_argument("--eps",       default='1./255.',       help=settings.HELP_AA_EPSILONS)
    # parser.add_argument("--eps",       default='0.5/255.',       help=settings.HELP_AA_EPSILONS)

    # Frequency Analysis
    parser.add_argument("--fr", default='8',  type=int, help="InputMFS frequency analysis") 
    parser.add_argument("--to", default='24', type=int, help="InputMFS frequency analysis") 

    args = parser.parse_args()

    # max frequency
    # if args.max_freq_on or ((args.net == 'cif100' or args.net == 'cif100vgg' or args.net == 'cif100rn34') and (args.attack=='cw' or args.attack=='df')):
    #     args.max_freq_on = True

    # input data
    input_path_dir = create_dir_attacks(args, root='./data/attacks/')

    # output path dir
    output_path_dir = create_dir_extracted_characteristics(args, root='./data/extracted_characteristics/', wait_input=False)

    save_args_to_file(args, output_path_dir)
    logger = Logger(output_path_dir + os.sep + 'log.txt')
    log_header(logger, args, output_path_dir, sys) # './data/extracted_characteristics/imagenet32/wrn_28_10/std/8_255/LayerMFS'

    # check args
    args = check_args(args, logger)

    #load model
    logger.log('INFO: Loading model...')
    model, _ = load_model(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.eval()

    layer_nr = int(args.nr)
    logger.log("INFO: layer_nr " + str(layer_nr) ) 

    if args.wanted_samples > 0:

        args.wanted_samples_tr = 0
        args.wanted_samples_te = 0

        characteristics_path, characteristics_advs_path = create_save_dir_path(output_path_dir, args, filename='characteristics')
        characteristics, characteristics_adv = extract_features(logger, args, model, input_path_dir, output_path_dir, args.wanted_samples, option=2)

        logger.log('INFO: characteristics:     ' + characteristics_path)
        logger.log('INFO: characteristics_adv: ' + characteristics_advs_path)

        torch.save(characteristics,      characteristics_path, pickle_protocol=4)
        torch.save(characteristics_adv,  characteristics_advs_path, pickle_protocol=4)


    if args.wanted_samples_tr > 0:
        characteristics_path, characteristics_advs_path = create_save_dir_path(output_path_dir, args, filename='characteristics_tr')
        characteristics, characteristics_adv = extract_features(logger, args, model, input_path_dir, output_path_dir, args.wanted_samples_tr, option=0)

        logger.log('INFO: characteristics:     ' + characteristics_path)
        logger.log('INFO: characteristics_adv: ' + characteristics_advs_path)

        torch.save(characteristics,      characteristics_path, pickle_protocol=4)
        torch.save(characteristics_adv,  characteristics_advs_path, pickle_protocol=4)

        characteristics = []; characteristics_adv = []

    if args.wanted_samples_te > 0:

        characteristics_path, characteristics_advs_path = create_save_dir_path(output_path_dir, args, filename='characteristics_te')
        characteristics, characteristics_adv = extract_features(logger, args, model, input_path_dir, output_path_dir, args.wanted_samples_te, option=1)

        logger.log('INFO: characteristics:     ' + characteristics_path)
        logger.log('INFO: characteristics_adv: ' + characteristics_advs_path)

        torch.save(characteristics,      characteristics_path, pickle_protocol=4)
        torch.save(characteristics_adv,  characteristics_advs_path, pickle_protocol=4)

        characteristics = []; characteristics_adv = []



    logger.log('INFO: Done performing extracting features!')
    