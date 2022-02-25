print('Load modules...')
import numpy as np
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

import pdb
import os
import pickle

from conf import settings
from utils import get_appendix
from utils import layer_name_cif10vgg, layer_name_cif10


# from nnif import get_knn_layers, calc_all_ranks_and_dists, append_suffix

NORMALIZED = settings.NORMALIZED



#processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--attack"  , default='fgsm',                 help=settings.help_attack)
parser.add_argument("--detector", default='LayerMFS',             help=settings.detector)
parser.add_argument("--net",      default='cif10vgg',             help=settings.help_net)
parser.add_argument("--nr",       default='-1'        , type=int, help=settings.help_layer_nr)
parser.add_argument("--wanted_samples", default='2000', type=int, help=settings.wanted_samples)
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument("--num_classes", default='1000', type=int,    help=settings.help_num_classes)


# parser.add_argument("--eps",       default='-1',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
parser.add_argument("--eps",       default='0.03137254901960784',  type=float, help="epsilon: 8/255")
# parser.add_argument("--eps",       default='0.01568627450980392', type=float, help="epsilon: 4/255, ")
# parser.add_argument("--eps",       default='0.00784313725490196', type=float, help="epsilon: 2/255, ")
# parser.add_argument("--eps",       default='0.00392156862745098', type=float, help="epsilon: 1/255, ")
# parser.add_argument("--eps",       default='0.00196078431372549', type=float, help="epsilon: 0.5/255")
# parser.add_argument("--eps",       default='0.0004901960784313725', type=float, help="epsilon: 0.125/255")

args = parser.parse_args()

#choose attack
attack_method = args.attack
detector = args.detector
net = args.net
print("normalized: ", NORMALIZED, args)

appendix = get_appendix(args.num_classes, settings.MAX_CLASSES_IMAGENET)

#load adversarials and their non-adversarial counterpart
print('Loading images and adversarial examples...')
if attack_method == 'apgd-ce' or attack_method == 'apgd-t' or attack_method == 'fab-t' or attack_method == 'square':
    images =      torch.load("/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete_imagenet/ind/" + attack_method + "_individ_0.03137_Linf_orig")
    images_advs = torch.load("/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete_imagenet/ind/" + attack_method + "_individ_0.03137_Linf_pert")
elif attack_method == 'std':
    nr_samples = 6000
    if args.eps >= 0.03137254901960780:
            images_path =      "/home/lorenzp/adversialml/src/src/auto-attack/adv_complete_" + args.net + "/std/0.031373_std_6000_Linf_sorted_orig"
            images_advs_path = "/home/lorenzp/adversialml/src/src/auto-attack/adv_complete_" + args.net + "/std/0.031373_std_6000_Linf_sorted_pert"
    elif args.eps >= 0.0156862745098039:
        images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete_imagenet/std/0.015686_std_6000_Linf_sorted_orig"
        images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete_imagenet/std/0.015686_std_6000_Linf_sorted_pert"
    elif args.eps >= 0.0078431372549019:
        images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete_imagenet/std/0.007843_std_6000_Linf_sorted_orig"
        images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete_imagenet/std/0.007843_std_6000_Linf_sorted_pert"
    elif args.eps >= 0.0039215686274509:
        images_path =       "/home/lorenzp/adversialml/src/src/auto-attack/adv_complete_" + args.net + "/std/0.003922_std_6000_Linf_sorted_orig"
        images_advs_path =  "/home/lorenzp/adversialml/src/src/auto-attack/adv_complete_" + args.net + "/std/0.003922_std_6000_Linf_sorted_pert"
    elif args.eps >= 0.0019607843137254:
        images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete_imagenet/std/0.001961_std_6000_Linf_sorted_orig"
        images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete_imagenet/std/0.001961_std_6000_Linf_sorted_pert"
    elif args.eps >= 0.000490196078431372:
        images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete_imagenet/std/0.000490_std_6000_Linf_sorted_orig"
        images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete_imagenet/std/0.000490_std_6000_Linf_sorted_pert"
else:
    images_path =      './data/' + net + '_adversarial_images/' + net + '_images_'    + attack_method + appendix
    images_advs_path = './data/' + net + '_adversarial_images/' + net + '_images_adv_'+ attack_method + appendix


images =      torch.load(images_path)[:args.wanted_samples]
images_advs = torch.load(images_advs_path)[:args.wanted_samples]


print("images_path ", images_path)
print("images_advs ", images_advs_path)

number_images = len(images)

print("eps", args.eps, "nr_img", number_images)


#load model
print('Loading model...')
model, _ = load_model(args)


# depth = 28
# widen_factor = 10
# if net == 'imagenet':
#     model = wide_resnet50_2(pretrained=True)
# if net == 'imagenet32' or net == 'imagenet64' or net == 'imagenet128':
#     print("depth: ", depth, ", widen_factor", widen_factor)
#     model = WideResNet(num_classes=args.num_classes, block=WideBasic, depth=depth, widen_factor=widen_factor)
#     # model = wide_resnet50_2(pretrained=True)

#     ckpt = torch.load('/home/lorenzp/adversialml/src/pytorch-classification/checkpoints/' + str(args.net) + '/wideresent2810' + appendix + '/model_best.pth.tar')

#     new_state_dict = {}
#     for k, v in ckpt['state_dict'].items():
#         new_key = k[7:]
#         new_state_dict[new_key] = v

#     model.load_state_dict(new_state_dict)
#     # model.load_state_dict(ckpt)
#     # model.cuda()
# elif net == 'cif10':
#     model = WideResNet(num_classes=settings.MAX_CLASSES_CIF10, block=WideBasic, depth=depth, widen_factor=widen_factor)
#     # if not NORMALIZED:
#     #     checkpoint = torch.load('/home/lorenzp/adversialml/src/src/pytorch-CelebAHQ/autoattack/resnet_wide_orig.ckpt')
#     # else:
#     checkpoint = torch.load('./checkpoint/wideresnet_2810/wide_resnet_ckpt.pth')
#     # model.load_state_dict(checkpoint['state_dict'])
#     new_state_dict = OrderedDict()
#     for k, v in checkpoint['net'].items():
#         name = k[7:] # remove `module.`
#         new_state_dict[name] = v
#     model.load_state_dict(new_state_dict)

#     # checkpoint = torch.load('/home/lorenzp/adversialml/src/src/auto-attack/resnet_wide_orig.ckpt')

# elif net == 'cif10vgg':
#     depth = 16
#     widen_factor = 0
#     model = VGG('VGG16')
#     checkpoint = torch.load('./checkpoint/vgg16/original/models/vgg_cif10.pth')
#     new_state_dict = OrderedDict()
#     for k, v in checkpoint['net'].items():
#         name = k[7:] # remove `module.`
#         new_state_dict[name] = v
#     model.load_state_dict(new_state_dict)


# elif net == 'cif100':
#     model = WideResNet(num_classes=settings.MAX_CLASSES_CIF100, block=WideBasic, depth=depth, widen_factor=widen_factor)
#     # if not NORMALIZED:
#     #     checkpoint = torch.load('/home/lorenzp/adversialml/src/src/pytorch-CelebAHQ/autoattack/resnet_wide_orig.ckpt')
#     # else:
#     checkpoint = torch.load('./../pytorch-classification/checkpoints/cifar100/wideresnet2810/model_best.pth.tar')
#     # model.load_state_dict(checkpoint['state_dict'])
#     new_state_dict = OrderedDict()
#     for k, v in checkpoint['state_dict'].items():
#         name = k[7:] # remove `module.`
#         new_state_dict[name] = v
#     model.load_state_dict(new_state_dict)

# elif net == 'celebaHQ32':
#     model = WideResNet(num_classes=settings.MAX_CLASSES_CELEBAHQ, block=WideBasic, depth=depth, widen_factor=widen_factor)
#     ckpt = torch.load('./checkpoint/wrn2810/32x32_64_0.1_Smiling_a100_Wednesday_18_August_2021_16h_02m_16s/wrn2810-175-best.pth')
#     model.load_state_dict(ckpt)

# elif net == 'celebaHQ64':
#     model = WideResNet(num_classes=settings.MAX_CLASSES_CELEBAHQ, block=WideBasic, depth=depth, widen_factor=widen_factor)
#     ckpt = torch.load('./checkpoint/wrn2810/64x64_64_0.1_Smiling_a100_Wednesday_18_August_2021_16h_11m_52s/wrn2810-152-best.pth')
#     model.load_state_dict(ckpt)

# elif net == 'celebaHQ128':
#     model = WideResNet(num_classes=settings.MAX_CLASSES_CELEBAHQ, block=WideBasic, depth=depth, widen_factor=widen_factor)
#     ckpt = torch.load('./checkpoint/wrn2810/128x128_64_0.1_Smiling_a100_Sunday_22_August_2021_13h_01m_15s/wrn2810-80-regular.pth')
#     model.load_state_dict(ckpt)

# else:
#     print('unknown model')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = model.eval()

# pdb.set_trace()

# normalization
def cifar_normalize(images):
    if net == 'cif10' or net == 'cif10vgg':
        images[:,0,:,:] = (images[:,0,:,:] - 0.4914)/0.2023
        images[:,1,:,:] = (images[:,1,:,:] - 0.4822)/0.1994
        images[:,2,:,:] = (images[:,2,:,:] - 0.4465)/0.2010
    elif net == 'cif100':
        images[:,0,:,:] = (images[:,0,:,:] - 0.4914)/0.2023
        images[:,1,:,:] = (images[:,1,:,:] - 0.4822)/0.1994
        images[:,2,:,:] = (images[:,2,:,:] - 0.4465)/0.2010
        # images[:,0,:,:] = (images[:,0,:,:] - 0.5071)/0.2675
        # images[:,1,:,:] = (images[:,1,:,:] - 0.4867)/0.2565
        # images[:,2,:,:] = (images[:,2,:,:] - 0.4408)/0.2761
    return images

def imagenet_normalize(images, size=None):
    if size == None:
        images[:,0,:,:] = (images[:,0,:,:] - 0.485)/0.229
        images[:,1,:,:] = (images[:,1,:,:] - 0.456)/0.224
        images[:,2,:,:] = (images[:,2,:,:] - 0.406)/0.225
    elif size == 32:
        images[:,0,:,:] = (images[:,0,:,:] - 0.4810)/0.2146
        images[:,1,:,:] = (images[:,1,:,:] - 0.4574)/0.2104
        images[:,2,:,:] = (images[:,2,:,:] - 0.4078)/0.2138

    return images

# layer_nr = layer_i
layer_nr = int(args.nr)

print("layer_nr", layer_nr)

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

if not net == 'cif10vgg':

    def get_layer_feature_maps(activation_dict, act_layer_list):
        act_val_list = []
        for it in act_layer_list:
            act_val = activation_dict[it]
            act_val_list.append(act_val)
        return act_val_list




    layer_name =  layer_name_cif10

    fourier_act_layers_more = [
                        'conv3_0_relu_1', 'conv3_0_relu_4', 'conv3_1_relu_1',
                        'conv3_1_relu_4', 'conv3_2_relu_1', 'conv3_2_relu_4',
                        'conv4_0_relu_1', 'conv4_0_relu_4', 'conv4_1_relu_1',
                        'conv4_1_relu_4', 'conv4_2_relu_1', 'conv4_2_relu_4'
                        ]


    fourier_act_layers = [ 'conv4_0_relu_1', 'conv4_0_relu_4', 'conv4_1_relu_1',
                        'conv4_1_relu_4', 'conv4_2_relu_1', 'conv4_2_relu_4']

    fourier_act_layers = [ 'conv2_0_relu_1', 'conv2_0_relu_4', 'conv2_1_relu_1',
                        'conv2_1_relu_4', 'conv2_2_relu_1', 'conv2_2_relu_4', 
                        'conv2_3_relu_1', 'conv2_3_relu_4']

    last_layer = [     'relu'      ]

    # model.relu.register_forward_hook(get_activation('relu_0'))

    # model.layer1[0].relu.register_forward_hook(get_activation('layer_1_0_relu'))
    # model.layer1[1].relu.register_forward_hook(get_activation('layer_1_1_relu'))
    # model.layer1[2].relu.register_forward_hook(get_activation('layer_1_2_relu'))

    # model.layer2[0].relu.register_forward_hook(get_activation('layer_2_0_relu'))
    # model.layer2[1].relu.register_forward_hook(get_activation('layer_2_1_relu'))
    # model.layer2[2].relu.register_forward_hook(get_activation('layer_2_2_relu'))

    # # layers = ['layer_1_0_relu', 'layer_1_1_relu', 'layer_1_2_relu', 'layer_2_0_relu', 'layer_2_1_relu', 'layer_2_2_relu']
    # layers = ['layer_1_0_relu', 'layer_1_1_relu', 'layer_1_2_relu', 'layer_2_0_relu']
    # # layers = ['layer_1_0_relu',  'layer_1_2_relu']
    # # layers = [ 'layer_1_2_relu']

    # import pdb; pdb.set_trace()

    ###################
    # model.conv2[0].residual[1].register_forward_hook(get_activation('conv2_0_relu_1'))
    # model.conv2[0].residual[4].register_forward_hook(get_activation('conv2_0_relu_4'))

    # model.conv2[1].residual[1].register_forward_hook(get_activation('conv2_1_relu_1'))
    # model.conv2[1].residual[4].register_forward_hook(get_activation('conv2_1_relu_4'))

    # model.conv2[2].residual[1].register_forward_hook(get_activation('conv2_2_relu_1'))
    # model.conv2[2].residual[4].register_forward_hook(get_activation('conv2_2_relu_4'))

    # model.conv2[3].residual[1].register_forward_hook(get_activation('conv2_3_relu_1'))
    # model.conv2[3].residual[4].register_forward_hook(get_activation('conv2_3_relu_4'))
    #######################

    # model.conv3[0].residual[1].register_forward_hook(get_activation('conv3_0_relu_1'))
    # model.conv3[0].residual[4].register_forward_hook(get_activation('conv3_0_relu_4'))

    # model.conv3[1].residual[1].register_forward_hook(get_activation('conv3_1_relu_1'))
    # model.conv3[1].residual[4].register_forward_hook(get_activation('conv3_1_relu_4'))

    # model.conv3[2].residual[1].register_forward_hook(get_activation('conv3_2_relu_1'))
    # model.conv3[2].residual[4].register_forward_hook(get_activation('conv3_2_relu_4'))

    # model.conv3[3].residual[1].register_forward_hook(get_activation('conv3_3_relu_1'))
    # model.conv3[3].residual[4].register_forward_hook(get_activation('conv3_3_relu_4'))


    # model.conv4[0].residual[1].register_forward_hook(get_activation('conv4_0_relu_1'))
    # model.conv4[0].residual[4].register_forward_hook(get_activation('conv4_0_relu_4'))

    # model.conv4[1].residual[1].register_forward_hook(get_activation('conv4_1_relu_1'))
    # model.conv4[1].residual[4].register_forward_hook(get_activation('conv4_1_relu_4'))

    # model.conv4[2].residual[1].register_forward_hook(get_activation('conv4_2_relu_1'))
    # model.conv4[2].residual[4].register_forward_hook(get_activation('conv4_2_relu_4'))

    model.conv4[3].residual[1].register_forward_hook(get_activation('conv4_3_relu_1'))
    model.conv4[3].residual[4].register_forward_hook(get_activation('conv4_3_relu_4'))

    # model.relu.register_forward_hook(get_activation('relu'))


    # layers = ['conv2_0_relu_1', 'conv2_0_relu_4', 'conv2_1_relu_1', 'conv2_1_relu_4', 'conv2_2_relu_4', 'conv2_3_relu_1']

    # layers = ['conv2_0_relu_1', 'conv2_1_relu_1']
    layers = ['conv4_3_relu_1', 'conv4_3_relu_4']


    # layers = ['conv2_0_relu_1', 'conv2_0_relu_4', 'conv2_1_relu_1', 'conv2_1_relu_4', 'conv2_2_relu_4', 'conv2_3_relu_1']

    print(layers)
    # layers = ['layer_1_0_relu', 'layer_1_1_relu', 'layer_1_2_relu', 'layer_2_0_relu']


    if layer_nr == 0:
        layers = ['conv2_0_relu_1', 'conv2_0_relu_4']
    elif layer_nr == 1:
        layers = ['conv2_1_relu_1', 'conv2_1_relu_4']
    elif layer_nr == 2:
        layers = ['conv2_2_relu_1', 'conv2_2_relu_4']
    elif layer_nr == 3:
        layers = ['conv2_3_relu_1', 'conv2_3_relu_4']
    elif layer_nr == 4:
        layers = ['conv3_0_relu_1', 'conv3_0_relu_4']
    elif layer_nr == 5:
        layers = ['conv3_1_relu_1', 'conv3_1_relu_4']
    elif layer_nr == 6:
        layers = ['conv3_2_relu_1', 'conv3_2_relu_4']
    elif layer_nr == 7:
        layers = ['conv3_3_relu_1', 'conv3_3_relu_4']
    elif layer_nr == 8:
        layers = ['conv4_0_relu_1', 'conv4_0_relu_4']
    elif layer_nr == 9:
        layers = ['conv4_1_relu_1', 'conv4_1_relu_4']
    elif layer_nr == 10:
        layers = ['conv4_2_relu_1', 'conv4_2_relu_4']
    elif layer_nr == 11:
        layers = ['conv4_3_relu_1', 'conv4_3_relu_4']
    elif layer_nr == 12:
        layers = ['relu']
    else:
        print("error number > 12")

else:

    layer_name = layer_name_cif10vgg

    # indice of activation layers
    layers = [0, 1]
    act_layers= [2,5,9,12,16,19,22,26,29,32,36,39,42]
    fourier_act_layers = [9,16,22,29,36,42]
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
        print("error number > 12")


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
    if net == 'cif100' and (attack_method=='cw' or attack_method=='df'):
        fourier_spectrum *= 1/np.max(fourier_spectrum)
    return fourier_spectrum


def calculate_spectra(images, typ='MFS'):
    fs = []   
    for i in range(len(images)):
        image = images[i]
        fourier_image = calculate_fourier_spectrum(image, typ=typ)
        fs.append(fourier_image.flatten())
    return fs


###Fourier Input
print('Extracting ' + detector+' characteristic...')
if detector == 'InputMFS':
    mfs = calculate_spectra(images)
    mfs_advs = calculate_spectra(images_advs)

    characteristics       = np.asarray(mfs, dtype=np.float32)
    characteristics_adv   = np.asarray(mfs_advs, dtype=np.float32)
    
elif detector == 'InputPFS':
    pfs = calculate_spectra(images, typ='PFS')
    pfs_advs = calculate_spectra(images_advs, typ='PFS')
        
    characteristics       = np.asarray(pfs, dtype=np.float32)
    characteristics_adv   = np.asarray(pfs_advs, dtype=np.float32)

###Fourier Layer   
elif detector == 'LayerMFS':
    mfs = []
    mfs_advs = []

    if args.nr == -1:
        if net == 'cif100' and (attack_method=='cw' or attack_method=='df'):
            layers =  last_layer # [42]
        elif net == 'imagenet' or  net == 'imagenet32' or  net == 'imagenet64' or  net == 'imagenet128':
            pass
        elif net == 'cif10' or net == 'cif10vgg':
            pass
        elif net == 'celebaHQ32' or net == 'celebaHQ64' or net == 'celebaHQ128':
            pass
        else:
            layers = fourier_act_layers

    for i in tqdm(range(number_images)):
        image = images[i].unsqueeze_(0)
        adv = images_advs[i].unsqueeze_(0)

        if net == 'imagenet' or net == 'imagenet32' or  net == 'imagenet64' or net == 'imagenet128':
            image = imagenet_normalize(image, size=32)
            adv   = imagenet_normalize(adv, size=32)

        if net == 'cif10' or net == 'cif100' or net == 'cif10vgg':
            image = cifar_normalize(image)
            adv   = cifar_normalize(adv)

        if not net == 'cif10vgg':

            if args.nr == -1:
                feat_img = model(image.cuda())
                image_feature_maps = [image] + get_layer_feature_maps(activation, layers)
                feat_adv = model(adv.cuda())
                adv_feature_maps   = [adv]   + get_layer_feature_maps(activation, layers)
            else:
                feat_img = model(image.cuda())
                image_feature_maps = get_layer_feature_maps(activation, layers)
                feat_adv = model(adv.cuda())
                adv_feature_maps   = get_layer_feature_maps(activation, layers)
        else:
            image_feature_maps = get_layer_feature_maps(image.cuda(), layers)
            adv_feature_maps   = get_layer_feature_maps(adv.cuda(), layers)

        fourier_maps     = calculate_spectra(image_feature_maps)
        fourier_maps_adv = calculate_spectra(adv_feature_maps)
        mfs.append(np.hstack(fourier_maps))
        mfs_advs.append(np.hstack(fourier_maps_adv))

    if not args.nr == -1:
        nr_param = 1
        for i in image_feature_maps[0].shape:
            nr_param = nr_param * i
        print("parameters: ", image_feature_maps[0].shape, nr_param)
        
    characteristics       = np.asarray(mfs,      dtype=np.float32)
    characteristics_adv   = np.asarray(mfs_advs, dtype=np.float32)
    
elif detector == 'LayerPFS':

    pfs = []
    pfs_advs = []

    if args.nr == -1:
        if net == 'cif100' and (attack_method=='cw' or attack_method=='df'):
            layers =  last_layer # [42]
        elif net == 'imagenet' or  net == 'imagenet32' or  net == 'imagenet64' or  net == 'imagenet128':
            pass
        elif net == 'cif10' or net == 'cif10vgg':
            pass
        elif net == 'celebaHQ32' or net == 'celebaHQ64' or net == 'celebaHQ128':
            pass
        else:
            layers = fourier_act_layers

    for i in tqdm(range(number_images)):
        image = images[i].unsqueeze_(0)
        adv = images_advs[i].unsqueeze_(0)

        if net == 'imagenet' or net == 'imagenet32' or  net == 'imagenet64' or net == 'imagenet128':
            image = imagenet_normalize(image, size=32)
            adv   = imagenet_normalize(adv, size=32)

        if net == 'cif10' or net == 'cif100' or net == 'cif10vgg':
            image = cifar_normalize(image)
            adv   = cifar_normalize(adv)

        if not net == 'cif10vgg':

            if args.nr == -1:
                feat_img = model(image.cuda())
                image_feature_maps = [image] + get_layer_feature_maps(activation, layers)
                feat_adv = model(adv.cuda())
                adv_feature_maps   = [adv]   + get_layer_feature_maps(activation, layers)
            else:
                feat_img = model(image.cuda())
                image_feature_maps = get_layer_feature_maps(activation, layers)
                feat_adv = model(adv.cuda())
                adv_feature_maps   = get_layer_feature_maps(activation, layers)
        else:
            image_feature_maps = get_layer_feature_maps(image.cuda(), layers)
            adv_feature_maps   = get_layer_feature_maps(adv.cuda(), layers)

        fourier_maps     = calculate_spectra(image_feature_maps, typ='PFS')
        fourier_maps_adv = calculate_spectra(adv_feature_maps,   typ='PFS')

        pfs.append(np.hstack(fourier_maps))
        pfs_advs.append(np.hstack(fourier_maps_adv))

    if not args.nr == -1:
        nr_param = 1
        for i in image_feature_maps[0].shape:
            nr_param = nr_param * i
        print("parameters: ", image_feature_maps[0].shape, nr_param)
        
    characteristics       = np.asarray(pfs, dtype=np.float32)
    characteristics_adv   = np.asarray(pfs_advs, dtype=np.float32)



#######LID section
elif detector == 'LID':
    
    # layers = fourier_act_layers

    #hyperparameters
    batch_size = 100
    if net == 'cif10':
        k = 20
    else:
        k = 10
    
    def mle_batch(data, batch, k):
        data = np.asarray(data, dtype=np.float32)
        batch = np.asarray(batch, dtype=np.float32)
        k = min(k, len(data)-1)
        f = lambda v: - k / np.sum(np.log(v/v[-1]))
        a = cdist(batch, data)
        a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
        a = np.apply_along_axis(f, axis=1, arr=a)
        return a

    lid_dim = len(layers)
    # lid_dim = len(act_layers)

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

        if NORMALIZED:
            batch = cifar_normalize(batch)
            batch_adv = cifar_normalize(batch_adv)
        # X_act       = get_layer_feature_maps(batch.to(device), act_layers)
        # X_adv_act   = get_layer_feature_maps(batch_adv.to(device), act_layers)

        feat_img = model(batch.cuda())
        X_act = get_layer_feature_maps(activation, layers)

        feat_adv = model(batch_adv.cuda())
        X_adv_act = get_layer_feature_maps(activation, layers)

        for i in range(lid_dim):
            X_act[i]       = np.asarray(X_act[i].cpu().detach().numpy()    , dtype=np.float32).reshape((n_feed, -1))
            X_adv_act[i]   = np.asarray(X_adv_act[i].cpu().detach().numpy(), dtype=np.float32).reshape((n_feed, -1))
            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
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

    characteristics       = np.asarray(lids, dtype=np.float32)
    characteristics_adv   = np.asarray(lids_adv, dtype=np.float32)

####### Mahalanobis section
elif detector == 'Mahalanobis':

    act_layers_mah = layers
    if not net == 'imagenet':
        act_layers_mah = fourier_act_layers

    is_sample_mean_calculated = True #set true if sample mean and precision are already calculated

    if net == 'cif10':
        num_classes = 10
    elif net == 'cif100':
        num_classes = 100
    elif net == 'imagenet':
        num_classes = 1000
    
    if not is_sample_mean_calculated:
        print('Calculate sample mean and precision for Mahalanobis...')
        #load cifar10 training data
        if net == 'cif10':
            if NORMALIZED:
                transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            else:
                transform = transforms.Compose([transforms.ToTensor()])
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
            num_classes = 10
        elif net == 'cif100':
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
            num_classes = 100
        elif net == 'imagenet':
            num_classes = 1000

            # transform_list = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
            transform_list = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            transform_chain = transforms.Compose(transform_list)
            data_dir = '/home/lorenzp/datasets/ImageNet3'

            item = datasets.ImageFolder(data_dir + '/train', transform=transform_chain)
            trainloader = data.DataLoader(item, batch_size=100, shuffle=True, num_workers=4)


        data_iter = iter(trainloader)
        im = data_iter.next()
        feature_list=[]
        feat_img = model(im[0].cuda())
        layers = get_layer_feature_maps(activation, act_layers_mah)
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
            with torch.no_grad():
                data = Variable(data)
                out_features = get_layer_feature_maps(activation, act_layers_mah)
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
                        list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
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

        np.save('./data/sample_mean_'+net,sample_class_mean)
        np.save('./data/precision_'+net,precision)
        
    #load sample mean and precision
    print('Loading sample mean and precision...')
    sample_mean = np.load('./data/sample_mean_'+net+'.npy',allow_pickle=True)
    precision   = np.load('./data/precision_'+net+'.npy',allow_pickle=True)    
    
    if net == 'cif10':
        if attack_method == 'fgsm':
            magnitude = 0.0002
        elif attack_method == 'cw':
            magnitude = 0.00001
        else:
            magnitude = 0.00005
    else:
        if attack_method == 'fgsm':
            magnitude = 0.005
        elif attack_method == 'cw':
            magnitude = 0.00001
        elif attack_method == 'df':
            magnitude = 0.0005
        else:
            magnitude = 0.01

    image_loader = torch.utils.data.DataLoader(images, batch_size=100, shuffle=True)
    adv_loader = torch.utils.data.DataLoader(images_advs, batch_size=100, shuffle=True)

    def get_mah(test_loader, layer_index):
        Mahalanobis = []
        for data in test_loader:
            data = cifar_normalize(data)
            data = data.cuda()
            data = Variable(data, requires_grad = True)

            feat_img = model(data)
            out_features = get_layer_feature_maps(activation, [act_layers_mah[layer_index]])[0]

            # out_features = get_layer_feature_maps(data, [act_layers[layer_index]])[0]
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
                feat_img = model(Variable(tempInputs))
                noise_out_features = get_layer_feature_maps(activation, [act_layers_mah[layer_index]])[0]
                # noise_out_features = get_layer_feature_maps(Variable(tempInputs), [act_layers[layer_index]])[0]
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
    print('Calculating Mahalanobis scores...')
    Mah_adv = np.zeros((len(images_advs),len(act_layers_mah)))
    Mah = np.zeros((len(images_advs),len(act_layers_mah)))
    
    for layer_index in tqdm(range(len(act_layers_mah))):
        Mah_adv[:,layer_index]=np.array(get_mah(adv_loader, layer_index))
        Mah[:,layer_index]=np.array(get_mah(image_loader, layer_index))
    
    characteristics = Mah
    characteristics_adv = Mah_adv


elif detector == 'NNIF':
    # assert FLAGS.only_last is True

    ablation = '1111'
    max_indices = -1


    # for ablation:
    sel_column = []
    for i in [0, 1, 2, 3]:
        if ablation[i] == '1':
            sel_column.append(i)

    if max_indices == -1:
        max_indices_vec = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    else:
        max_indices_vec = [max_indices]


    for i in tqdm(range(number_images)):
        image = images[i].unsqueeze_(0)
        adv = images_advs[i].unsqueeze_(0)
        if NORMALIZED:
            image = cifar_normalize(image.cuda())
            adv = cifar_normalize(adv.cuda())
        # image_feature_maps = get_layer_feature_maps(image, layers) 
        # adv_feature_maps = get_layer_feature_maps(adv, layers)
        feat_img =           model(image.cuda())
        image_feature_maps = get_layer_feature_maps(activation, layers)

        feat_adv =           model(adv.cuda())
        adv_feature_maps   = get_layer_feature_maps(activation, layers)


    for max_indices in tqdm(max_indices_vec):
        print('Extracting NNIF characteristics for max_indices={}'.format(max_indices))

        # training the knn layers
        knn_large_trainset = get_knn_layers(X_train, y_train_sparse)
        knn_small_trainset = get_knn_layers(X_train_mini, y_train_mini_sparse)

        # val
        all_normal_ranks, all_normal_dists = calc_all_ranks_and_dists(X_val, 'val', knn_large_trainset)
        all_adv_ranks   , all_adv_dists    = calc_all_ranks_and_dists(X_val_adv, 'val', knn_large_trainset)
        ranks, ranks_adv = get_nnif(X_val, 'val', max_indices)
        ranks     = ranks[:, :, sel_column]
        ranks_adv = ranks_adv[:, :, sel_column]
        characteristics, labels = merge_and_generate_labels(ranks_adv, ranks)
        print("NNIF train: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)
        file_name = 'max_indices_{}_ablation_{}_train'.format(max_indices, FLAGS.ablation)
        file_name = append_suffix(file_name)
        file_name = os.path.join(characteristics_dir, file_name)
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)
        end_val = time.time()
        print('total feature extraction time for val: {} sec'.format(end_val - start))

        # test
        all_normal_ranks, all_normal_dists = calc_all_ranks_and_dists(X_test, 'test', knn_small_trainset)
        all_adv_ranks   , all_adv_dists    = calc_all_ranks_and_dists(X_test_adv, 'test', knn_small_trainset)
        ranks, ranks_adv = get_nnif(X_test, 'test', max_indices)
        ranks[:, :, 0] *= (49/5)  # The mini train set contains only 5k images, not 49k images as in the train set
        ranks[:, :, 2] *= (49/5)  # Therefore, the ranks (both helpful and harmful) are scaled.
        ranks_adv[:, :, 0] *= (49/5)
        ranks_adv[:, :, 2] *= (49/5)
        ranks     = ranks[:, :, sel_column]
        ranks_adv = ranks_adv[:, :, sel_column]
        characteristics, labels = merge_and_generate_labels(ranks_adv, ranks)
        print("NNIF test: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)
        file_name = 'max_indices_{}_ablation_{}_test'.format(max_indices, FLAGS.ablation)
        file_name = append_suffix(file_name)
        file_name = os.path.join(characteristics_dir, file_name)
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)
        end_test = time.time()
        print('total feature extraction time for test: {} sec'.format(end_test - end_val))

else:
    print('unknown detector')


# Save
print("save extracted characteristics ...")
if attack_method == 'apgd-ce' or attack_method == 'apgd-t' or attack_method == 'fab-t' or attack_method == 'square':
    filename = './data/characteristics/0.03137/' + net + '/ind/' + net + '_' + attack_method + '_' + detector
elif attack_method == 'std':
    path_dir = './data/characteristics/' + net + '/std/'
    filename = path_dir + net + '_' + attack_method + '_' + detector + '_{}_eps_{:5f}'.format(nr_samples, args.eps)
    if not args.nr == -1:
        filename = filename + '_' + layer_name[layer_nr]
elif not args.nr == -1:
    path_dir = './data/characteristics/' + net + '/'
    filename = path_dir + net + '_' + attack_method + '_' + detector+'_'+layer_name[layer_nr]
else:
    path_dir = './data/characteristics/' + net + '/'
    filename = path_dir + net + '_' + attack_method + '_' + detector


if not os.path.exists(path_dir):
    os.mkdir(path_dir)

filename1 = filename + appendix
print(filename1)
if os.path.exists(filename1 + '.p'):
    os.remove(filename1 + '.p')
pickle.dump( characteristics, open( filename1 + '.p', "wb" ) )

filename2 = filename1 + '_adv' + '.p'
print(filename2)
if os.path.exists(filename2):
    os.remove(filename2)
pickle.dump( characteristics_adv, open(filename2, "wb" ) )


print('Done extracting and saving characteristics!')