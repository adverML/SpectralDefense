

import argparse
import torch
import torch.nn as nn
from detection import data_loader
import numpy as np
from detection import models
import matplotlib.pyplot as plt
import os
# import detection.jojo.adversary as adversary
# from detection.lib import attacks as DeltaAttack
import pdb
from torchvision import transforms
from torch.autograd import Variable
from detection import lib_generation


vae1_pth =    "/home/lorenzp/adversialml/src/submodules/CD-VAE/pretrained/cd-vae-1.pth"
vae2_pth =    "/home/lorenzp/adversialml/src/submodules/CD-VAE/pretrained/cd-vae-2.pth"
wrn2810_pth = "/home/lorenzp/adversialml/src/submodules/CD-VAE/pretrained/wide_resnet.pth"

batch_size = 200
dataset = 'cifar10'
dataroot = '/home/lorenzp/DATA/ITWM/cifar10'
argoutf = '/home/lorenzp/adversialml/src/submodules/CD-VAE/detection/data/cd-vae-1/'
num_classes = 10
net_type = 'resnet'
gpu = 0
adv_type = 'FGSM'
vae_path = vae2_pth