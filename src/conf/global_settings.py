""" configurations for this project

from author baiyu
extended by Peter Lorenz
"""

import os
from datetime import datetime

SHOW_DEBUG = True

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10

# Classes for each dataset
MAX_CLASSES_IMAGENET = 1000
MAX_CLASSES_MNIST    = 10
MAX_CLASSES_CIF10    = 10
MAX_CLASSES_CIF100   = 100
MAX_CLASSES_CELEBAHQ = 4

# Help Parameters Info
HELP_ATTACK = "the attack method you want to use in order to create adversarial examples. Either fgsm, bim, pgd, df, cw, std (for AutoAttack: apgd-ce, apgd-t, fab-t and square.), or aa+"
HELP_NET = "the dataset the net was trained on, either mnist or cif10 or cif10vgg or cif100 or imagenet, imagenet32, imagenet64, imagenet128, celebaHQ32, celebaHQ64, celebaHQ128"
HELP_NUM_CLASSES = "default: 1000; {10, 25, 50, 100, 250} only for imagenet32; 1000 classes for imagenet"
HELP_WANTED_SAMPLES = "nr of samples to process"
HELP_LAYER_NR =    "Only for WhiteBox when you want to extract from a specific layer"
HELP_DETECTOR =    "The detector your want to use, out of InputMFS, InputPFS, LayerMFS, LayerPFS, LID, Mahalanobis"
HELP_IMG_SIZE =    "Default: 32x32 pixels; For Imagenet and CelebaHQ"
HELP_MODE =        "Choose test or validation case"
HELP_BATCH_SIZE =  "Number of batches per epoch"
HELP_NET_NORMALIZATION = "Instead of data normalizeion. The data is normalized within the NN."
HELP_CLF =         "LR or RF; Logistic Regresion or Random Forest"
HELP_AA_EPSILONS = "epsilon: 8./255. 4/255, 3/255, 2/255, 1/255, 0.5/255"

# Warnings
WARN_DIR_EXISTS = "Directory already Exists! Do you want to continue?"

# CSV Paths
CELEBA_CSV_PATH = 'submodules/pytorch_ipynb/cnn/celeba-' # test_

# Weight Paths
MNIST_CKPT = '/home/lorenzp/adversialml/src/pytorch-classification/checkpoint/model_best.pth.tar'

CIF10_CKPT      = '/home/lorenzp/wide-resnet.pytorch/checkpoint_wrn/cifar10/wide-resnet-28x10_2022-04-17_14:12:50.pt' 
CIF10_M_CKPT    =  './submodules/CD-VAE/pretrained/cd-vae-1.pth'

CIF100_CKPT     = '/home/lorenzp/wide-resnet.pytorch/checkpoint_wrn/cifar100/wide-resnet-28x10_2022-04-17_14:13:21.pt'

CIF10VGG_CKPT   = './checkpoint/vgg16/vgg_cif10.pth'
CIF10VGG_CKPT_NEW   = '/home/scratch/adversarialml/pytorch-vgg-cifar10/save_vgg16/checkpoint_299.tar'
CIF100VGG_CKPT  = './checkpoint/vgg16/vgg_cif100.pth'
CIF10RN34_CKPT  = './submodules/pytorch-CelebAHQ/checkpoint/resnet_34/resnet_34_ckpt_test.pth'
CIF100RN34_CKPT = './submodules/pytorch-CelebAHQ/checkpoint/cif100_resnet_34/resnet_34_ckpt_test.pth'

CIF10RN34_SOTA_CKPT = './submodules/adversarial-detection/expts/models/cifar10_cnn.pt'

IMAGENET32_CKPT_1000   = './submodules/pytorch-classification/checkpoints/imagenet32/wideresent2810/model_best.pth.tar' # model_best.pth.tar
IMAGENET32_CKPT_250    = './submodules/pytorch-classification/checkpoints/imagenet32/wideresent2810_250/model_best.pth.tar'
IMAGENET32_CKPT_100    = './submodules/pytorch-classification/checkpoints/imagenet32/wideresent2810_100/model_best.pth.tar'
IMAGENET32_CKPT_75     = './submodules/pytorch-classification/checkpoints/imagenet32/wideresent2810_75/model_best.pth.tar'
IMAGENET32_CKPT_50     = './submodules/pytorch-classification/checkpoints/imagenet32/wideresent2810_50/model_best.pth.tar'
IMAGENET32_CKPT_25     = './submodules/pytorch-classification/checkpoints/imagenet32/wideresent2810_25/model_best.pth.tar'
IMAGENET32_CKPT_10     = './submodules/pytorch-classification/checkpoints/imagenet32/wideresent2810_10/model_best.pth.tar'

IMAGENET64_CKPT_1000   = './submodules/pytorch-classification/checkpoints/imagenet64/wideresent2810/model_best.pth.tar'
IMAGENET128_CKPT_1000  = './submodules/pytorch-classification/checkpoints/imagenet128/wideresent2810/model_best.pth.tar'

CELEBAHQ32_CKPT_2   = './submodules/pytorch-CelebAHQ/checkpoint/wrn2810/32x32_128_0.1_Smiling_Thursday_30_September_2021_11h_01m_19s/wrn2810-161-best.pth'
CELEBAHQ64_CKPT_2   = './submodules/pytorch-CelebAHQ/checkpoint/wrn2810/64x64_128_0.1_Smiling_Thursday_30_September_2021_15h_35m_05s/wrn2810-141-best.pth'
CELEBAHQ128_CKPT_2  = './submodules/pytorch-CelebAHQ/checkpoint/wrn2810/128x128_64_0.1_Smiling_Thursday_30_September_2021_15h_37m_55s/wrn2810-140-best.pth'

CELEBAHQ32_CKPT_4   = './submodules/pytorch-CelebAHQ/checkpoint/wrn2810/32x32_64_0.1_Hair_Color_Thursday_04_November_2021_14h_35m_14s/wrn2810-200-best.pth'
CELEBAHQ64_CKPT_4   = './submodules/pytorch-CelebAHQ/checkpoint/wrn2810/64x64_64_0.1_Hair_Color_Thursday_04_November_2021_17h_25m_16s/wrn2810-171-best.pth'
CELEBAHQ128_CKPT_4  = './submodules/pytorch-CelebAHQ/checkpoint/wrn2810/128x128_64_0.1_Hair_Color_Thursday_04_November_2021_17h_38m_53s/wrn2810-100-regular.pth' # 89%
CELEBAHQ256_CKPT_4  = './submodules/pytorch-CelebAHQ/checkpoint/wrn2810/256x256_24_0.1_Hair_Color_Friday_05_November_2021_16h_44m_36s/wrn2810-70-regular.pth'    # 78%


# Dataset Paths
MNIST_PATH   = "/home/DATA/ITWM/mnist"
CIF10_PATH   = "/home/DATA/ITWM/cifar10"
CIF100_PATH  = "/home/DATA/ITWM/cifar100"
RESTRICTED_IMAGENET_PATH = "/home/DATA/ITWM/Restricted_ImageNet"
IMAGENET_HIERARCHY_PATH  = "/home/DATA/ITWM/ImageNetHierarchy"
IMAGENET_PATH    = "/home/DATA/ITWM/ImageNet"
IMAGENET32_PATH  = "/home/DATA/ITWM/Imagenet32x32"
IMAGENET64_PATH  = "/home/DATA/ITWM/Imagenet64x64"
IMAGENET128_PATH = "/home/DATA/ITWM/Imagenet128x128"
IMAGENET240_PATH = "/home/DATA/ITWM/Imagenet240x240"
CELEBAHQ32_PATH  = "/home/DATA/ITWM/CelebAHQ/Img/hq/data32x32"
CELEBAHQ64_PATH  = "/home/DATA/ITWM/CelebAHQ/Img/hq/data64x64"
CELEBAHQ128_PATH = "/home/DATA/ITWM/CelebAHQ/Img/hq/data128x128"
CELEBAHQ256_PATH = "/home/DATA/ITWM/CelebAHQ/Img/hq/data256x256"


# Extact Features 
ISSAMPLEMEANCALCULATED = False # True # Mahalannobis; set true if sample mean and precision are already calculated

###########################################################################
# Detect Adversarials

TRTE = False
TRAINERR = False
SELECTED_COL = ['asr', 'auc',  'f1',  'acc', 'pre', 'tpr', 'tnr', 'fnr', 'asrd']
ATTACKS_LIST = ['fgsm', 'bim', 'std', 'pgd', 'df', 'cw'] 
DETECTOR_LIST_LAYERS = ['LayerMFS', 'LayerPFS']
DETECTOR_LIST = [ 'InputMFS', 'LayerMFS' ]
CLF = ['LR', 'RF']


SAVE_CLASSIFIER = True