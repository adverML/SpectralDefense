#!/usr/bin/env python3
""" Generate Clean Data

author Peter Lorenz
"""

#this script extracts the correctly classified images
print('Load modules...')
import pdb
import os
import json
from conf import settings
import argparse
import datetime
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets

from utils import *

from models.vgg_cif10 import VGG
from models.wideresidual import WideResNet, WideBasic

from datasets import smallimagenet

parser = argparse.ArgumentParser()
parser.add_argument("--net",            default='imagenet32',     help=settings.HELP_NET)
parser.add_argument("--img_size",       default='32',   type=int, help=settings.HELP_IMG_SIZE)
parser.add_argument("--batch_size",     default='1',    type=int, help=settings.HELP_BATCH_SIZE)
parser.add_argument("--num_classes",    default='1000', type=int, help=settings.HELP_NUM_CLASSES)
parser.add_argument("--wanted_samples", default='2000', type=int, help=settings.HELP_WANTED_SAMPLES)

args = parser.parse_args()
print("args: ", args)

if not args.batch_size == 1:
    get_debug_info(msg='Err: Batch size must be always 1!')
    assert True

appendix = get_appendix(args.num_classes, settings.MAX_CLASSES_IMAGENET)

net, depth, widen_factor = get_model_info(args)
clean_data_filename = net + '_' + str(depth) + '_' + str(widen_factor) + appendix 
clean_data_path = './data/clean_data/' + args.net + os.sep + clean_data_filename
get_debug_info(msg='Info: clean_data_path: ' + clean_data_path)
existed = make_dir(clean_data_path)
# if existed:
#     input("Directory already Exists! Do you want to continue?")

save_args_to_file(args, clean_data_path)
logger = Logger(clean_data_path + os.sep + 'log.txt')
logger.log(datetime.datetime.now())
logger.log(args.__dict__)
logger.log(clean_data_path)

logger.log('Load model...')

model, preprocessing = load_model(args)
model.cuda()
model.eval()


testloader            = load_test_set(args, preprocessing=None)
testloader_normalized = load_test_set(args, preprocessing=preprocessing)


data_iter = iter(testloader)
clean_dataset = []
correct = 0
total = 0
i = 0

logger.log('Classify images...')
for images, labels in testloader_normalized:
    data = data_iter.next()
    images = images.cuda()
    labels = labels.cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)

    correct += (predicted == labels).sum().item()
    if (predicted == labels):
        clean_dataset.append(data)
    i = i + 1

    if i % 500 == 0:
        logger.log('Accuracy of the network on the %d test images: %d, %d %%' % (args.wanted_samples, i, 100 * correct / total))

    if len(clean_dataset) > args.wanted_samples:
        break


# path = './data/clean_data_' + args.net + '_' + str(depth) + '_' + str(widen_factor) + appendix
# print("path: ", path, ", len(clean_dataset)", len(clean_dataset))

logger.log("clean_data_path: " + clean_data_path + ", len(clean_dataset) " + str(len(clean_dataset)) )

torch.save(clean_dataset, clean_data_path + os.sep + 'clean_data')
logger.log('Done extracting and saving correctly classified images!')