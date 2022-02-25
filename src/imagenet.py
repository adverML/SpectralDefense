#!/usr/bin/env python3
import pdb
import os, sys
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
import torchvision.models as models

def clean_accuracy(data_loader, model, wanted_samples=2000):
    clean_dataset = []; correct = 0; total = 0; i = 0

    for images, labels in data_loader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if (predicted == labels):
            clean_dataset.append([images, labels])
        i = i + 1
        if i % 500 == 0:
            acc = (wanted_samples, i, 100 * correct / total)
            print('INFO: Accuracy of the network on the %d test images: %d, %d %%' % acc)

        if len(clean_dataset) >= wanted_samples:
            break

model = models.wide_resnet50_2(pretrained=True)
model.cuda(); model.eval()

mean = [0.485, 0.456, 0.406] # https://pytorch.org/vision/stable/models.html#wide-resnet
std  = [0.229, 0.224, 0.225]

normalization = [transforms.Normalize(mean=mean, std=std)]
transform_list = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()] + normalization)
IMAGENET_PATH  = "/home/DATA/ITWM/ImageNet" # imagenet 2012
TRAIN_PATH     = os.path.join(IMAGENET_PATH, 'train') # contains subfolders
TEST_PATH      = os.path.join(IMAGENET_PATH, 'val') # contains subfolders 50k samples

train_loader  = torch.utils.data.DataLoader(datasets.ImageFolder(TRAIN_PATH , transform_list), batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
test_loader   = torch.utils.data.DataLoader(datasets.ImageFolder(TEST_PATH, transform_list), batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

clean_accuracy(test_loader, model, wanted_samples=2000)
clean_accuracy(train_loader, model, wanted_samples=2000)

