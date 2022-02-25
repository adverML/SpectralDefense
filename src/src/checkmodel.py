print('Load modules...')
import numpy as np
import pickle
import torch
import sys, os
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# from models.resnet_sota import ResNet34_SOTA
from models.resnet import  resnet34

def load_model_checkpoint(model, model_path, epoch=None):
    # model_path = get_model_file(model_type, epoch=epoch)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        raise InputError("Saved model checkpoint '{}' not found.".format(model_path))

    return model


# normalization = NORMALIZE_IMAGES['cifar10']

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(*NORMALIZE_IMAGES['cifar10'])
    ]
)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, )
num_classes = 10

model = ResNet34(normalization=None).to('cuda')
# model = resnet34(num_classes=10, preprocessing={}).to('cuda')

model = load_model_checkpoint(model, '/home/lorenzp/adversialml/src/src/submodules/adversarial-detection/expts/models/cifar10_cnn.pt')

import pdb; pdb.set_trace()