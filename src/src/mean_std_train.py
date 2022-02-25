import os
import torch
import torchvision
import pdb

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from dataset import CelebaDataset

import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import natsort

import cv2

import decimal
decimal.getcontext().prec=50

# from skimage.io import imread

# if torch.cuda.is_available():
#     torch.backends.cudnn.deterministic = True

# DATA_SPLIT = 80
DATA_SPLIT = '70'
# DATA_SPLIT = 60

# IMAGE_SIZE = '1024x1024'
# IMAGE_SIZE = '512x512'
# IMAGE_SIZE = '256x256'
# IMAGE_SIZE = '128x128'
# IMAGE_SIZE = '64x64'
# IMAGE_SIZE = '32x32'

LIST_IMAGE_SIZE = ['32x32', '64x64', '128x128', '256x256', '512x512', '1024x1024']

batch_size = 1

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

# my_dataset = CustomDataSet('/home/lorenzp/datasets/CelebAHQ/Img/hq/data'+IMAGE_SIZE+'/', transform=transforms.ToTensor())
# loader = DataLoader(my_dataset , batch_size=batch_size, shuffle=False, 
#                                num_workers=4, drop_last=True)

for IMAGE_SIZE in LIST_IMAGE_SIZE:

    train_dataset = CelebaDataset(
                            csv_path = '/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-train_hair_color_hq_ext_' + DATA_SPLIT + '.csv',
                            # csv_path='/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-train_smiling_hq_' + DATA_SPLIT + '.csv',
                            img_dir='/home/lorenzp/datasets/CelebAHQ/Img/hq/data' + IMAGE_SIZE + '/',
                            data='Hair_Color',
                            # data='Smiling',
                            transform=transforms.ToTensor())

    loader = DataLoader(dataset=train_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=4)

    mean = 0.
    meansq = 0.
    for data in loader:

        data = data[0]
        # pdb.set_trace()

        mean_r = data[0][0].mean()
        mean_g = data[0][1].mean()
        mean_b = data[0][2].mean()

        meansq_r = (data[0][0]**2).mean()
        meansq_g = (data[0][1]**2).mean()
        meansq_b = (data[0][2]**2).mean()

    std_r = torch.sqrt(meansq_r - mean_r**2)
    std_g = torch.sqrt(meansq_g - mean_g**2)
    std_b = torch.sqrt(meansq_b - mean_b**2)

    # pdb.set_trace()

    print("IMAGE_SIZE ", IMAGE_SIZE)
    print("mean: " + '(' + str(meansq_r.item()) + ', ' + str(meansq_g.item()) + ', ' + str(meansq_b.item()) + ')' )
    print("std: " + '(' + str(std_r.item()) + ', ' + str(std_g.item()) + ', ' + str(std_b.item())+ ')')
    

# /home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-train_smiling_hq_70.csv
# /home/lorenzp/datasets/CelebAHQ/Img/hq/data32x32/
# IMAGE_SIZE  32x32
# mean: 0.36015135049819946 0.21252931654453278 0.1168241947889328
# std: 0.24773411452770233 0.20017878711223602 0.17963241040706635
# /home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-train_smiling_hq_70.csv
# /home/lorenzp/datasets/CelebAHQ/Img/hq/data64x64/
# IMAGE_SIZE  64x64
# mean: 0.3610851764678955 0.2132178544998169 0.11681009083986282
# std: 0.24751928448677063 0.19908452033996582 0.17821228504180908
# /home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-train_smiling_hq_70.csv
# /home/lorenzp/datasets/CelebAHQ/Img/hq/data128x128/
# IMAGE_SIZE  128x128
# mean: 0.3617511987686157 0.21352902054786682 0.11670646071434021
# std: 0.24808333814144135 0.19945485889911652 0.17840421199798584
# /home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-train_smiling_hq_70.csv
# /home/lorenzp/datasets/CelebAHQ/Img/hq/data256x256/
# IMAGE_SIZE  256x256
# mean: 0.3618541657924652 0.21353766322135925 0.1166912168264389
# std: 0.24811546504497528 0.19950546324253082 0.17830605804920197
# /home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-train_smiling_hq_70.csv
# /home/lorenzp/datasets/CelebAHQ/Img/hq/data512x512/
# IMAGE_SIZE  512x512
# mean: 0.3620170056819916 0.21373045444488525 0.11688751727342606
# std: 0.24830812215805054 0.19977152347564697 0.17850320041179657
# /home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-train_smiling_hq_70.csv
# /home/lorenzp/datasets/CelebAHQ/Img/hq/data1024x1024/
