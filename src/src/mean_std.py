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

IMAGE_SIZE = '1024x1024'
# IMAGE_SIZE = '512x512'
# IMAGE_SIZE = '256x256'
# IMAGE_SIZE = '128x128'
# IMAGE_SIZE = '64x64'
# IMAGE_SIZE = '32x32'

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

my_dataset = CustomDataSet('/home/lorenzp/datasets/CelebAHQ/Img/hq/data'+IMAGE_SIZE+'/', transform=transforms.ToTensor())
loader = DataLoader(my_dataset , batch_size=batch_size, shuffle=False, 
                               num_workers=4, drop_last=True)

mean = 0.
meansq = 0.
for data in loader:
    mean_r = data[0][0].mean()
    mean_g = data[0][1].mean()
    mean_b = data[0][2].mean()

    meansq_r = (data[0][0]**2).mean()
    meansq_g = (data[0][1]**2).mean()
    meansq_b = (data[0][2]**2).mean()

std_r = torch.sqrt(meansq_r - mean_r**2)
std_g = torch.sqrt(meansq_g - mean_g**2)
std_b = torch.sqrt(meansq_b - mean_b**2)

print("mean: " + str(meansq_r) + ' ' + str(meansq_g) + ' ' + str(meansq_b))
print("std: " + str(std_r) + ' ' + str(std_g) + ' ' + str(std_b))
print()

# dataset = datasets.ImageFolder(('/home/lorenzp/datasets/CelebAHQ/Img/hq/data'+IMAGE_SIZE+'/'),transform = transforms.ToTensor())

# # dataset = CelebaDataset(csv_path='/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-gender-train_gender_hq_'+ DATA_SPLIT +IMAGE_SIZE+'.csv',
# #                         img_dir='/home/lorenzp/datasets/CelebAHQ/Img/hq/data'+IMAGE_SIZE+'/')
# loader = DataLoader(
#     dataset,
#     batch_size=8,
#     num_workers=1,
#     shuffle=False
# )

# mean = 0.
# std = 0.
# nb_samples = 0.


# for data in loader:
#     print('hello')
#     pdb.set_trace()
#     batch_samples = data.size(0)
# #     data = data.view(batch_samples, data.size(1), -1)
# #     mean += data.mean(2).sum(0)
# #     std += data.std(2).sum(0)
# #     nb_samples += batch_samples

# # mean /= nb_samples
# # std /= nb_samples


# R_channel = 0
# G_channel = 0
# B_channel = 0

# filepath = '/home/lorenzp/datasets/CelebAHQ/Img/hq/data'+IMAGE_SIZE+'/'
# pathDir = ['/home/lorenzp/datasets/CelebAHQ/Img/hq/data'+IMAGE_SIZE+'/']



# total_pixel = 0
# for idx in range(len(pathDir)):
#     filename = pathDir[idx]
#     print('filename', filename)
#     img = cv2.imread(os.path.join(filepath, filename))

#     im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     total_pixel = total_pixel + im_rgb.shape[0] * im_rgb.shape[1]

#     R_total = R_total + np.sum((im_rgb[:, :, 0] - R_mean) ** 2)
#     G_total = G_total + np.sum((im_rgb[:, :, 1] - G_mean) ** 2)
#     B_total = B_total + np.sum((im_rgb[:, :, 2] - B_mean) ** 2)

# R_std = sqrt(R_total / total_count)
# G_std = sqrt(G_total / total_count)
# B_std = sqrt(B_total / total_count)


# loader = DataLoader(
#     dataset,
#     batch_size=10,
#     num_workers=0,
#     shuffle=False
# )

