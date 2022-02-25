""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from dataset import CelebaDataset, CelebaDatasetPath

from conf import settings

DATA_SPLIT = settings.DATA_SPLIT
IMAGE_SIZE = settings.IMAGE_SIZE

# from global_settings import DATA_SPLIT
# from global_settings import IMAGE_SIZE

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    torch.dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    torch.dist.destroy_process_group()

def get_network(args):
    """ return given network
    """
    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_classes=settings.NUM_CLASSES)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(num_classes=settings.NUM_CLASSES)
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_classes=settings.NUM_CLASSES)
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_classes=settings.NUM_CLASSES)
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(num_classes=settings.NUM_CLASSES)
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        # net = net.cuda()

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)
        # import pdb; pdb.set_trace()
        # model = CreateModel()

        if args.parallel:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # net = nn.parallel.DistributedDataParallel(net, device_ids=list(range(torch.cuda.device_count())))
            # net = nn.parallel.DataParallel(net, device_ids=[0,1])
            net = nn.parallel.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))

            #net = nn.parallel.DataParallel(net, device_ids=[0,1])
            #net = nn.parallel.DataParallel(net, device_ids=[0,1])

        # net = nn.parallel.DistributedDataParallel(net, device_ids=[0,1,2])
        # net = nn.DataParallel(net)
        # net.to(device)

        net = net.cuda()

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #     net = nn.DataParallel(net, device_ids = [ 0, 1])
        #     # net.to(f'cuda:{net.device_ids[0]}')
        #     # net = nn.DataParallel(net)

    return net

custom_transform = transforms.Compose([
                                    # transforms.RandomCrop(128, padding=4),
                                    # transforms.CenterCrop((178, 178)),
                                    # transforms.Resize((128, 128)),
                                    # transforms.RandomCrop(512), 
                                    transforms.ToTensor()
                                    ])

def get_training_dataloader(mean, std, data='Male', batch_size=64, num_workers=8, shuffle=True):

    if data == 'Male':
        csv_path = '/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-train_gender_hq_' + DATA_SPLIT + '.csv'
    elif data == 'Smiling':
        csv_path = '/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-train_smiling_hq _' + DATA_SPLIT + '.csv'
    else:
        csv_path = '/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-train_hair_hq_ext_' + DATA_SPLIT + '.csv'
        # csv_path = '/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-gender-train_hair_hq_'+ DATA_SPLIT + IMAGE_SIZE +'.csv'

    print(csv_path)

    train_dataset = CelebaDataset(csv_path=csv_path,
                        img_dir='/home/lorenzp/datasets/CelebAHQ/Img/hq/data'+IMAGE_SIZE+'/',
#                           img_dir='/home/lorenzp/datasets/CelebA/Img/img_align_celeba/',
                        data=data,
                        transform=custom_transform)

    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

    return train_loader


def get_validation_dataloader(mean, std, data='Male', batch_size=64, num_workers=8, shuffle=False):

    if data == 'Male':
        csv_path = '/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-valid_gender_hq_'+ DATA_SPLIT + '.csv'
    elif data == 'Smiling':
        csv_path = '/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-valid_smiling_hq_'+ DATA_SPLIT + '.csv'
    else:
        csv_path='/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-valid_hair_hq_' + DATA_SPLIT + '.csv'

    print(csv_path)

    val_dataset = CelebaDataset(csv_path=csv_path,
                                img_dir='/home/lorenzp/datasets/CelebAHQ/Img/hq/data'+IMAGE_SIZE+'/',
    #                              img_dir='/home/lorenzp/datasets/CelebA/Img/img_align_celeba/',
                                data=data,
                                transform=custom_transform)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=False)

    return val_loader


def get_test_dataloader(mean, std, data='Male', batch_size=64, num_workers=8, shuffle=False):

    if data == 'Male':
        csv_path='/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-test_gender_hq_'+ DATA_SPLIT +IMAGE_SIZE+'.csv'
    elif data == 'Smiling':
        csv_path = '/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-test_smiling_hq_'+ DATA_SPLIT + IMAGE_SIZE +'.csv'
    else:
        csv_path='/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-test_hair_hq_'+ DATA_SPLIT +IMAGE_SIZE+'.csv'

    print(csv_path)

    test_dataset = CelebaDataset(csv_path=csv_path,
                                img_dir='/home/lorenzp/datasets/CelebAHQ/Img/hq/data'+IMAGE_SIZE+'/',
    #                              img_dir='/home/lorenzp/datasets/CelebA/Img/img_align_celeba/',
                                data=data,
                                transform=custom_transform)

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=False)

    return test_loader

def get_clean_data_dataloader(mean, std, data='Male', batch_size=64, num_workers=8, shuffle=False):
    if data == 'Male':
        csv_path = '/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-classified_gender_hq_'+ DATA_SPLIT +'.csv'
    if data == 'Smiling':
        csv_path = '/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-classified_smiling_hq_'+ DATA_SPLIT +'.csv'
    else:
        csv_path = '/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-classified_hair_color_hq_'+ DATA_SPLIT +IMAGE_SIZE+'.csv'
    
    test_dataset = CelebaDataset(csv_path=csv_path,
                                img_dir='/home/lorenzp/datasets/CelebAHQ/Img/hq/data'+IMAGE_SIZE+'/',
    #                              img_dir='/home/lorenzp/datasets/CelebA/Img/img_align_celeba/',
                                data=data,
                                transform=custom_transform)


    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=False)

    return test_loader

def get_validation_dataloader_path(mean, std, data='Male', batch_size=64, num_workers=8, shuffle=False):
    if data == 'Male':
        csv_path = '/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-gender-valid_gender_hq_'+ DATA_SPLIT +IMAGE_SIZE+'.csv'
    elif data == 'Smiling':
        csv_path = '/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-gender-valid_smiling_hq_'+ DATA_SPLIT + IMAGE_SIZE +'.csv'
    else:
        csv_path='/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-gender-valid_hair_hq_'+ DATA_SPLIT +IMAGE_SIZE+'.csv'

    print(csv_path)

    val_dataset = CelebaDatasetPath(csv_path=csv_path,
                                img_dir='/home/lorenzp/datasets/CelebAHQ/Img/hq/data'+IMAGE_SIZE+'/',
    #                              img_dir='/home/lorenzp/datasets/CelebA/Img/img_align_celeba/',
                                data=data,
                                transform=custom_transform)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=False)

    return val_loader


def get_test_dataloader_path(mean, std, data='Male', batch_size=64, num_workers=8, shuffle=False):

    if data == 'Male':
        csv_path='/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-gender-test_gender_hq_'+ DATA_SPLIT +IMAGE_SIZE+'.csv'
    elif data == 'Smiling':
        csv_path = '/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-gender-test_smiling_hq_'+ DATA_SPLIT + IMAGE_SIZE +'.csv'
    else:
        csv_path='/home/lorenzp/adversialml/src/pytorch_ipynb/cnn/celeba-gender-test_hair_hq_'+ DATA_SPLIT +IMAGE_SIZE+'.csv'

    test_dataset = CelebaDatasetPath(csv_path=csv_path,
                                img_dir='/home/lorenzp/datasets/CelebAHQ/Img/hq/data'+IMAGE_SIZE+'/',
    #                              img_dir='/home/lorenzp/datasets/CelebA/Img/img_align_celeba/',
                                data=data,
                                transform=custom_transform)

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=True)

    return test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    import pdb; pdb.set_trace()
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]


def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]
    