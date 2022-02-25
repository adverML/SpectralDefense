#this script extracts the correctly classified images
print('Load modules...')
import os
import time
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets

from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
# from models.vgg_cif10 import VGG
# from models.vgg import vgg16_bn

from models.wideresidual import WideResNet, WideBasic

import argparse
import pdb
import pandas as pd

from conf import settings
from utils import get_network, get_training_dataloader, get_validation_dataloader, get_test_dataloader, \
    get_validation_dataloader_path, get_test_dataloader_path, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

IMAGE_SIZE = settings.IMAGE_SIZE
DATA_SPLIT = settings.DATA_SPLIT

parser = argparse.ArgumentParser()
parser.add_argument('-net', type=str, default='resnet50', help='net type')
parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('-resume', action='store_true', default=False, help='resume training')
parser.add_argument('-parallel', action='store_true', default=False, help='train on multiple GPUs training')
args = parser.parse_args()

print(IMAGE_SIZE, args)
print(settings.CHECKPOINT_PATH)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

batch_time = AverageMeter('Time', ':6.3f')
data_time =  AverageMeter('Data', ':6.3f')
losses =     AverageMeter('Loss', ':.4e')
top1 =       AverageMeter('Acc@1', ':6.2f')
top5 =       AverageMeter('Acc@5', ':6.2f')




if settings.DATA == 'Hair_Color' or settings.DATA == 'Smiling':
    #choose attack
    net = args.net

    print('Load model...')
    model = get_network(args)

    new_state_dict = torch.load(settings.CHECKPOINT_PATH)
    model.load_state_dict(new_state_dict)
    model.eval()
elif settings.DATA == "TinyImageNet":
    model = models.__dict__['resnet50'](pretrained=False)
    model = torch.nn.DataParallel(model).cuda()
    new_state_dict = torch.load('/home/lorenzp/adversialml/src/pytorch-CelebAHQ/tiny_model_best.pth.tar')
    model.load_state_dict(new_state_dict['state_dict'])
    model.eval()
    # model.load_state_dict(new_state_dict)
elif settings.DATA == "cif-10":
    model = VGG('VGG16')
    checkpoint = torch.load('./models/vgg_cif10.pth')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    


#normalizing the data
# print('Load CIFAR-10 test set')

if settings.DATA == 'Hair_Color':
    testloader = get_validation_dataloader_path(
        settings.CELEABAHQ_TRAIN_MEAN,
        settings.CELEABAHQ_TRAIN_STD,
        data=settings.DATA,
        num_workers=1,
        batch_size=args.b,
        shuffle=False)
elif settings.DATA == "Smiling":
    testloader = get_test_dataloader_path(
            settings.CELEABAHQ_TRAIN_MEAN,
            settings.CELEABAHQ_TRAIN_STD,
            data=settings.DATA,
            num_workers=1,
            batch_size=args.b,
            shuffle=False)
elif settings.DATA == "TinyImageNet":
    testdir = os.path.join('/home/lorenzp/datasets/IMagenet/tiny-imagenet-200', 'val')

    testloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # normalize,
        ])),
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)
elif settings.DATA == "cif-10":

    #normalizing the data
    print('Load CIFAR-10 test set')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    testset_normalized = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader_normalized = torch.utils.data.DataLoader(testset_normalized, batch_size=1, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=2)

# model = model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# data_iter = iter(testloader)
clean_dataset = []
clean_img_names = []
correct = 0
total = 0
i = 0
print('Classify images...')



# for images, labels in testloader_normalized:
# with torch.no_grad():
#     end = time.time()
#     for i, (images, target) in enumerate(testloader):
#         images = images.cuda(non_blocking=True)
#         target = target.cuda(non_blocking=True)

#         # compute output
#         output = model(images)
#         # loss = criterion(output, target)

#         # measure accuracy and record loss
#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
#         # losses.update(loss.item(), images.size(0))
#         top1.update(acc1[0], images.size(0))
#         top5.update(acc5[0], images.size(0))

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if i % 10 == 0:
#             print(top1, top5)
#             # progress.display(i)

#     # TODO: this should also be done with the ProgressMeter
#     print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
#             .format(top1=top1, top5=top5))


# with torch.no_grad():
#     for idx, (images, labels) in enumerate(testloader):
#         images = images.cuda()
#         labels = labels.cuda()
#         outputs = model(images)
#         # pdb.set_trace()

#         _, pred = torch.max(outputs.data, 1)
#         # _, pred = outputs.topk(5, 1, largest=True, sorted=True)
#         # pdb.set_trace()

#         total += labels.size(0)
#         correct += (pred == labels).sum().item()
#         if (pred == labels):
#             clean_dataset.append([images, labels])
#             clean_img_names.append([x[0],  labels.cpu().numpy()[0], 2])
#         i += 1
#         print('Accuracy of the network on the %d test images: %d %%' % (idx, 100 * correct / total))


with torch.no_grad():
    for idx, (images, labels, x, y) in enumerate(testloader):
    # for idx, (images, labels) in enumerate(testloader):
        # data = data_iter.next()
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)

        #if idx > 100:
        #    break

        _, pred = torch.max(outputs.data, 1)
        # _, pred = outputs.topk(5, 1, largest=True, sorted=True)
        # pdb.set_trace()

        total += labels.size(0)
        correct += (pred == labels).sum().item()
        if (pred == labels):
            clean_dataset.append([images, labels])
            clean_img_names.append([x[0],  labels.cpu().numpy()[0], 2])
        i += 1
        print('Accuracy of the network on the %d test images: %d %%' % (idx, 100 * correct / total))


if settings.DATA == 'cif-10':
    torch.save(clean_dataset, './data/clean_data_cif_100'+net)

else:

    df = pd.DataFrame(clean_img_names, columns=['', settings.DATA, 'Partitions'])


    tmp_normalized = ''
    if settings.NORMALIZED:
        tmp_normalized = '_norm'

    # pdb.set_trace()
    df.to_csv('../pytorch_ipynb/cnn/celeba-classified_' + settings.DATA.lower() + '_hq_' + DATA_SPLIT + IMAGE_SIZE + tmp_normalized + '.csv', index=False)

    # torch.save(clean_dataset, './data/clean_data_' + IMAGE_SIZE + '_' + net)
print('Done extracting and saving correctly classified images!')