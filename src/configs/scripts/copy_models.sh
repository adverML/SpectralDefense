#!/bin/bash


CIF10_CKPT='./checkpoint/wideresnet_2810/wide_resnet_ckpt.pth'
CIF10VGG_CKPT='./checkpoint/vgg16/original/models/vgg_cif10.pth'
CIF100VGG_CKPT='./checkpoint/vgg16/original/models/vgg_cif100.pth'
CIF100_CKPT='./../pytorch-classification/checkpoints/cifar100/wideresnet2810/model_best.pth.tar'

IMAGENET32_CKPT_1000='./../pytorch-classification/checkpoints/imagenet32/wideresent2810/model_best.pth.tar'#model_best.pth.tar
IMAGENET32_CKPT_250='./../pytorch-classification/checkpoints/imagenet32/wideresent2810_250/model_best.pth.tar'
IMAGENET32_CKPT_100='./../pytorch-classification/checkpoints/imagenet32/wideresent2810_100/model_best.pth.tar'
IMAGENET32_CKPT_75='./../pytorch-classification/checkpoints/imagenet32/wideresent2810_75/model_best.pth.tar'
IMAGENET32_CKPT_50='./../pytorch-classification/checkpoints/imagenet32/wideresent2810_50/model_best.pth.tar'
IMAGENET32_CKPT_25='./../pytorch-classification/checkpoints/imagenet32/wideresent2810_25/model_best.pth.tar'
IMAGENET32_CKPT_10='./../pytorch-classification/checkpoints/imagenet32/wideresent2810_10/model_best.pth.tar'

IMAGENET64_CKPT_1000='./../pytorch-classification/checkpoints/imagenet64/wideresent2810/model_best.pth.tar'
IMAGENET128_CKPT_1000='./../pytorch-classification/checkpoints/imagenet128/wideresent2810/model_best.pth.tar'

CELEBAHQ32_CKPT_2='/home/lorenzp/adversialml/src/pytorch-CelebAHQ/checkpoint/wrn2810/32x32_128_0.1_Smiling_Thursday_30_September_2021_11h_01m_19s/wrn2810-161-best.pth'
CELEBAHQ64_CKPT_2='/home/lorenzp/adversialml/src/pytorch-CelebAHQ/checkpoint/wrn2810/64x64_128_0.1_Smiling_Thursday_30_September_2021_15h_35m_05s/wrn2810-141-best.pth'
CELEBAHQ128_CKPT_2='/home/lorenzp/adversialml/src/pytorch-CelebAHQ/checkpoint/wrn2810/128x128_64_0.1_Smiling_Thursday_30_September_2021_15h_37m_55s/wrn2810-140-best.pth'

CELEBAHQ32_CKPT_4='/home/lorenzp/adversialml/src/pytorch-CelebAHQ/checkpoint/wrn2810/32x32_64_0.1_Hair_Color_Thursday_04_November_2021_14h_35m_14s/wrn2810-200-best.pth'
CELEBAHQ64_CKPT_4='/home/lorenzp/adversialml/src/pytorch-CelebAHQ/checkpoint/wrn2810/64x64_64_0.1_Hair_Color_Thursday_04_November_2021_17h_25m_16s/wrn2810-171-best.pth'
CELEBAHQ128_CKPT_4='/home/lorenzp/adversialml/src/pytorch-CelebAHQ/checkpoint/wrn2810/128x128_64_0.1_Hair_Color_Thursday_04_November_2021_17h_38m_53s/wrn2810-100-regular.pth'
CELEBAHQ256_CKPT_4='/home/lorenzp/adversialml/src/pytorch-CelebAHQ/checkpoint/wrn2810/256x256_24_0.1_Hair_Color_Friday_05_November_2021_16h_44m_36s/wrn2810-70-regular.pth'


# for mod in CELEBAHQ256_CKPT_4
for mod in CIF10_CKPT CIF10VGG_CKPT CIF100VGG_CKPT CIF100_CKPT IMAGENET32_CKPT_1000 IMAGENET32_CKPT_250 IMAGENET32_CKPT_100 IMAGENET32_CKPT_75 IMAGENET32_CKPT_50 IMAGENET32_CKPT_25 IMAGENET32_CKPT_10 IMAGENET64_CKPT_1000 IMAGENET128_CKPT_1000 CELEBAHQ32_CKPT_2 CELEBAHQ64_CKPT_2 CELEBAHQ128_CKPT_2 CELEBAHQ32_CKPT_4 CELEBAHQ64_CKPT_4 CELEBAHQ128_CKPT_4 CELEBAHQ256_CKPT_4
do
    destiny="/m/scratch/itwm/lorenzp/usedmodels/"$mod
    echo "${!mod}"
    echo $destiny 
    ssh lorenzp@ssh2.itwm.fraunhofer.de "mkdir -p $destiny"
    scp -r "${!mod}" lorenzp@ssh2.itwm.fraunhofer.de:$destiny
done

# RUNSPATH='/home/lorenzp/adversialml/src/src/log_evaluation'

# for mod in RUNSPATH
# do
#     destiny="/m/scratch/itwm/lorenzp/log_evaluation/"$mod
#     echo "${!mod}"
#     echo $destiny 
#     ssh lorenzp@ssh2.itwm.fraunhofer.de "mkdir -p $destiny"
#     scp -r "${!mod}" lorenzp@ssh2.itwm.fraunhofer.de:$destiny
# done