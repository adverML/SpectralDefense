#!/bin/bash


# git filter-branch --tree-filter 'rm -rf src/pytorch-CelebAHQ/checkpoint.pth.tar' HEAD
# git filter-branch --tree-filter 'rm -rf src/pytorch-CelebAHQ/tiny_model_best.pth.tar' -f HEAD
# git filter-branch --tree-filter 'rm -rf src/pytorch_ipynb/cnn/data/cifar-10-python.tar.gz' -f HEAD
# git filter-branch --tree-filter 'rm -rf src/pytorch-CelebAHQ/checkpoint.pth.tar' -f HEAD
# git filter-branch --tree-filter 'rm -rf src/pytorch-CelebAHQ/model_best.pth.tar' -f  HEAD
# git filter-branch --tree-filter 'rm -rf src/pytorch-CelebAHQ/resnet50_imagenet_model_best.pth.tar' -f HEAD
# git filter-branch --tree-filter 'rm -rf src/pytorch-CelebAHQ/tiny_checkpoint.pth.tar' -f HEAD



# cp src/pytorch-CelebAHQ/checkpoint.pth.tar  ../weights
# cp src/pytorch-CelebAHQ/tiny_model_best.pth.tar ../weights 
# cp src/pytorch_ipynb/cnn/data/cifar-10-python.tar.gz ../weights
# cp src/pytorch-CelebAHQ/checkpoint.pth.tar ../weights
# cp src/pytorch-CelebAHQ/model_best.pth.tar ../weights
# cp src/pytorch-CelebAHQ/resnet50_imagenet_model_best.pth.tar ../weights
# cp src/pytorch-CelebAHQ/tiny_checkpoint.pth.tar ../weights


git rm src/pytorch-CelebAHQ/checkpoint.pth.tar  
git rm src/pytorch-CelebAHQ/tiny_model_best.pth.tar 
git rm src/pytorch_ipynb/cnn/data/cifar-10-python.tar.gz 
git rm src/pytorch-CelebAHQ/checkpoint.pth.tar 
git rm src/pytorch-CelebAHQ/model_best.pth.tar 
git rm src/pytorch-CelebAHQ/resnet50_imagenet_model_best.pth.tar
git rm src/pytorch-CelebAHQ/tiny_checkpoint.pth.tar 