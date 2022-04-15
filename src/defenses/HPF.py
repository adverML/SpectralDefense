#!/usr/bin/env python3

# https://www.askpython.com/python-modules/opencv-filter2d
# http://www.psychocodes.in/image-sharpening-by-high-pass-filter-using-python-and-opencv.html
# https://stackoverflow.com/questions/6094957/high-pass-filter-for-image-processing-in-python-by-using-scipy-numpy

import torch
import numpy as np
import cv2
from scipy import ndimage

import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter

from PIL.ImageFilter import (
    BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
    EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
    )


# kind of sharbpening filter
kernel = np.array([[0.0, -1.0, 0.0], 
                   [-1.0, 4.0, -1.0],
                   [0.0, -1.0, 0.0]])

kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)

def calc_highpass(images):
    hp = []       
    for i in range(len(images)):
        image = images[i] * 255.
        hp_image  = cv2.filter2D(image.cpu().numpy(), -1, kernel)  / 255.
    
        hp.append(hp_image.flatten())
    return hp
    
    
def high_pass_filter(args, images, images_advs):

    hpfs      = calc_highpass(images)
    hpfs_advs = calc_highpass(images_advs)
    
    characteristics       = np.asarray(hpfs, dtype=np.float32)
    characteristics_adv   = np.asarray(hpfs_advs, dtype=np.float32)
    
    return characteristics, characteristics_adv


def test():
    # img_pth = "/home/lorenzp/adversialml/src/data/attacks/run_8/cif10/wrn_28_10_10/gauss/images"
    img_pth = "/home/lorenzp/adversialml/src/analysis/highpass/sample.png"
    out_pth = "/home/lorenzp/adversialml/src/analysis/highpass/result_test.jpg"
    # image = plt.imread(img_pth) * 255.
    
    image  = cv2.imread(img_pth)
    image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  / 255.
        
    print(image.shape)
    hp_image = cv2.filter2D(image,-1,kernel)
    
    import pdb; pdb.set_trace()
    
    # plt.imsave(, hp_image)
    cv2.imwrite(out_pth, hp_image)


def test2():
    img_pth = "/home/lorenzp/adversialml/src/data/attacks/run_8/cif10/wrn_28_10_10/gauss/images"
    in_pth = "/home/lorenzp/adversialml/src/analysis/highpass/cif10_in.png"
    out_pth = "/home/lorenzp/adversialml/src/analysis/highpass/cif10_out.jpg"
    
    # image = plt.imread(img_pth) * 255.
    
    image = torch.load(img_pth)[0].cpu().numpy()
    import pdb; pdb.set_trace()
    
    image = image.transpose((1,2,0)) * 255.
    # plt.imsave(in_pth, image)
    cv2.imwrite(in_pth, image)
    
    # image  = cv2.imread(img_pth)
    # image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  / 255.
    
    print(image.shape)
    hp_image = cv2.filter2D(image,-1,kernel)
    
    
    
    # plt.imsave(, hp_image)
    cv2.imwrite(out_pth, hp_image)



def test3():
    img_pth = "/home/lorenzp/adversialml/src/data/attacks/run_8/cif10/wrn_28_10_10/gauss/images"
    in_pth = "/home/lorenzp/adversialml/src/analysis/highpass/cif10_in_255.png"
    out_pth = "/home/lorenzp/adversialml/src/analysis/highpass/cif10_out_255.png"
    
    # image = plt.imread(img_pth) * 255.
    image = torch.load(img_pth)[0].cpu().numpy()
    # import pdb; pdb.set_trace()
    
    image = image.transpose((1,2,0))
    print("image: ", image.shape)
    plt.imsave(in_pth, image*255.)
    
    print(image.shape)
    hp_image = cv2.filter2D(image,-1,kernel)
    plt.imsave(out_pth, hp_image*255.)


if __name__ == "__main__":
    # test()
    # test2()
    test3()
    pass
    
    