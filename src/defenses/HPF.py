#!/usr/bin/env python3

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


kernel = np.array([[0.0, -1.0, 0.0], 
                   [-1.0, 4.0, -1.0],
                   [0.0, -1.0, 0.0]])

kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)

def calc_highpass(images):
    hp = []       
    for i in range(len(images)):
        image = images[i]
        
        # import pdb; pdb.set_trace()
        
        hp_image  = cv2.filter2D(image.cpu().numpy(),-1,kernel)
        
        
        hp.append(hp_image.flatten())
    return hp
    
    
def high_pass_filter(args, images, images_advs):

    hpfs      = calc_highpass(images)
    hpfs_advs = calc_highpass(images_advs)
    
    characteristics       = np.asarray(hpfs, dtype=np.float32)
    characteristics_adv   = np.asarray(hpfs_advs, dtype=np.float32)
    
    return characteristics, characteristics_adv