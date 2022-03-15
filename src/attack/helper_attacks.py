#!/usr/bin/env python3

from conf import settings

import torch
import os
import numpy as np





def adapt_batchsize(args, device_name):
    batch_size = 128 
    if device_name == 'titan v' and (args.net == 'imagenet128' or args.net == 'celebaHQ128'):
        batch_size = 24
    if device_name == 'a100' and (args.net == 'imagenet' or  args.net == 'imagenet128' or args.net == 'celebaHQ128'):
        batch_size = 48

    if device_name == 'titan v' and (args.attack == "apgd-ce" or args.attack == "apgd-t" or args.attack == "fab-t" or args.attack == "square"):
        batch_size = 256

    if device_name == 'titan v' and (( args.attack == "cw" or args.attack == "df") and (args.net == 'imagenet64' or args.net == 'celebaHQ64')):
        batch_size = 48
    elif args.net == 'celebaHQ256':
        batch_size = 24
    elif device_name == 'titan v' and args.net == 'imagenet':
        batch_size = 32
    
    return batch_size



def check_args_attack(args, logger):
    if args.net_normalization:
        if not args.attack == 'std' and not args.attack == 'apgd-ce' and not args.attack == 'apgd-t' and not args.attack == 'fab-t' and not args.attack == 'square':
            logger.log("Warning: Net normalization must be switched off!  Net normalization is switched off now!")
            args.net_normalization = False
            
    return args