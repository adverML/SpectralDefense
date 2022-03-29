#!/usr/bin/env python3

from conf import settings
from utils import (
    get_debug_info
)
from attack.helper_attacks import (
    check_args_attack
)

import torch


def check_args_generate_clean_data(args):
    if not args.batch_size == 1:
        get_debug_info(msg='Err: Batch size must be always 1!')
        assert True

    if (args.net == 'imagenet' or args.net == 'imagenet32' or args.net == 'imagenet64' or args.net == 'imagenet128' ) and not args.shuffle_off:
        get_debug_info("Warning: Shuffle data for ImageNet and variants must be switched off!")
        args.shuffle_off = True
        
    args = check_args_attack(args, net_normalization=False, num_classes=True, img_size=True)
    
    return args

