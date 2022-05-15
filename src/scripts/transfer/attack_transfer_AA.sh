#!/bin/bash

# std
# python attack_transfer.py --attack std  --attack_eval std  --detector InputMFS --clf LR --eps '8./255.' --eps_to '4./255.'
# python attack_transfer.py --attack std  --attack_eval std  --detector InputMFS --clf LR --eps '8./255.' --eps_to '2./255.'
# python attack_transfer.py --attack std  --attack_eval std  --detector InputMFS --clf LR --eps '8./255.' --eps_to '1./255.'
# python attack_transfer.py --attack std  --attack_eval std  --detector InputMFS --clf LR --eps '8./255.' --eps_to '0.5/255.'

# python attack_transfer.py --attack std  --attack_eval std  --detector InputMFS --clf RF --eps '8./255.' --eps_to '4./255.'
# python attack_transfer.py --attack std  --attack_eval std  --detector InputMFS --clf RF --eps '8./255.' --eps_to '2./255.'
# python attack_transfer.py --attack std  --attack_eval std  --detector InputMFS --clf RF --eps '8./255.' --eps_to '1./255.'
# python attack_transfer.py --attack std  --attack_eval std  --detector InputMFS --clf RF --eps '8./255.' --eps_to '0.5/255.'

########### Cif10
# BB 
# python attack_transfer.py --attack std  --attack_eval std  --detector InputMFS --clf LR --eps '8./255.' --eps_to '1./255.'
# python attack_transfer.py --attack std  --attack_eval std  --detector InputMFS --clf RF --eps '8./255.' --eps_to '1./255.'

# python attack_transfer.py --attack std  --attack_eval std  --detector InputMFS --clf LR --eps '1./255.' --eps_to '8./255.'
# python attack_transfer.py --attack std  --attack_eval std  --detector InputMFS --clf RF --eps '1./255.' --eps_to '8./255.'


# # WB
# python attack_transfer.py --attack std  --attack_eval std  --detector LayerMFS --clf LR --eps '8./255.' --eps_to '1./255.'
# python attack_transfer.py --attack std  --attack_eval std  --detector LayerMFS --clf RF --eps '8./255.' --eps_to '1./255.'

# python attack_transfer.py --attack std  --attack_eval std  --detector LayerMFS --clf LR --eps '1./255.' --eps_to '8./255.'
# python attack_transfer.py --attack std  --attack_eval std  --detector LayerMFS --clf RF --eps '1./255.' --eps_to '8./255.'



# # ########### ImageNet32
# # # BB 
# python attack_transfer.py --attack std --net imagenet32  --num_classes 1000 --attack_eval std  --detector InputMFS --clf LR --eps '8./255.' --eps_to '1./255.'
# python attack_transfer.py --attack std --net imagenet32  --num_classes 1000 --attack_eval std  --detector InputMFS --clf RF --eps '8./255.' --eps_to '1./255.'

# python attack_transfer.py --attack std --net imagenet32  --num_classes 1000 --attack_eval std  --detector InputMFS --clf LR --eps '1./255.' --eps_to '8./255.'
# python attack_transfer.py --attack std --net imagenet32  --num_classes 1000 --attack_eval std  --detector InputMFS --clf RF --eps '1./255.' --eps_to '8./255.'


# # WB
# python attack_transfer.py --attack std --net imagenet32  --num_classes 1000 --attack_eval std  --detector LayerMFS --clf LR --eps '8./255.' --eps_to '1./255.'
# python attack_transfer.py --attack std --net imagenet32  --num_classes 1000 --attack_eval std  --detector LayerMFS --clf RF --eps '8./255.' --eps_to '1./255.'

# python attack_transfer.py --attack std --net imagenet32  --num_classes 1000 --attack_eval std  --detector LayerMFS --clf LR --eps '1./255.' --eps_to '8./255.'
# python attack_transfer.py --attack std --net imagenet32  --num_classes 1000 --attack_eval std  --detector LayerMFS --clf RF --eps '1./255.' --eps_to '8./255.'


# ########### ImageNet64
# # BB 
# python attack_transfer.py --attack std --net imagenet64  --num_classes 1000 --attack_eval std  --detector InputMFS --clf LR --eps '8./255.' --eps_to '1./255.'
# python attack_transfer.py --attack std --net imagenet64  --num_classes 1000 --attack_eval std  --detector InputMFS --clf RF --eps '8./255.' --eps_to '1./255.'

# python attack_transfer.py --attack std --net imagenet64  --num_classes 1000 --attack_eval std  --detector InputMFS --clf LR --eps '1./255.' --eps_to '8./255.'
# python attack_transfer.py --attack std --net imagenet64  --num_classes 1000 --attack_eval std  --detector InputMFS --clf RF --eps '1./255.' --eps_to '8./255.'


# # WB
# python attack_transfer.py --attack std --net imagenet64  --num_classes 1000 --attack_eval std  --detector LayerMFS --clf LR --eps '8./255.' --eps_to '1./255.'
# python attack_transfer.py --attack std --net imagenet64  --num_classes 1000 --attack_eval std  --detector LayerMFS --clf RF --eps '8./255.' --eps_to '1./255.'

# python attack_transfer.py --attack std --net imagenet64  --num_classes 1000 --attack_eval std  --detector LayerMFS --clf LR --eps '1./255.' --eps_to '8./255.'
# python attack_transfer.py --attack std --net imagenet64  --num_classes 1000 --attack_eval std  --detector LayerMFS --clf RF --eps '1./255.' --eps_to '8./255.'



########### ImageNet128
# BB 
# python attack_transfer.py --attack std --net imagenet128  --num_classes 1000 --attack_eval std  --detector InputMFS --clf LR --eps '8./255.' --eps_to '1./255.'
# python attack_transfer.py --attack std --net imagenet128  --num_classes 1000 --attack_eval std  --detector InputMFS --clf RF --eps '8./255.' --eps_to '1./255.'

# python attack_transfer.py --attack std --net imagenet128  --num_classes 1000 --attack_eval std  --detector InputMFS --clf LR --eps '1./255.' --eps_to '8./255.'
# python attack_transfer.py --attack std --net imagenet128  --num_classes 1000 --attack_eval std  --detector InputMFS --clf RF --eps '1./255.' --eps_to '8./255.'


# # WB
# python attack_transfer.py --attack std --net imagenet128  --num_classes 1000 --attack_eval std  --detector LayerMFS --clf LR --eps '8./255.' --eps_to '1./255.'
# python attack_transfer.py --attack std --net imagenet128  --num_classes 1000 --attack_eval std  --detector LayerMFS --clf RF --eps '8./255.' --eps_to '1./255.'

# python attack_transfer.py --attack std --net imagenet128  --num_classes 1000 --attack_eval std  --detector LayerMFS --clf LR --eps '1./255.' --eps_to '8./255.'
# python attack_transfer.py --attack std --net imagenet128  --num_classes 1000 --attack_eval std  --detector LayerMFS --clf RF --eps '1./255.' --eps_to '8./255.'




# # ########### celebaHQ32
# # # BB 
# python attack_transfer.py --attack std --net celebaHQ32  --num_classes 4 --attack_eval std  --detector InputMFS --clf LR --eps '8./255.' --eps_to '1./255.'
# python attack_transfer.py --attack std --net celebaHQ32  --num_classes 4 --attack_eval std  --detector InputMFS --clf RF --eps '8./255.' --eps_to '1./255.'

# python attack_transfer.py --attack std --net celebaHQ32  --num_classes 4 --attack_eval std  --detector InputMFS --clf LR --eps '1./255.' --eps_to '8./255.'
# python attack_transfer.py --attack std --net celebaHQ32  --num_classes 4 --attack_eval std  --detector InputMFS --clf RF --eps '1./255.' --eps_to '8./255.'


# # WB
# python attack_transfer.py --attack std --net celebaHQ32  --num_classes 4 --attack_eval std  --detector LayerMFS --clf LR --eps '8./255.' --eps_to '1./255.'
# python attack_transfer.py --attack std --net celebaHQ32  --num_classes 4 --attack_eval std  --detector LayerMFS --clf RF --eps '8./255.' --eps_to '1./255.'

# python attack_transfer.py --attack std --net celebaHQ32  --num_classes 4 --attack_eval std  --detector LayerMFS --clf LR --eps '1./255.' --eps_to '8./255.'
# python attack_transfer.py --attack std --net celebaHQ32  --num_classes 4 --attack_eval std  --detector LayerMFS --clf RF --eps '1./255.' --eps_to '8./255.'


########### celebaHQ64
# BB 
# python attack_transfer.py --attack std --net celebaHQ64  --num_classes 4 --attack_eval std  --detector InputMFS --clf LR --eps '8./255.' --eps_to '1./255.'
# python attack_transfer.py --attack std --net celebaHQ64  --num_classes 4 --attack_eval std  --detector InputMFS --clf RF --eps '8./255.' --eps_to '1./255.'

# python attack_transfer.py --attack std --net celebaHQ64  --num_classes 4 --attack_eval std  --detector InputMFS --clf LR --eps '1./255.' --eps_to '8./255.'
# python attack_transfer.py --attack std --net celebaHQ64  --num_classes 4 --attack_eval std  --detector InputMFS --clf RF --eps '1./255.' --eps_to '8./255.'


# WB
# python attack_transfer.py --attack std --net celebaHQ64  --num_classes 4 --attack_eval std  --detector LayerMFS --clf LR --eps '8./255.' --eps_to '1./255.'
python attack_transfer.py --attack std --net celebaHQ64  --num_classes 4 --attack_eval std  --detector LayerMFS --clf RF --eps '8./255.' --eps_to '1./255.' --wanted_samples 1700

# python attack_transfer.py --attack std --net celebaHQ64  --num_classes 4 --attack_eval std  --detector LayerMFS --clf LR --eps '1./255.' --eps_to '8./255.'
# python attack_transfer.py --attack std --net celebaHQ64  --num_classes 4 --attack_eval std  --detector LayerMFS --clf RF --eps '1./255.' --eps_to '8./255.'



# ########### celebaHQ128
# # BB 
# python attack_transfer.py --attack std --net celebaHQ128  --num_classes 4 --attack_eval std  --detector InputMFS --clf LR --eps '8./255.' --eps_to '1./255.'
# python attack_transfer.py --attack std --net celebaHQ128  --num_classes 4 --attack_eval std  --detector InputMFS --clf RF --eps '8./255.' --eps_to '1./255.'

# python attack_transfer.py --attack std --net celebaHQ128  --num_classes 4 --attack_eval std  --detector InputMFS --clf LR --eps '1./255.' --eps_to '8./255.'
# python attack_transfer.py --attack std --net celebaHQ128  --num_classes 4 --attack_eval std  --detector InputMFS --clf RF --eps '1./255.' --eps_to '8./255.'


# # WB
# python attack_transfer.py --attack std --net celebaHQ128  --num_classes 4 --attack_eval std  --detector LayerMFS --clf LR --eps '8./255.' --eps_to '1./255.'
# python attack_transfer.py --attack std --net celebaHQ128  --num_classes 4 --attack_eval std  --detector LayerMFS --clf RF --eps '8./255.' --eps_to '1./255.'

# python attack_transfer.py --attack std --net celebaHQ128  --num_classes 4 --attack_eval std  --detector LayerMFS --clf LR --eps '1./255.' --eps_to '8./255.'
# python attack_transfer.py --attack std --net celebaHQ128  --num_classes 4 --attack_eval std  --detector LayerMFS --clf RF --eps '1./255.' --eps_to '8./255.'
