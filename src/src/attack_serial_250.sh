#!/bin/bash

# define python command including script and all parameters to execute


# NOTE: the "-u" flag is essential as we want the output of the python script directly and not the moment the buffer it full
#-----------------------------------------------------------------------------------------------------------------------------------

# python -u attack_imagenet32.py --net imagenet32 --num_classes 25 --attack fgsm
# python -u attack_imagenet32.py --net imagenet32 --num_classes 25 --attack bim
# python -u attack_imagenet32.py --net imagenet32 --num_classes 25 --attack pgd
# python -u attack_imagenet32.py --net imagenet32 --num_classes 25 --attack df
# python -u attack_imagenet32.py --net imagenet32 --num_classes 25 --attack cw


python -u attack_imagenet32.py --net imagenet32 --num_classes 250 --attack fgsm
python -u attack_imagenet32.py --net imagenet32 --num_classes 250 --attack bim
python -u attack_imagenet32.py --net imagenet32 --num_classes 250 --attack pgd
python -u attack_imagenet32.py --net imagenet32 --num_classes 250 --attack df
python -u attack_imagenet32.py --net imagenet32 --num_classes 250 --attack cw


# python -u attack_imagenet32.py --net imagenet32 --num_classes 75 --attack fgsm
# python -u attack_imagenet32.py --net imagenet32 --num_classes 75 --attack bim
# python -u attack_imagenet32.py --net imagenet32 --num_classes 75 --attack pgd
# python -u attack_imagenet32.py --net imagenet32 --num_classes 75 --attack df
# python -u attack_imagenet32.py --net imagenet32 --num_classes 75 --attack cw


# python -u attack_imagenet32.py --net imagenet32 --num_classes 250 --attack fgsm
# python -u attack_imagenet32.py --net imagenet32 --num_classes 250 --attack bim
# python -u attack_imagenet32.py --net imagenet32 --num_classes 250 --attack pgd
# python -u attack_imagenet32.py --net imagenet32 --num_classes 250 --attack df
# python -u attack_imagenet32.py --net imagenet32 --num_classes 250 --attack cw



#-----------------------------------------------------------------------------------------------------------------------------------
echo "finished experiment"

exit 0
#-----------------------------------------------------------------------------------------------------------------------------------