#!/bin/bash

# define python command including script and all parameters to execute
PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --attack bim --net cif10 --nr 0"
PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --attack bim --net cif10 --nr 1"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --net cif10 --detector LayerPFS --attack pgd"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --net cif10 --detector LayerPFS --attack df"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --net cif10 --detector LayerPFS --attack cw"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --net cif10 --detector LayerPFS --attack std"

# NOTE: the "-u" flag is essential as we want the output of the python script directly and not the moment the buffer it full
#-----------------------------------------------------------------------------------------------------------------------------------




# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 25  --net imagenet32 --attack fgsm
# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 25  --net imagenet32 --attack bim
# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 25  --net imagenet32 --attack pgd
# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 25  --net imagenet32 --attack df
# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 25  --net imagenet32 --attack cw


# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 50  --net imagenet32 --attack fgsm
# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 50  --net imagenet32 --attack bim
# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 50  --net imagenet32 --attack pgd
# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 50  --net imagenet32 --attack df
# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 50  --net imagenet32 --attack cw

# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 75  --net imagenet32 --attack fgsm
# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 75  --net imagenet32 --attack bim
# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 75  --net imagenet32 --attack pgd
# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 75  --net imagenet32 --attack df
# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 75  --net imagenet32 --attack cw

# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 250  --net imagenet32 --attack fgsm
# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 250  --net imagenet32 --attack bim
# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 250  --net imagenet32 --attack pgd
# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 250  --net imagenet32 --attack df
# python -u extract_characteristics_imagenet32.py --detector InputMFS --num_classes 250  --net imagenet32 --attack cw


python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack std --net cif10vgg --nr 0
python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack std --net cif10vgg --nr 1
python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack std --net cif10vgg --nr 2
python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack std --net cif10vgg --nr 3
python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack std --net cif10vgg --nr 4
python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack std --net cif10vgg --nr 5
python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack std --net cif10vgg --nr 6
python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack std --net cif10vgg --nr 7
python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack std --net cif10vgg --nr 8
python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack std --net cif10vgg --nr 9
python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack std --net cif10vgg --nr 10
python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack std --net cif10vgg --nr 11
python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack std --net cif10vgg --nr 12

# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack bim --net cif10 --nr 0
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack bim --net cif10 --nr 1
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack bim --net cif10 --nr 2
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack bim --net cif10 --nr 3
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack bim --net cif10 --nr 4
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack bim --net cif10 --nr 5
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack bim --net cif10 --nr 6
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack bim --net cif10 --nr 7
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack bim --net cif10 --nr 8
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack bim --net cif10 --nr 9
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack bim --net cif10 --nr 10
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack bim --net cif10 --nr 11
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack bim --net cif10 --nr 12

# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack pgd --net cif10 --nr 0
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack pgd --net cif10 --nr 1
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack pgd --net cif10 --nr 2
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack pgd --net cif10 --nr 3
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack pgd --net cif10 --nr 4
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack pgd --net cif10 --nr 5
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack pgd --net cif10 --nr 6
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack pgd --net cif10 --nr 7
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack pgd --net cif10 --nr 8
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack pgd --net cif10 --nr 9
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack pgd --net cif10 --nr 10
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack pgd --net cif10 --nr 11
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack pgd --net cif10 --nr 12

# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack df --net cif10 --nr 0
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack df --net cif10 --nr 1
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack df --net cif10 --nr 2
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack df --net cif10 --nr 3
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack df --net cif10 --nr 4
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack df --net cif10 --nr 5
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack df --net cif10 --nr 6
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack df --net cif10 --nr 7
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack df --net cif10 --nr 8
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack df --net cif10 --nr 9
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack df --net cif10 --nr 10
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack df --net cif10 --nr 11
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack df --net cif10 --nr 12

# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack cw --net cif10 --nr 0
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack cw --net cif10 --nr 1
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack cw --net cif10 --nr 2
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack cw --net cif10 --nr 3
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack cw --net cif10 --nr 4
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack cw --net cif10 --nr 5
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack cw --net cif10 --nr 6
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack cw --net cif10 --nr 7
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack cw --net cif10 --nr 8
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack cw --net cif10 --nr 9
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack cw --net cif10 --nr 10
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack cw --net cif10 --nr 11
# python -u extract_characteristics_imagenet32.py --detector LayerMFS --attack cw --net cif10 --nr 12



#-----------------------------------------------------------------------------------------------------------------------------------
echo "finished experiment"

exit 0
#-----------------------------------------------------------------------------------------------------------------------------------