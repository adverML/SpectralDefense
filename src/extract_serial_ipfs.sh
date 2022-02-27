#!/bin/bash

# define python command including script and all parameters to execute
PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --attack bim --net cif10 --nr 0"
PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --attack bim --net cif10 --nr 1"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --net cif10 --detector InputPFS --attack pgd"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --net cif10 --detector InputPFS --attack df"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --net cif10 --detector InputPFS --attack cw"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --net cif10 --detector InputPFS --attack std"

# NOTE: the "-u" flag is essential as we want the output of the python script directly and not the moment the buffer it full
#-----------------------------------------------------------------------------------------------------------------------------------

python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net cif10 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim  --net cif10 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd  --net cif10 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std  --net cif10 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df   --net cif10 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw   --net cif10 --nr -1


python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net cif100 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim  --net cif100 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd  --net cif100 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std  --net cif100 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df   --net cif100 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw   --net cif100 --nr -1


python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net imagenet32 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim  --net imagenet32 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd  --net imagenet32 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std  --net imagenet32 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df   --net imagenet32 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw   --net imagenet32 --nr -1


python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net imagenet64 --img_size 64 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim  --net imagenet64 --img_size 64 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd  --net imagenet64 --img_size 64 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std  --net imagenet64 --img_size 64 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df   --net imagenet64 --img_size 64 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw   --net imagenet64 --img_size 64 --nr -1


python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net imagenet128 --img_size 128 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim  --net imagenet128 --img_size 128 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd  --net imagenet128 --img_size 128 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std  --net imagenet128 --img_size 128 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df   --net imagenet128 --img_size 128 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw   --net imagenet128 --img_size 128 --nr -1


python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net celebaHQ32 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim  --net celebaHQ32 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd  --net celebaHQ32 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std  --net celebaHQ32 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df   --net celebaHQ32 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw   --net celebaHQ32 --nr -1


python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net celebaHQ64 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim  --net celebaHQ64 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd  --net celebaHQ64 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std  --net celebaHQ64 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df   --net celebaHQ64 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw   --net celebaHQ64 --nr -1

python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net celebaHQ128 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim  --net celebaHQ128 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd  --net celebaHQ128 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std  --net celebaHQ128 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df   --net celebaHQ128 --nr -1
python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw   --net celebaHQ128 --nr -1




# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std --net cif10 --nr 0
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std --net cif10 --nr 1
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std --net cif10 --nr 2
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std --net cif10 --nr 3
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std --net cif10 --nr 4
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std --net cif10 --nr 5
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std --net cif10 --nr 6
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std --net cif10 --nr 7
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std --net cif10 --nr 8
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std --net cif10 --nr 9
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std --net cif10 --nr 10
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std --net cif10 --nr -1
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack std --net cif10 --nr 12


# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net cif10 --nr 0
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net cif10 --nr 1
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net cif10 --nr 2
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net cif10 --nr 3
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net cif10 --nr 4
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net cif10 --nr 5
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net cif10 --nr 6
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net cif10 --nr 7
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net cif10 --nr 8
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net cif10 --nr 9
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net cif10 --nr 10
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net cif10 --nr -1
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack fgsm --net cif10 --nr 12

# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim --net cif10 --nr 0
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim --net cif10 --nr 1
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim --net cif10 --nr 2
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim --net cif10 --nr 3
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim --net cif10 --nr 4
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim --net cif10 --nr 5
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim --net cif10 --nr 6
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim --net cif10 --nr 7
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim --net cif10 --nr 8
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim --net cif10 --nr 9
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim --net cif10 --nr 10
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim --net cif10 --nr -1
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack bim --net cif10 --nr 12

# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd --net cif10 --nr 0
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd --net cif10 --nr 1
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd --net cif10 --nr 2
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd --net cif10 --nr 3
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd --net cif10 --nr 4
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd --net cif10 --nr 5
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd --net cif10 --nr 6
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd --net cif10 --nr 7
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd --net cif10 --nr 8
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd --net cif10 --nr 9
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd --net cif10 --nr 10
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd --net cif10 --nr -1
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack pgd --net cif10 --nr 12

# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df --net cif10 --nr 0
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df --net cif10 --nr 1
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df --net cif10 --nr 2
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df --net cif10 --nr 3
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df --net cif10 --nr 4
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df --net cif10 --nr 5
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df --net cif10 --nr 6
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df --net cif10 --nr 7
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df --net cif10 --nr 8
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df --net cif10 --nr 9
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df --net cif10 --nr 10
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df --net cif10 --nr -1
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack df --net cif10 --nr 12

# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw --net cif10 --nr 0
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw --net cif10 --nr 1
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw --net cif10 --nr 2
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw --net cif10 --nr 3
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw --net cif10 --nr 4
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw --net cif10 --nr 5
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw --net cif10 --nr 6
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw --net cif10 --nr 7
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw --net cif10 --nr 8
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw --net cif10 --nr 9
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw --net cif10 --nr 10
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw --net cif10 --nr -1
# python -u extract_characteristics_imagenet32.py --detector InputPFS --attack cw --net cif10 --nr 12



#-----------------------------------------------------------------------------------------------------------------------------------
echo "finished experiment"

exit 0
#-----------------------------------------------------------------------------------------------------------------------------------