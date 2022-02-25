#!/bin/bash

# bash detect_serial_imfs.sh &>> log_evaluation/imfs.log

# define python command including script and all parameters to execute
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --attack bim --net cif10vgg --nr 0"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --attack bim --net cif10vgg --nr 1"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --net cif10 --detector InputMFS --attack pgd"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --net cif10 --detector InputMFS --attack df"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --net cif10 --detector InputMFS --attack cw"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --net cif10 --detector InputMFS --attack std"

# NOTE: the "-u" flag is essential as we want the output of the python script directly and not the moment the buffer it full
#-----------------------------------------------------------------------------------------------------------------------------------


python detect_adversarials_imagenet32.py --net cif10 --detector InputMFS --wanted_samples 1500 --attack fgsm --nr -1
python detect_adversarials_imagenet32.py --net cif10 --detector InputMFS --wanted_samples 1500 --attack bim --nr -1
python detect_adversarials_imagenet32.py --net cif10 --detector InputMFS --wanted_samples 1500 --attack pgd --nr -1
python detect_adversarials_imagenet32.py --net cif10 --detector InputMFS --wanted_samples 1500 --attack std --nr -1
python detect_adversarials_imagenet32.py --net cif10 --detector InputMFS --wanted_samples 1500 --attack df  --nr -1
python detect_adversarials_imagenet32.py --net cif10 --detector InputMFS --wanted_samples 1500 --attack cw  --nr -1


echo "#############################################################################"

python detect_adversarials_imagenet32.py --net cif10 --detector InputMFS --wanted_samples 1500 --attack fgsm --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net cif10 --detector InputMFS --wanted_samples 1500 --attack bim --nr  -1 --clf RF
python detect_adversarials_imagenet32.py --net cif10 --detector InputMFS --wanted_samples 1500 --attack pgd --nr  -1 --clf RF
python detect_adversarials_imagenet32.py --net cif10 --detector InputMFS --wanted_samples 1500 --attack std --nr  -1 --clf RF
python detect_adversarials_imagenet32.py --net cif10 --detector InputMFS --wanted_samples 1500 --attack df  --nr  -1 --clf RF
python detect_adversarials_imagenet32.py --net cif10 --detector InputMFS --wanted_samples 1500 --attack cw  --nr  -1 --clf RF


echo "#############################################################################"

python detect_adversarials_imagenet32.py --net cif100 --detector InputMFS --wanted_samples 1500 --attack fgsm --nr -1
python detect_adversarials_imagenet32.py --net cif100 --detector InputMFS --wanted_samples 1500 --attack bim --nr -1
python detect_adversarials_imagenet32.py --net cif100 --detector InputMFS --wanted_samples 1500 --attack pgd --nr -1
python detect_adversarials_imagenet32.py --net cif100 --detector InputMFS --wanted_samples 1500 --attack std --nr -1
python detect_adversarials_imagenet32.py --net cif100 --detector InputMFS --wanted_samples 1500 --attack df --nr -1
python detect_adversarials_imagenet32.py --net cif100 --detector InputMFS --wanted_samples 1500 --attack cw --nr -1


echo "#############################################################################"

python detect_adversarials_imagenet32.py --net cif100 --detector InputMFS --wanted_samples 1500 --attack fgsm --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net cif100 --detector InputMFS --wanted_samples 1500 --attack bim --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net cif100 --detector InputMFS --wanted_samples 1500 --attack pgd --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net cif100 --detector InputMFS --wanted_samples 1500 --attack std --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net cif100 --detector InputMFS --wanted_samples 1500 --attack df --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net cif100 --detector InputMFS --wanted_samples 1500 --attack cw --nr -1 --clf RF

echo "#############################################################################"



python detect_adversarials_imagenet32.py --net imagenet32 --detector InputMFS --wanted_samples 1500 --attack fgsm --nr -1
python detect_adversarials_imagenet32.py --net imagenet32 --detector InputMFS --wanted_samples 1500 --attack bim --nr -1
python detect_adversarials_imagenet32.py --net imagenet32 --detector InputMFS --wanted_samples 1500 --attack pgd --nr -1
python detect_adversarials_imagenet32.py --net imagenet32 --detector InputMFS --wanted_samples 1500 --attack std --nr -1
python detect_adversarials_imagenet32.py --net imagenet32 --detector InputMFS --wanted_samples 1500 --attack df --nr -1
python detect_adversarials_imagenet32.py --net imagenet32 --detector InputMFS --wanted_samples 1500 --attack cw --nr -1

echo "#############################################################################"

python detect_adversarials_imagenet32.py --net imagenet32 --detector InputMFS --wanted_samples 1500 --attack fgsm --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net imagenet32 --detector InputMFS --wanted_samples 1500 --attack bim --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net imagenet32 --detector InputMFS --wanted_samples 1500 --attack pgd --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net imagenet32 --detector InputMFS --wanted_samples 1500 --attack std --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net imagenet32 --detector InputMFS --wanted_samples 1500 --attack df --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net imagenet32 --detector InputMFS --wanted_samples 1500 --attack cw --nr -1 --clf RF

echo "#############################################################################"

python detect_adversarials_imagenet32.py --net imagenet64 --detector InputMFS --wanted_samples 1500 --attack fgsm --nr -1
python detect_adversarials_imagenet32.py --net imagenet64 --detector InputMFS --wanted_samples 1500 --attack bim --nr -1
python detect_adversarials_imagenet32.py --net imagenet64 --detector InputMFS --wanted_samples 1500 --attack pgd --nr -1
python detect_adversarials_imagenet32.py --net imagenet64 --detector InputMFS --wanted_samples 1500 --attack std --nr -1
python detect_adversarials_imagenet32.py --net imagenet64 --detector InputMFS --wanted_samples 1500 --attack df --nr -1
python detect_adversarials_imagenet32.py --net imagenet64 --detector InputMFS --wanted_samples 1500 --attack cw --nr -1


echo "#############################################################################"

python detect_adversarials_imagenet32.py --net imagenet128 --detector InputMFS --wanted_samples 1500 --attack fgsm --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net imagenet128 --detector InputMFS --wanted_samples 1500 --attack bim --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net imagenet128 --detector InputMFS --wanted_samples 1500 --attack pgd --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net imagenet128 --detector InputMFS --wanted_samples 1500 --attack std --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net imagenet128 --detector InputMFS --wanted_samples 1500 --attack df --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net imagenet128 --detector InputMFS --wanted_samples 1500 --attack cw --nr -1 --clf RF

echo "#############################################################################"


python detect_adversarials_imagenet32.py --net celebaHQ32 --detector InputMFS --wanted_samples 1500 --attack fgsm --nr -1
python detect_adversarials_imagenet32.py --net celebaHQ32 --detector InputMFS --wanted_samples 1500 --attack bim --nr -1
python detect_adversarials_imagenet32.py --net celebaHQ32 --detector InputMFS --wanted_samples 1500 --attack pgd --nr -1
python detect_adversarials_imagenet32.py --net celebaHQ32 --detector InputMFS --wanted_samples 1500 --attack std --nr -1
python detect_adversarials_imagenet32.py --net celebaHQ32 --detector InputMFS --wanted_samples 1500 --attack df --nr -1
python detect_adversarials_imagenet32.py --net celebaHQ32 --detector InputMFS --wanted_samples 1500 --attack cw --nr -1


echo "#############################################################################"

python detect_adversarials_imagenet32.py --net celebaHQ32 --detector InputMFS --wanted_samples 1500 --attack fgsm --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net celebaHQ32 --detector InputMFS --wanted_samples 1500 --attack bim --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net celebaHQ32 --detector InputMFS --wanted_samples 1500 --attack pgd --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net celebaHQ32 --detector InputMFS --wanted_samples 1500 --attack std --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net celebaHQ32 --detector InputMFS --wanted_samples 1500 --attack df --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net celebaHQ32 --detector InputMFS --wanted_samples 1500 --attack cw --nr -1 --clf RF

echo "#############################################################################"

python detect_adversarials_imagenet32.py --net celebaHQ64 --detector InputMFS --wanted_samples 1500 --attack fgsm --nr -1
python detect_adversarials_imagenet32.py --net celebaHQ64 --detector InputMFS --wanted_samples 1500 --attack bim --nr -1
python detect_adversarials_imagenet32.py --net celebaHQ64 --detector InputMFS --wanted_samples 1500 --attack pgd --nr -1
python detect_adversarials_imagenet32.py --net celebaHQ64 --detector InputMFS --wanted_samples 1500 --attack std --nr -1
python detect_adversarials_imagenet32.py --net celebaHQ64 --detector InputMFS --wanted_samples 1500 --attack df --nr -1
python detect_adversarials_imagenet32.py --net celebaHQ64 --detector InputMFS --wanted_samples 1500 --attack cw --nr -1


echo "#############################################################################"

python detect_adversarials_imagenet32.py --net celebaHQ64 --detector InputMFS --wanted_samples 1500 --attack fgsm --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net celebaHQ64 --detector InputMFS --wanted_samples 1500 --attack bim --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net celebaHQ64 --detector InputMFS --wanted_samples 1500 --attack pgd --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net celebaHQ64 --detector InputMFS --wanted_samples 1500 --attack std --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net celebaHQ64 --detector InputMFS --wanted_samples 1500 --attack df --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net celebaHQ64 --detector InputMFS --wanted_samples 1500 --attack cw --nr -1 --clf RF

echo "#############################################################################"



python detect_adversarials_imagenet32.py --net celebaHQ128 --detector InputMFS --wanted_samples 1500 --attack fgsm --nr -1
python detect_adversarials_imagenet32.py --net celebaHQ128 --detector InputMFS --wanted_samples 1500 --attack bim --nr -1
python detect_adversarials_imagenet32.py --net celebaHQ128 --detector InputMFS --wanted_samples 1500 --attack pgd --nr -1
python detect_adversarials_imagenet32.py --net celebaHQ128 --detector InputMFS --wanted_samples 1500 --attack std --nr -1
python detect_adversarials_imagenet32.py --net celebaHQ128 --detector InputMFS --wanted_samples 1500 --attack df --nr -1
python detect_adversarials_imagenet32.py --net celebaHQ128 --detector InputMFS --wanted_samples 1500 --attack cw --nr -1


echo "#############################################################################"

python detect_adversarials_imagenet32.py --net celebaHQ128 --detector InputMFS --wanted_samples 1500 --attack fgsm --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net celebaHQ128 --detector InputMFS --wanted_samples 1500 --attack bim --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net celebaHQ128 --detector InputMFS --wanted_samples 1500 --attack pgd --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net celebaHQ128 --detector InputMFS --wanted_samples 1500 --attack std --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net celebaHQ128 --detector InputMFS --wanted_samples 1500 --attack df --nr -1 --clf RF
python detect_adversarials_imagenet32.py --net celebaHQ128 --detector InputMFS --wanted_samples 1500 --attack cw --nr -1 --clf RF

echo "#############################################################################"




# echo "#############################################################################"


# python detect_adversarials_imagenet32.py --net cif10vgg --detector LayerMFS --attack std --nr 0
# python detect_adversarials_imagenet32.py --net cif10vgg --detector LayerMFS --attack std --nr 1
# python detect_adversarials_imagenet32.py --net cif10vgg --detector LayerMFS --attack std --nr 2
# python detect_adversarials_imagenet32.py --net cif10vgg --detector LayerMFS --attack std --nr 3
# python detect_adversarials_imagenet32.py --net cif10vgg --detector LayerMFS --attack std --nr 4
# python detect_adversarials_imagenet32.py --net cif10vgg --detector LayerMFS --attack std --nr 5
# python detect_adversarials_imagenet32.py --net cif10vgg --detector LayerMFS --attack std --nr 6
# python detect_adversarials_imagenet32.py --net cif10vgg --detector LayerMFS --attack std --nr 7
# python detect_adversarials_imagenet32.py --net cif10vgg --detector LayerMFS --attack std --nr 8
# python detect_adversarials_imagenet32.py --net cif10vgg --detector LayerMFS --attack std --nr 9
# python detect_adversarials_imagenet32.py --net cif10vgg --detector LayerMFS --attack std --nr 10
# python detect_adversarials_imagenet32.py --net cif10vgg --detector LayerMFS --attack std --nr -1
# python detect_adversarials_imagenet32.py --net cif10vgg --detector LayerMFS --attack std --nr 12

# echo "#############################################################################"

# python detect_adversarials_imagenet32.py --net cif10vgg --detector InputMFS --attack std --nr 0
# python detect_adversarials_imagenet32.py --net cif10vgg --detector InputMFS --attack std --nr 1
# python detect_adversarials_imagenet32.py --net cif10vgg --detector InputMFS --attack std --nr 2
# python detect_adversarials_imagenet32.py --net cif10vgg --detector InputMFS --attack std --nr 3
# python detect_adversarials_imagenet32.py --net cif10vgg --detector InputMFS --attack std --nr 4
# python detect_adversarials_imagenet32.py --net cif10vgg --detector InputMFS --attack std --nr 5
# python detect_adversarials_imagenet32.py --net cif10vgg --detector InputMFS --attack std --nr 6
# python detect_adversarials_imagenet32.py --net cif10vgg --detector InputMFS --attack std --nr 7
# python detect_adversarials_imagenet32.py --net cif10vgg --detector InputMFS --attack std --nr 8
# python detect_adversarials_imagenet32.py --net cif10vgg --detector InputMFS --attack std --nr 9
# python detect_adversarials_imagenet32.py --net cif10vgg --detector InputMFS --attack std --nr 10
# python detect_adversarials_imagenet32.py --net cif10vgg --detector InputMFS --attack std --nr -1
# python detect_adversarials_imagenet32.py --net cif10vgg --detector InputMFS --attack std --nr 12



# python -u detect_adversarials_imagenet32.py --detector InputMFS --num_classes 25 --clf RF --net imagenet32 --attack fgsm 
# python -u detect_adversarials_imagenet32.py --detector InputMFS --num_classes 25 --clf RF --net imagenet32 --attack bim
# python -u detect_adversarials_imagenet32.py --detector InputMFS --num_classes 25 --clf RF --net imagenet32 --attack pgd
# python -u detect_adversarials_imagenet32.py --detector InputMFS --num_classes 25 --clf RF --net imagenet32 --attack df
# python -u detect_adversarials_imagenet32.py --detector InputMFS --num_classes 25 --clf RF --net imagenet32 --attack cw



# python detect_adversarials_imagenet32.py --attack fgsm --nr 0
# python detect_adversarials_imagenet32.py --attack fgsm --nr 1
# python detect_adversarials_imagenet32.py --attack fgsm --nr 2
# python detect_adversarials_imagenet32.py --attack fgsm --nr 3
# python detect_adversarials_imagenet32.py --attack fgsm --nr 4
# python detect_adversarials_imagenet32.py --attack fgsm --nr 5
# python detect_adversarials_imagenet32.py --attack fgsm --nr 6
# python detect_adversarials_imagenet32.py --attack fgsm --nr 7
# python detect_adversarials_imagenet32.py --attack fgsm --nr 8
# python detect_adversarials_imagenet32.py --attack fgsm --nr 9
# python detect_adversarials_imagenet32.py --attack fgsm --nr 10
# python detect_adversarials_imagenet32.py --attack fgsm --nr -1
# python detect_adversarials_imagenet32.py --attack fgsm --nr 12

# echo "#############################################################################"

# python detect_adversarials_imagenet32.py --attack bim --nr 0
# python detect_adversarials_imagenet32.py --attack bim --nr 1
# python detect_adversarials_imagenet32.py --attack bim --nr 2
# python detect_adversarials_imagenet32.py --attack bim --nr 3
# python detect_adversarials_imagenet32.py --attack bim --nr 4
# python detect_adversarials_imagenet32.py --attack bim --nr 5
# python detect_adversarials_imagenet32.py --attack bim --nr 6
# python detect_adversarials_imagenet32.py --attack bim --nr 7
# python detect_adversarials_imagenet32.py --attack bim --nr 8
# python detect_adversarials_imagenet32.py --attack bim --nr 9
# python detect_adversarials_imagenet32.py --attack bim --nr 10
# python detect_adversarials_imagenet32.py --attack bim --nr -1
# python detect_adversarials_imagenet32.py --attack bim --nr 12


echo "#############################################################################"


# python detect_adversarials_imagenet32.py --attack pgd --nr 0
# python detect_adversarials_imagenet32.py --attack pgd --nr 1
# python detect_adversarials_imagenet32.py --attack pgd --nr 2
# python detect_adversarials_imagenet32.py --attack pgd --nr 3
# python detect_adversarials_imagenet32.py --attack pgd --nr 4
# python detect_adversarials_imagenet32.py --attack pgd --nr 5
# python detect_adversarials_imagenet32.py --attack pgd --nr 6
# python detect_adversarials_imagenet32.py --attack pgd --nr 7
# python detect_adversarials_imagenet32.py --attack pgd --nr 8
# python detect_adversarials_imagenet32.py --attack pgd --nr 9
# python detect_adversarials_imagenet32.py --attack pgd --nr 10
# python detect_adversarials_imagenet32.py --attack pgd --nr -1
# python detect_adversarials_imagenet32.py --attack pgd --nr 12


echo "#############################################################################"


# python detect_adversarials_imagenet32.py --attack df --nr 0
# python detect_adversarials_imagenet32.py --attack df --nr 1
# python detect_adversarials_imagenet32.py --attack df --nr 2
# python detect_adversarials_imagenet32.py --attack df --nr 3
# python detect_adversarials_imagenet32.py --attack df --nr 4
# python detect_adversarials_imagenet32.py --attack df --nr 5
# python detect_adversarials_imagenet32.py --attack df --nr 6
# python detect_adversarials_imagenet32.py --attack df --nr 7
# python detect_adversarials_imagenet32.py --attack df --nr 8
# python detect_adversarials_imagenet32.py --attack df --nr 9
# python detect_adversarials_imagenet32.py --attack df --nr 10
# python detect_adversarials_imagenet32.py --attack df --nr -1
# python detect_adversarials_imagenet32.py --attack df --nr 12

# echo "#############################################################################"


# python detect_adversarials_imagenet32.py --attack cw --nr 0
# python detect_adversarials_imagenet32.py --attack cw --nr 1
# python detect_adversarials_imagenet32.py --attack cw --nr 2
# python detect_adversarials_imagenet32.py --attack cw --nr 3
# python detect_adversarials_imagenet32.py --attack cw --nr 4
# python detect_adversarials_imagenet32.py --attack cw --nr 5
# python detect_adversarials_imagenet32.py --attack cw --nr 6
# python detect_adversarials_imagenet32.py --attack cw --nr 7
# python detect_adversarials_imagenet32.py --attack cw --nr 8
# python detect_adversarials_imagenet32.py --attack cw --nr 9
# python detect_adversarials_imagenet32.py --attack cw --nr 10
# python detect_adversarials_imagenet32.py --attack cw --nr -1
# python detect_adversarials_imagenet32.py --attack cw --nr 12


echo "#############################################################################"


#-----------------------------------------------------------------------------------------------------------------------------------
echo  "finished experiment"

exit 0
#-----------------------------------------------------------------------------------------------------------------------------------