#!/bin/bash



# define python command including script and all parameters to execute
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --attack bim --net cif10vgg --nr 0"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --attack bim --net cif10vgg --nr 1"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --net cif10 --detector LayerPFS --attack pgd"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --net cif10 --detector LayerPFS --attack df"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --net cif10 --detector LayerPFS --attack cw"
# PYTHON_COMMAND="python -u extract_characteristics_imagenet32.py --net cif10 --detector LayerPFS --attack std"

# NOTE: the "-u" flag is essential as we want the output of the python script directly and not the moment the buffer it full
#-----------------------------------------------------------------------------------------------------------------------------------
for classifier in LR RF
do
    for nrclass in 25  50  100  250  1000
    do
        for att in fgsm bim  pgd df cw
        do
            for nrsamples in 300 500 1000 1200 1500 2000
            do
                echo " nrclasses: $nrclass classifier: $classifier #samples: $nrsamples attack: $att"
                # python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes $nrclass --clf $classifier --net imagenet32 --wanted_samples $nrsamples --attack $att 
            done
        done
    done
done

# echo "WB LR #############################################################################"

# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 10 --clf LR --net imagenet32 --attack fgsm 
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 10 --clf LR --net imagenet32 --attack bim
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 10 --clf LR --net imagenet32 --attack pgd
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 10 --clf LR --net imagenet32 --attack df
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 10 --clf LR --net imagenet32 --attack cw

# echo "WB RF #############################################################################"

# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 10 --clf RF --net imagenet32 --attack fgsm 
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 10 --clf RF --net imagenet32 --attack bim
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 10 --clf RF --net imagenet32 --attack pgd
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 10 --clf RF --net imagenet32 --attack df
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 10 --clf RF --net imagenet32 --attack cw

# echo "WB LR #############################################################################"

# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 25 --clf LR --net imagenet32 --wanted_samples 1500 --attack fgsm 
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 25 --clf LR --net imagenet32 --wanted_samples 1500 --attack bim
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 25 --clf LR --net imagenet32 --wanted_samples 1500 --attack pgd
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 25 --clf LR --net imagenet32 --wanted_samples 1500 --attack df
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 25 --clf LR --net imagenet32 --wanted_samples 1500 --attack cw

# echo "WB RF #############################################################################"

# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 25 --clf RF --net imagenet32 --wanted_samples 1500 --attack fgsm 
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 25 --clf RF --net imagenet32 --wanted_samples 1500 --attack bim
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 25 --clf RF --net imagenet32 --wanted_samples 1500 --attack pgd
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 25 --clf RF --net imagenet32 --wanted_samples 1500 --attack df
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 25 --clf RF --net imagenet32 --wanted_samples 1500 --attack cw

# echo "WB LR #############################################################################"

# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 50 --clf LR --net imagenet32 --wanted_samples 1500 --attack fgsm 
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 50 --clf LR --net imagenet32 --wanted_samples 1500 --attack bim
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 50 --clf LR --net imagenet32 --wanted_samples 1500 --attack pgd
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 50 --clf LR --net imagenet32 --wanted_samples 1500 --attack df
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 50 --clf LR --net imagenet32 --wanted_samples 1500 --attack cw

# echo "WB RF #############################################################################"

# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 50 --clf RF --net imagenet32 --wanted_samples 1500 --attack fgsm 
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 50 --clf RF --net imagenet32 --wanted_samples 1500 --attack bim
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 50 --clf RF --net imagenet32 --wanted_samples 1500 --attack pgd
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 50 --clf RF --net imagenet32 --wanted_samples 1500 --attack df
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 50 --clf RF --net imagenet32 --wanted_samples 1500 --attack cw


# echo "WB LR #############################################################################"

# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 75 --clf LR --net imagenet32 --wanted_samples 1500 --attack fgsm 
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 75 --clf LR --net imagenet32 --wanted_samples 1500 --attack bim
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 75 --clf LR --net imagenet32 --wanted_samples 1500 --attack pgd
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 75 --clf LR --net imagenet32 --wanted_samples 1500 --attack df
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 75 --clf LR --net imagenet32 --wanted_samples 1500 --attack cw

# echo "WB RF #############################################################################"

# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 75 --clf RF --net imagenet32 --wanted_samples 1500 --attack fgsm 
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 75 --clf RF --net imagenet32 --wanted_samples 1500 --attack bim
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 75 --clf RF --net imagenet32 --wanted_samples 1500 --attack pgd
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 75 --clf RF --net imagenet32 --wanted_samples 1500 --attack df
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 75 --clf RF --net imagenet32 --wanted_samples 1500 --attack cw

# echo "WB LR #############################################################################"

# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 100 --clf LR --net imagenet32 --wanted_samples 1500 --attack fgsm 
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 100 --clf LR --net imagenet32 --wanted_samples 1500 --attack bim
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 100 --clf LR --net imagenet32 --wanted_samples 1500 --attack pgd
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 100 --clf LR --net imagenet32 --wanted_samples 1500 --attack df
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 100 --clf LR --net imagenet32 --wanted_samples 1500 --attack cw

# echo "WB RF #############################################################################"

# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 100 --clf RF --net imagenet32 --wanted_samples 1500 --attack fgsm 
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 100 --clf RF --net imagenet32 --wanted_samples 1500 --attack bim
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 100 --clf RF --net imagenet32 --wanted_samples 1500 --attack pgd
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 100 --clf RF --net imagenet32 --wanted_samples 1500 --attack df
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 100 --clf RF --net imagenet32 --wanted_samples 1500 --attack cw


# echo "WB LR #############################################################################"

# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 250 --clf LR --net imagenet32 --wanted_samples 1500 --attack fgsm 
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 250 --clf LR --net imagenet32 --wanted_samples 1500 --attack bim
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 250 --clf LR --net imagenet32 --wanted_samples 1500 --attack pgd
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 250 --clf LR --net imagenet32 --wanted_samples 1500 --attack df
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 250 --clf LR --net imagenet32 --wanted_samples 1500 --attack cw

# echo "WB RF #############################################################################"

# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 250 --clf RF --net imagenet32 --wanted_samples 1500 --attack fgsm 
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 250 --clf RF --net imagenet32 --wanted_samples 1500 --attack bim
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 250 --clf RF --net imagenet32 --wanted_samples 1500 --attack pgd
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 250 --clf RF --net imagenet32 --wanted_samples 1500 --attack df
# python -u detect_adversarials_imagenet32.py --detector LayerMFS --num_classes 250 --clf RF --net imagenet32 --wanted_samples 1500 --attack cw



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
# python detect_adversarials_imagenet32.py --attack fgsm --nr 11
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
# python detect_adversarials_imagenet32.py --attack bim --nr 11
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
# python detect_adversarials_imagenet32.py --attack pgd --nr 11
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
# python detect_adversarials_imagenet32.py --attack df --nr 11
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
# python detect_adversarials_imagenet32.py --attack cw --nr 11
# python detect_adversarials_imagenet32.py --attack cw --nr 12


echo "#############################################################################"


#-----------------------------------------------------------------------------------------------------------------------------------
echo  "finished experiment"

exit 0
#-----------------------------------------------------------------------------------------------------------------------------------