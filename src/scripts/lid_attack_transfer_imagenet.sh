#!/bin/bash


######## imagenet
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval bim --detector LID --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval pgd --detector LID --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval std --detector LID --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval df  --detector LID --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval cw  --detector LID --clf LR

# python attack_transfer.py --net imagenet  --num_classes 1000 --attack bim  --attack_eval pgd --detector LID --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack bim  --attack_eval std --detector LID --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack bim  --attack_eval df  --detector LID --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack bim  --attack_eval cw  --detector LID --clf LR

# python attack_transfer.py --net imagenet  --num_classes 1000 --attack pgd  --attack_eval std --detector LID --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack pgd  --attack_eval df  --detector LID --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack pgd  --attack_eval cw  --detector LID --clf LR

# python attack_transfer.py --net imagenet  --num_classes 1000 --attack std  --attack_eval df  --detector LID --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack std  --attack_eval cw  --detector LID --clf LR

# python attack_transfer.py --net imagenet  --num_classes 1000 --attack df   --attack_eval cw  --detector LID --clf LR


# Random Forest
python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval bim --detector LID --clf RF
python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval pgd --detector LID --clf RF
python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval std --detector LID --clf RF
python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval df  --detector LID --clf RF
python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval cw  --detector LID --clf RF

python attack_transfer.py --net imagenet  --num_classes 1000 --attack bim  --attack_eval pgd --detector LID --clf RF
python attack_transfer.py --net imagenet  --num_classes 1000 --attack bim  --attack_eval std --detector LID --clf RF
python attack_transfer.py --net imagenet  --num_classes 1000 --attack bim  --attack_eval df  --detector LID --clf RF
python attack_transfer.py --net imagenet  --num_classes 1000 --attack bim  --attack_eval cw  --detector LID --clf RF

python attack_transfer.py --net imagenet  --num_classes 1000 --attack pgd  --attack_eval std --detector LID --clf RF
python attack_transfer.py --net imagenet  --num_classes 1000 --attack pgd  --attack_eval df  --detector LID --clf RF
python attack_transfer.py --net imagenet  --num_classes 1000 --attack pgd  --attack_eval cw  --detector LID --clf RF

python attack_transfer.py --net imagenet  --num_classes 1000 --attack std  --attack_eval df  --detector LID --clf RF
python attack_transfer.py --net imagenet  --num_classes 1000 --attack std  --attack_eval cw  --detector LID --clf RF

python attack_transfer.py --net imagenet  --num_classes 1000 --attack df   --attack_eval cw  --detector LID --clf RF


######## imagenet64

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval bim --detector LID --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval pgd --detector LID --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval std --detector LID --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval df  --detector LID --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval cw  --detector LID --clf LR

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack bim  --attack_eval pgd --detector LID --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack bim  --attack_eval std --detector LID --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack bim  --attack_eval df  --detector LID --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack bim  --attack_eval cw  --detector LID --clf LR

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack pgd  --attack_eval std --detector LID --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack pgd  --attack_eval df  --detector LID --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack pgd  --attack_eval cw  --detector LID --clf LR

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack std  --attack_eval df  --detector LID --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack std  --attack_eval cw  --detector LID --clf LR

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack df   --attack_eval cw  --detector LID --clf LR


# # Random Forest
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval bim --detector LID --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval pgd --detector LID --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval std --detector LID --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval df  --detector LID --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval cw  --detector LID --clf RF

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack bim  --attack_eval pgd --detector LID --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack bim  --attack_eval std --detector LID --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack bim  --attack_eval df  --detector LID --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack bim  --attack_eval cw  --detector LID --clf RF

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack pgd  --attack_eval std --detector LID --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack pgd  --attack_eval df  --detector LID --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack pgd  --attack_eval cw  --detector LID --clf RF

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack std  --attack_eval df  --detector LID --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack std  --attack_eval cw  --detector LID --clf RF

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack df   --attack_eval cw  --detector LID --clf RF


######## imagenet128

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval bim --detector LID --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval pgd --detector LID --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval std --detector LID --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval df  --detector LID --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval cw  --detector LID --clf LR

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack bim  --attack_eval pgd --detector LID --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack bim  --attack_eval std --detector LID --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack bim  --attack_eval df  --detector LID --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack bim  --attack_eval cw  --detector LID --clf LR

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack pgd  --attack_eval std --detector LID --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack pgd  --attack_eval df  --detector LID --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack pgd  --attack_eval cw  --detector LID --clf LR

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack std  --attack_eval df  --detector LID --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack std  --attack_eval cw  --detector LID --clf LR

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack df   --attack_eval cw  --detector LID --clf LR


# # Random Forest
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval bim --detector LID --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval pgd --detector LID --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval std --detector LID --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval df  --detector LID --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval cw  --detector LID --clf RF

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack bim  --attack_eval pgd --detector LID --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack bim  --attack_eval std --detector LID --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack bim  --attack_eval df  --detector LID --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack bim  --attack_eval cw  --detector LID --clf RF

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack pgd  --attack_eval std --detector LID --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack pgd  --attack_eval df  --detector LID --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack pgd  --attack_eval cw  --detector LID --clf RF

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack std  --attack_eval df  --detector LID --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack std  --attack_eval cw  --detector LID --clf RF

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack df   --attack_eval cw  --detector LID --clf RF



######################## WB



# ######## imagenet
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval bim --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval pgd --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval std --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval df  --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval cw  --detector LayerMFS --clf LR

# python attack_transfer.py --net imagenet  --num_classes 1000 --attack bim  --attack_eval pgd --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack bim  --attack_eval std --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack bim  --attack_eval df  --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack bim  --attack_eval cw  --detector LayerMFS --clf LR

# python attack_transfer.py --net imagenet  --num_classes 1000 --attack pgd  --attack_eval std --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack pgd  --attack_eval df  --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack pgd  --attack_eval cw  --detector LayerMFS --clf LR

# python attack_transfer.py --net imagenet  --num_classes 1000 --attack std  --attack_eval df  --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack std  --attack_eval cw  --detector LayerMFS --clf LR

# python attack_transfer.py --net imagenet  --num_classes 1000 --attack df   --attack_eval cw  --detector LayerMFS --clf LR


# # Random Forest
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval bim --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval pgd --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval std --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval df  --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack fgsm --attack_eval cw  --detector LayerMFS --clf RF

# python attack_transfer.py --net imagenet  --num_classes 1000 --attack bim  --attack_eval pgd --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack bim  --attack_eval std --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack bim  --attack_eval df  --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack bim  --attack_eval cw  --detector LayerMFS --clf RF

# python attack_transfer.py --net imagenet  --num_classes 1000 --attack pgd  --attack_eval std --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack pgd  --attack_eval df  --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack pgd  --attack_eval cw  --detector LayerMFS --clf RF

# python attack_transfer.py --net imagenet  --num_classes 1000 --attack std  --attack_eval df  --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet  --num_classes 1000 --attack std  --attack_eval cw  --detector LayerMFS --clf RF

# python attack_transfer.py --net imagenet  --num_classes 1000 --attack df   --attack_eval cw  --detector LayerMFS --clf RF


######## imagenet64

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval bim --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval pgd --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval std --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval df  --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval cw  --detector LayerMFS --clf LR

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack bim  --attack_eval pgd --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack bim  --attack_eval std --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack bim  --attack_eval df  --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack bim  --attack_eval cw  --detector LayerMFS --clf LR

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack pgd  --attack_eval std --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack pgd  --attack_eval df  --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack pgd  --attack_eval cw  --detector LayerMFS --clf LR

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack std  --attack_eval df  --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack std  --attack_eval cw  --detector LayerMFS --clf LR

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack df   --attack_eval cw  --detector LayerMFS --clf LR


# # Random Forest
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval bim --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval pgd --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval std --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval df  --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack fgsm --attack_eval cw  --detector LayerMFS --clf RF

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack bim  --attack_eval pgd --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack bim  --attack_eval std --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack bim  --attack_eval df  --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack bim  --attack_eval cw  --detector LayerMFS --clf RF

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack pgd  --attack_eval std --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack pgd  --attack_eval df  --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack pgd  --attack_eval cw  --detector LayerMFS --clf RF

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack std  --attack_eval df  --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack std  --attack_eval cw  --detector LayerMFS --clf RF

# python attack_transfer.py --net imagenet64 --num_classes 1000 --attack df   --attack_eval cw  --detector LayerMFS --clf RF


# ######## imagenet128

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval bim --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval pgd --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval std --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval df  --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval cw  --detector LayerMFS --clf LR

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack bim  --attack_eval pgd --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack bim  --attack_eval std --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack bim  --attack_eval df  --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack bim  --attack_eval cw  --detector LayerMFS --clf LR

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack pgd  --attack_eval std --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack pgd  --attack_eval df  --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack pgd  --attack_eval cw  --detector LayerMFS --clf LR

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack std  --attack_eval df  --detector LayerMFS --clf LR
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack std  --attack_eval cw  --detector LayerMFS --clf LR

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack df   --attack_eval cw  --detector LayerMFS --clf LR


# # Random Forest
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval bim --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval pgd --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval std --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval df  --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack fgsm --attack_eval cw  --detector LayerMFS --clf RF

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack bim  --attack_eval pgd --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack bim  --attack_eval std --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack bim  --attack_eval df  --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack bim  --attack_eval cw  --detector LayerMFS --clf RF

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack pgd  --attack_eval std --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack pgd  --attack_eval df  --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack pgd  --attack_eval cw  --detector LayerMFS --clf RF

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack std  --attack_eval df  --detector LayerMFS --clf RF
# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack std  --attack_eval cw  --detector LayerMFS --clf RF

# python attack_transfer.py --net imagenet128 --num_classes 1000 --attack df   --attack_eval cw  --detector LayerMFS --clf RF