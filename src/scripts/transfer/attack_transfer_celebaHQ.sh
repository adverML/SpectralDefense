#!/bin/bash






######## celebaHQ32
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval bim --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval pgd --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval std --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval df  --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval cw  --detector InputMFS --clf LR

python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack bim  --attack_eval pgd --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack bim  --attack_eval std --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack bim  --attack_eval df  --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack bim  --attack_eval cw  --detector InputMFS --clf LR

python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack pgd  --attack_eval std --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack pgd  --attack_eval df  --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack pgd  --attack_eval cw  --detector InputMFS --clf LR

python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack std  --attack_eval df  --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack std  --attack_eval cw  --detector InputMFS --clf LR

python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack df   --attack_eval cw  --detector InputMFS --clf LR


# Random Forest
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval bim --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval pgd --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval std --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval df  --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval cw  --detector InputMFS --clf RF

python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack bim  --attack_eval pgd --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack bim  --attack_eval std --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack bim  --attack_eval df  --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack bim  --attack_eval cw  --detector InputMFS --clf RF

python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack pgd  --attack_eval std --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack pgd  --attack_eval df  --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack pgd  --attack_eval cw  --detector InputMFS --clf RF

python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack std  --attack_eval df  --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack std  --attack_eval cw  --detector InputMFS --clf RF

python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack df   --attack_eval cw  --detector InputMFS --clf RF


######## celebaHQ64

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval bim --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval pgd --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval std --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval df  --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval cw  --detector InputMFS --clf LR

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack bim  --attack_eval pgd --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack bim  --attack_eval std --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack bim  --attack_eval df  --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack bim  --attack_eval cw  --detector InputMFS --clf LR

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack pgd  --attack_eval std --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack pgd  --attack_eval df  --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack pgd  --attack_eval cw  --detector InputMFS --clf LR

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack std  --attack_eval df  --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack std  --attack_eval cw  --detector InputMFS --clf LR

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack df   --attack_eval cw  --detector InputMFS --clf LR


# Random Forest
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval bim --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval pgd --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval std --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval df  --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval cw  --detector InputMFS --clf RF

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack bim  --attack_eval pgd --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack bim  --attack_eval std --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack bim  --attack_eval df  --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack bim  --attack_eval cw  --detector InputMFS --clf RF

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack pgd  --attack_eval std --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack pgd  --attack_eval df  --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack pgd  --attack_eval cw  --detector InputMFS --clf RF

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack std  --attack_eval df  --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack std  --attack_eval cw  --detector InputMFS --clf RF

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack df   --attack_eval cw  --detector InputMFS --clf RF


######## celebaHQ128

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval bim --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval pgd --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval std --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval df  --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval cw  --detector InputMFS --clf LR

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack bim  --attack_eval pgd --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack bim  --attack_eval std --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack bim  --attack_eval df  --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack bim  --attack_eval cw  --detector InputMFS --clf LR

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack pgd  --attack_eval std --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack pgd  --attack_eval df  --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack pgd  --attack_eval cw  --detector InputMFS --clf LR

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack std  --attack_eval df  --detector InputMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack std  --attack_eval cw  --detector InputMFS --clf LR

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack df   --attack_eval cw  --detector InputMFS --clf LR


# Random Forest
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval bim --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval pgd --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval std --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval df  --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval cw  --detector InputMFS --clf RF

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack bim  --attack_eval pgd --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack bim  --attack_eval std --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack bim  --attack_eval df  --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack bim  --attack_eval cw  --detector InputMFS --clf RF

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack pgd  --attack_eval std --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack pgd  --attack_eval df  --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack pgd  --attack_eval cw  --detector InputMFS --clf RF

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack std  --attack_eval df  --detector InputMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack std  --attack_eval cw  --detector InputMFS --clf RF

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack df   --attack_eval cw  --detector InputMFS --clf RF



######################## WB



######## celebaHQ32
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval bim --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval pgd --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval std --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval df  --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval cw  --detector LayerMFS --clf LR

python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack bim  --attack_eval pgd --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack bim  --attack_eval std --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack bim  --attack_eval df  --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack bim  --attack_eval cw  --detector LayerMFS --clf LR

python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack pgd  --attack_eval std --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack pgd  --attack_eval df  --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack pgd  --attack_eval cw  --detector LayerMFS --clf LR

python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack std  --attack_eval df  --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack std  --attack_eval cw  --detector LayerMFS --clf LR

python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack df   --attack_eval cw  --detector LayerMFS --clf LR


# Random Forest
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval bim --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval pgd --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval std --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval df  --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack fgsm --attack_eval cw  --detector LayerMFS --clf RF

python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack bim  --attack_eval pgd --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack bim  --attack_eval std --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack bim  --attack_eval df  --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack bim  --attack_eval cw  --detector LayerMFS --clf RF

python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack pgd  --attack_eval std --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack pgd  --attack_eval df  --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack pgd  --attack_eval cw  --detector LayerMFS --clf RF

python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack std  --attack_eval df  --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack std  --attack_eval cw  --detector LayerMFS --clf RF

python attack_transfer.py --net celebaHQ32  --num_classes 4 --attack df   --attack_eval cw  --detector LayerMFS --clf RF


######## celebaHQ64

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval bim --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval pgd --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval std --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval df  --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval cw  --detector LayerMFS --clf LR

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack bim  --attack_eval pgd --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack bim  --attack_eval std --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack bim  --attack_eval df  --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack bim  --attack_eval cw  --detector LayerMFS --clf LR

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack pgd  --attack_eval std --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack pgd  --attack_eval df  --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack pgd  --attack_eval cw  --detector LayerMFS --clf LR

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack std  --attack_eval df  --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack std  --attack_eval cw  --detector LayerMFS --clf LR

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack df   --attack_eval cw  --detector LayerMFS --clf LR


# Random Forest
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval bim --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval pgd --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval std --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval df  --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack fgsm --attack_eval cw  --detector LayerMFS --clf RF

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack bim  --attack_eval pgd --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack bim  --attack_eval std --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack bim  --attack_eval df  --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack bim  --attack_eval cw  --detector LayerMFS --clf RF

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack pgd  --attack_eval std --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack pgd  --attack_eval df  --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack pgd  --attack_eval cw  --detector LayerMFS --clf RF

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack std  --attack_eval df  --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack std  --attack_eval cw  --detector LayerMFS --clf RF

python attack_transfer.py --net celebaHQ64 --num_classes 4 --attack df   --attack_eval cw  --detector LayerMFS --clf RF


######## celebaHQ128

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval bim --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval pgd --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval std --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval df  --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval cw  --detector LayerMFS --clf LR

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack bim  --attack_eval pgd --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack bim  --attack_eval std --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack bim  --attack_eval df  --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack bim  --attack_eval cw  --detector LayerMFS --clf LR

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack pgd  --attack_eval std --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack pgd  --attack_eval df  --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack pgd  --attack_eval cw  --detector LayerMFS --clf LR

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack std  --attack_eval df  --detector LayerMFS --clf LR
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack std  --attack_eval cw  --detector LayerMFS --clf LR

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack df   --attack_eval cw  --detector LayerMFS --clf LR


# Random Forest
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval bim --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval pgd --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval std --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval df  --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack fgsm --attack_eval cw  --detector LayerMFS --clf RF

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack bim  --attack_eval pgd --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack bim  --attack_eval std --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack bim  --attack_eval df  --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack bim  --attack_eval cw  --detector LayerMFS --clf RF

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack pgd  --attack_eval std --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack pgd  --attack_eval df  --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack pgd  --attack_eval cw  --detector LayerMFS --clf RF

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack std  --attack_eval df  --detector LayerMFS --clf RF
python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack std  --attack_eval cw  --detector LayerMFS --clf RF

python attack_transfer.py --net celebaHQ128 --num_classes 4 --attack df   --attack_eval cw  --detector LayerMFS --clf RF