#!/bin/bash

######## cif10

python attack_transfer.py --attack fgsm --attack_eval bim --detector InputMFS --clf LR
python attack_transfer.py --attack fgsm --attack_eval pgd --detector InputMFS --clf LR
python attack_transfer.py --attack fgsm --attack_eval std --detector InputMFS --clf LR
python attack_transfer.py --attack fgsm --attack_eval df  --detector InputMFS --clf LR
python attack_transfer.py --attack fgsm --attack_eval cw  --detector InputMFS --clf LR

python attack_transfer.py --attack bim  --attack_eval pgd --detector InputMFS --clf LR
python attack_transfer.py --attack bim  --attack_eval std --detector InputMFS --clf LR
python attack_transfer.py --attack bim  --attack_eval df  --detector InputMFS --clf LR
python attack_transfer.py --attack bim  --attack_eval cw  --detector InputMFS --clf LR

python attack_transfer.py --attack pgd  --attack_eval std --detector InputMFS --clf LR
python attack_transfer.py --attack pgd  --attack_eval df  --detector InputMFS --clf LR
python attack_transfer.py --attack pgd  --attack_eval cw  --detector InputMFS --clf LR

python attack_transfer.py --attack std  --attack_eval df  --detector InputMFS --clf LR
python attack_transfer.py --attack std  --attack_eval cw  --detector InputMFS --clf LR

python attack_transfer.py --attack df   --attack_eval cw  --detector InputMFS --clf LR


# Random Forest
python attack_transfer.py --attack fgsm --attack_eval bim --detector InputMFS --clf RF
python attack_transfer.py --attack fgsm --attack_eval pgd --detector InputMFS --clf RF
python attack_transfer.py --attack fgsm --attack_eval std --detector InputMFS --clf RF
python attack_transfer.py --attack fgsm --attack_eval df  --detector InputMFS --clf RF
python attack_transfer.py --attack fgsm --attack_eval cw  --detector InputMFS --clf RF

python attack_transfer.py --attack bim  --attack_eval pgd --detector InputMFS --clf RF
python attack_transfer.py --attack bim  --attack_eval std --detector InputMFS --clf RF
python attack_transfer.py --attack bim  --attack_eval df  --detector InputMFS --clf RF
python attack_transfer.py --attack bim  --attack_eval cw  --detector InputMFS --clf RF

python attack_transfer.py --attack pgd  --attack_eval std --detector InputMFS --clf RF
python attack_transfer.py --attack pgd  --attack_eval df  --detector InputMFS --clf RF
python attack_transfer.py --attack pgd  --attack_eval cw  --detector InputMFS --clf RF

python attack_transfer.py --attack std  --attack_eval df  --detector InputMFS --clf RF
python attack_transfer.py --attack std  --attack_eval cw  --detector InputMFS --clf RF

python attack_transfer.py --attack df   --attack_eval cw  --detector InputMFS --clf RF




######## cif10 - WB

python attack_transfer.py --attack fgsm --attack_eval bim --detector LayerMFS --clf LR
python attack_transfer.py --attack fgsm --attack_eval pgd --detector LayerMFS --clf LR
python attack_transfer.py --attack fgsm --attack_eval std --detector LayerMFS --clf LR
python attack_transfer.py --attack fgsm --attack_eval df  --detector LayerMFS --clf LR
python attack_transfer.py --attack fgsm --attack_eval cw  --detector LayerMFS --clf LR

python attack_transfer.py --attack bim  --attack_eval pgd --detector LayerMFS --clf LR
python attack_transfer.py --attack bim  --attack_eval std --detector LayerMFS --clf LR
python attack_transfer.py --attack bim  --attack_eval df  --detector LayerMFS --clf LR
python attack_transfer.py --attack bim  --attack_eval cw  --detector LayerMFS --clf LR

python attack_transfer.py --attack pgd  --attack_eval std --detector LayerMFS --clf LR
python attack_transfer.py --attack pgd  --attack_eval df  --detector LayerMFS --clf LR
python attack_transfer.py --attack pgd  --attack_eval cw  --detector LayerMFS --clf LR

python attack_transfer.py --attack std  --attack_eval df  --detector LayerMFS --clf LR
python attack_transfer.py --attack std  --attack_eval cw  --detector LayerMFS --clf LR

python attack_transfer.py --attack df   --attack_eval cw  --detector LayerMFS --clf LR


# Random Forest
python attack_transfer.py --attack fgsm --attack_eval bim --detector LayerMFS --clf RF
python attack_transfer.py --attack fgsm --attack_eval pgd --detector LayerMFS --clf RF
python attack_transfer.py --attack fgsm --attack_eval std --detector LayerMFS --clf RF
python attack_transfer.py --attack fgsm --attack_eval df  --detector LayerMFS --clf RF
python attack_transfer.py --attack fgsm --attack_eval cw  --detector LayerMFS --clf RF

python attack_transfer.py --attack bim  --attack_eval pgd --detector LayerMFS --clf RF
python attack_transfer.py --attack bim  --attack_eval std --detector LayerMFS --clf RF
python attack_transfer.py --attack bim  --attack_eval df  --detector LayerMFS --clf RF
python attack_transfer.py --attack bim  --attack_eval cw  --detector LayerMFS --clf RF

python attack_transfer.py --attack pgd  --attack_eval std --detector LayerMFS --clf RF
python attack_transfer.py --attack pgd  --attack_eval df  --detector LayerMFS --clf RF
python attack_transfer.py --attack pgd  --attack_eval cw  --detector LayerMFS --clf RF

python attack_transfer.py --attack std  --attack_eval df  --detector LayerMFS --clf RF
python attack_transfer.py --attack std  --attack_eval cw  --detector LayerMFS --clf RF

python attack_transfer.py --attack df   --attack_eval cw  --detector LayerMFS --clf RF



