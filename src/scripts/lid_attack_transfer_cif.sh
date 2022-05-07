#!/bin/bash




## Random Forest
# python attack_transfer.py --attack fgsm --attack_eval bim --detector LID --clf RF
# python attack_transfer.py --attack fgsm --attack_eval pgd --detector LID --clf RF
# python attack_transfer.py --attack fgsm --attack_eval std --detector LID --clf RF
# python attack_transfer.py --attack fgsm --attack_eval df  --detector LID --clf RF
# python attack_transfer.py --attack fgsm --attack_eval cw  --detector LID --clf RF

# python attack_transfer.py --attack bim  --attack_eval pgd --detector LID --clf RF
# python attack_transfer.py --attack bim  --attack_eval std --detector LID --clf RF
# python attack_transfer.py --attack bim  --attack_eval df  --detector LID --clf RF
# python attack_transfer.py --attack bim  --attack_eval cw  --detector LID --clf RF

# python attack_transfer.py --attack pgd  --attack_eval std --detector LID --clf RF
# python attack_transfer.py --attack pgd  --attack_eval df  --detector LID --clf RF
# python attack_transfer.py --attack pgd  --attack_eval cw  --detector LID --clf RF

# python attack_transfer.py --attack std  --attack_eval df  --detector LID --clf RF
# python attack_transfer.py --attack std  --attack_eval cw  --detector LID --clf RF

# python attack_transfer.py --attack df   --attack_eval cw  --detector LID --clf RF



python attack_transfer.py --net   cif100 --num_classes 100 --attack fgsm --attack_eval bim --detector LID --clf RF
python attack_transfer.py --net   cif100 --num_classes 100 --attack fgsm --attack_eval pgd --detector LID --clf RF
python attack_transfer.py --net   cif100 --num_classes 100 --attack fgsm --attack_eval std --detector LID --clf RF
python attack_transfer.py --net   cif100 --num_classes 100 --attack fgsm --attack_eval df  --detector LID --clf RF
python attack_transfer.py --net   cif100 --num_classes 100 --attack fgsm --attack_eval cw  --detector LID --clf RF

python attack_transfer.py --net   cif100 --num_classes 100 --attack bim  --attack_eval pgd --detector LID --clf RF
python attack_transfer.py --net   cif100 --num_classes 100 --attack bim  --attack_eval std --detector LID --clf RF
python attack_transfer.py --net   cif100 --num_classes 100 --attack bim  --attack_eval df  --detector LID --clf RF
python attack_transfer.py --net   cif100 --num_classes 100 --attack bim  --attack_eval cw  --detector LID --clf RF

python attack_transfer.py --net   cif100 --num_classes 100 --attack pgd  --attack_eval std --detector LID --clf RF
python attack_transfer.py --net   cif100 --num_classes 100 --attack pgd  --attack_eval df  --detector LID --clf RF
python attack_transfer.py --net   cif100 --num_classes 100 --attack pgd  --attack_eval cw  --detector LID --clf RF

python attack_transfer.py --net   cif100 --num_classes 100 --attack std  --attack_eval df  --detector LID --clf RF
python attack_transfer.py --net   cif100 --num_classes 100 --attack std  --attack_eval cw  --detector LID --clf RF

python attack_transfer.py --net   cif100 --num_classes 100 --attack df   --attack_eval cw  --detector LID --clf RF



python attack_transfer.py --net   cif10vgg --attack fgsm --attack_eval bim --detector LID --clf RF
python attack_transfer.py --net   cif10vgg --attack fgsm --attack_eval pgd --detector LID --clf RF
python attack_transfer.py --net   cif10vgg --attack fgsm --attack_eval std --detector LID --clf RF
python attack_transfer.py --net   cif10vgg --attack fgsm --attack_eval df  --detector LID --clf RF
python attack_transfer.py --net   cif10vgg --attack fgsm --attack_eval cw  --detector LID --clf RF

python attack_transfer.py --net   cif10vgg --attack bim  --attack_eval pgd --detector LID --clf RF
python attack_transfer.py --net   cif10vgg --attack bim  --attack_eval std --detector LID --clf RF
python attack_transfer.py --net   cif10vgg --attack bim  --attack_eval df  --detector LID --clf RF
python attack_transfer.py --net   cif10vgg --attack bim  --attack_eval cw  --detector LID --clf RF

python attack_transfer.py --net   cif10vgg --attack pgd  --attack_eval std --detector LID --clf RF
python attack_transfer.py --net   cif10vgg --attack pgd  --attack_eval df  --detector LID --clf RF
python attack_transfer.py --net   cif10vgg --attack pgd  --attack_eval cw  --detector LID --clf RF

python attack_transfer.py --net   cif10vgg --attack std  --attack_eval df  --detector LID --clf RF
python attack_transfer.py --net   cif10vgg --attack std  --attack_eval cw  --detector LID --clf RF

python attack_transfer.py --net   cif10vgg --attack df   --attack_eval cw  --detector LID --clf RF




python attack_transfer.py --net   cif100vgg --num_classes 100 --attack fgsm --attack_eval bim --detector LID --clf RF
python attack_transfer.py --net   cif100vgg --num_classes 100 --attack fgsm --attack_eval pgd --detector LID --clf RF
python attack_transfer.py --net   cif100vgg --num_classes 100 --attack fgsm --attack_eval std --detector LID --clf RF
python attack_transfer.py --net   cif100vgg --num_classes 100 --attack fgsm --attack_eval df  --detector LID --clf RF
python attack_transfer.py --net   cif100vgg --num_classes 100 --attack fgsm --attack_eval cw  --detector LID --clf RF

python attack_transfer.py --net   cif100vgg --num_classes 100 --attack bim  --attack_eval pgd --detector LID --clf RF
python attack_transfer.py --net   cif100vgg --num_classes 100 --attack bim  --attack_eval std --detector LID --clf RF
python attack_transfer.py --net   cif100vgg --num_classes 100 --attack bim  --attack_eval df  --detector LID --clf RF
python attack_transfer.py --net   cif100vgg --num_classes 100 --attack bim  --attack_eval cw  --detector LID --clf RF

python attack_transfer.py --net   cif100vgg --num_classes 100 --attack pgd  --attack_eval std --detector LID --clf RF
python attack_transfer.py --net   cif100vgg --num_classes 100 --attack pgd  --attack_eval df  --detector LID --clf RF
python attack_transfer.py --net   cif100vgg --num_classes 100 --attack pgd  --attack_eval cw  --detector LID --clf RF

python attack_transfer.py --net   cif100vgg --num_classes 100 --attack std  --attack_eval df  --detector LID --clf RF
python attack_transfer.py --net   cif100vgg --num_classes 100 --attack std  --attack_eval cw  --detector LID --clf RF

python attack_transfer.py --net   cif100vgg --num_classes 100 --attack df   --attack_eval cw  --detector LID --clf RF