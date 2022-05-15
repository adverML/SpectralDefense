#!/bin/bash

######## cif10

# python data_transfer.py --attack fgsm  --net cif10  --net_eval imagenet32 --detector LID --num_classes 10 --num_classes_eval 1000 --clf LR 
# python data_transfer.py --attack fgsm  --net cif10  --net_eval celebaHQ32 --detector LID --num_classes 10 --num_classes_eval 4    --clf LR 
# python data_transfer.py --attack bim   --net cif10  --net_eval imagenet32 --detector LID --num_classes 10 --num_classes_eval 1000 --clf LR 
# python data_transfer.py --attack bim   --net cif10  --net_eval celebaHQ32 --detector LID --num_classes 10 --num_classes_eval 4    --clf LR 
# python data_transfer.py --attack pgd   --net cif10  --net_eval imagenet32 --detector LID --num_classes 10 --num_classes_eval 1000 --clf LR 
# python data_transfer.py --attack pgd   --net cif10  --net_eval celebaHQ32 --detector LID --num_classes 10 --num_classes_eval 4    --clf LR 
# python data_transfer.py --attack std   --net cif10  --net_eval imagenet32 --detector LID --num_classes 10 --num_classes_eval 1000 --clf LR 
# python data_transfer.py --attack std   --net cif10  --net_eval celebaHQ32 --detector LID --num_classes 10 --num_classes_eval 4    --clf LR 
# python data_transfer.py --attack df    --net cif10  --net_eval imagenet32 --detector LID --num_classes 10 --num_classes_eval 1000 --clf LR 
# python data_transfer.py --attack df    --net cif10  --net_eval celebaHQ32 --detector LID --num_classes 10 --num_classes_eval 4    --clf LR 
# python data_transfer.py --attack cw    --net cif10  --net_eval imagenet32 --detector LID --num_classes 10 --num_classes_eval 1000 --clf LR 
# python data_transfer.py --attack cw    --net cif10  --net_eval celebaHQ32 --detector LID --num_classes 10 --num_classes_eval 4    --clf LR 

python data_transfer.py --attack fgsm  --net cif10  --net_eval cif100 --detector LID --num_classes 10 --num_classes_eval 100 --clf RF  
python data_transfer.py --attack bim   --net cif10  --net_eval cif100 --detector LID --num_classes 10 --num_classes_eval 100 --clf RF  
python data_transfer.py --attack pgd   --net cif10  --net_eval cif100 --detector LID --num_classes 10 --num_classes_eval 100 --clf RF 
python data_transfer.py --attack std   --net cif10  --net_eval cif100 --detector LID --num_classes 10 --num_classes_eval 100 --clf RF 
python data_transfer.py --attack df    --net cif10  --net_eval cif100 --detector LID --num_classes 10 --num_classes_eval 100 --clf RF 
python data_transfer.py --attack cw    --net cif10  --net_eval cif100 --detector LID --num_classes 10 --num_classes_eval 100 --clf RF 

python data_transfer.py --attack fgsm  --net cif10vgg  --net_eval cif100vgg --detector LID --num_classes 10 --num_classes_eval 100 --clf RF  
python data_transfer.py --attack bim   --net cif10vgg  --net_eval cif100vgg --detector LID --num_classes 10 --num_classes_eval 100 --clf RF  
python data_transfer.py --attack pgd   --net cif10vgg  --net_eval cif100vgg --detector LID --num_classes 10 --num_classes_eval 100 --clf RF 
python data_transfer.py --attack std   --net cif10vgg  --net_eval cif100vgg --detector LID --num_classes 10 --num_classes_eval 100 --clf RF 
python data_transfer.py --attack df    --net cif10vgg  --net_eval cif100vgg --detector LID --num_classes 10 --num_classes_eval 100 --clf RF 
python data_transfer.py --attack cw    --net cif10vgg  --net_eval cif100vgg --detector LID --num_classes 10 --num_classes_eval 100 --clf RF 