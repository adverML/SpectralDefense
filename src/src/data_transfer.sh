#!/bin/bash

######## cif10

python data_transfer.py --attack fgsm  --net cif10  --net_eval imagenet32 --detector InputMFS --num_classes 10 --num_classes_eval 1000 --clf LR 
python data_transfer.py --attack fgsm  --net cif10  --net_eval celebaHQ32 --detector InputMFS --num_classes 10 --num_classes_eval 4    --clf LR 
python data_transfer.py --attack bim   --net cif10  --net_eval imagenet32 --detector InputMFS --num_classes 10 --num_classes_eval 1000 --clf LR 
python data_transfer.py --attack bim   --net cif10  --net_eval celebaHQ32 --detector InputMFS --num_classes 10 --num_classes_eval 4    --clf LR 
python data_transfer.py --attack pgd   --net cif10  --net_eval imagenet32 --detector InputMFS --num_classes 10 --num_classes_eval 1000 --clf LR 
python data_transfer.py --attack pgd   --net cif10  --net_eval celebaHQ32 --detector InputMFS --num_classes 10 --num_classes_eval 4    --clf LR 
python data_transfer.py --attack std   --net cif10  --net_eval imagenet32 --detector InputMFS --num_classes 10 --num_classes_eval 1000 --clf LR 
python data_transfer.py --attack std   --net cif10  --net_eval celebaHQ32 --detector InputMFS --num_classes 10 --num_classes_eval 4    --clf LR 
python data_transfer.py --attack df    --net cif10  --net_eval imagenet32 --detector InputMFS --num_classes 10 --num_classes_eval 1000 --clf LR 
python data_transfer.py --attack df    --net cif10  --net_eval celebaHQ32 --detector InputMFS --num_classes 10 --num_classes_eval 4    --clf LR 
python data_transfer.py --attack cw    --net cif10  --net_eval imagenet32 --detector InputMFS --num_classes 10 --num_classes_eval 1000 --clf LR 
python data_transfer.py --attack cw    --net cif10  --net_eval celebaHQ32 --detector InputMFS --num_classes 10 --num_classes_eval 4    --clf LR 

python data_transfer.py --attack fgsm  --net cif10  --net_eval imagenet32 --detector InputMFS --num_classes 10 --num_classes_eval 1000 --clf RF 
python data_transfer.py --attack fgsm  --net cif10  --net_eval celebaHQ32 --detector InputMFS --num_classes 10 --num_classes_eval 4    --clf RF 
python data_transfer.py --attack bim   --net cif10  --net_eval imagenet32 --detector InputMFS --num_classes 10 --num_classes_eval 1000 --clf RF 
python data_transfer.py --attack bim   --net cif10  --net_eval celebaHQ32 --detector InputMFS --num_classes 10 --num_classes_eval 4    --clf RF 
python data_transfer.py --attack pgd   --net cif10  --net_eval imagenet32 --detector InputMFS --num_classes 10 --num_classes_eval 1000 --clf RF 
python data_transfer.py --attack pgd   --net cif10  --net_eval celebaHQ32 --detector InputMFS --num_classes 10 --num_classes_eval 4    --clf RF 
python data_transfer.py --attack std   --net cif10  --net_eval imagenet32 --detector InputMFS --num_classes 10 --num_classes_eval 1000 --clf RF 
python data_transfer.py --attack std   --net cif10  --net_eval celebaHQ32 --detector InputMFS --num_classes 10 --num_classes_eval 4    --clf RF 
python data_transfer.py --attack df    --net cif10  --net_eval imagenet32 --detector InputMFS --num_classes 10 --num_classes_eval 1000 --clf RF 
python data_transfer.py --attack df    --net cif10  --net_eval celebaHQ32 --detector InputMFS --num_classes 10 --num_classes_eval 4    --clf RF 
python data_transfer.py --attack cw    --net cif10  --net_eval imagenet32 --detector InputMFS --num_classes 10 --num_classes_eval 1000 --clf RF 
python data_transfer.py --attack cw    --net cif10  --net_eval celebaHQ32 --detector InputMFS --num_classes 10 --num_classes_eval 4    --clf RF 


python data_transfer.py --attack fgsm  --net cif10  --net_eval imagenet32 --detector LayerMFS --num_classes 10 --num_classes_eval 1000 --clf LR 
python data_transfer.py --attack fgsm  --net cif10  --net_eval celebaHQ32 --detector LayerMFS --num_classes 10 --num_classes_eval 4    --clf LR 
python data_transfer.py --attack bim   --net cif10  --net_eval imagenet32 --detector LayerMFS --num_classes 10 --num_classes_eval 1000 --clf LR 
python data_transfer.py --attack bim   --net cif10  --net_eval celebaHQ32 --detector LayerMFS --num_classes 10 --num_classes_eval 4    --clf LR 
python data_transfer.py --attack pgd   --net cif10  --net_eval imagenet32 --detector LayerMFS --num_classes 10 --num_classes_eval 1000 --clf LR 
python data_transfer.py --attack pgd   --net cif10  --net_eval celebaHQ32 --detector LayerMFS --num_classes 10 --num_classes_eval 4    --clf LR 
python data_transfer.py --attack std   --net cif10  --net_eval imagenet32 --detector LayerMFS --num_classes 10 --num_classes_eval 1000 --clf LR 
python data_transfer.py --attack std   --net cif10  --net_eval celebaHQ32 --detector LayerMFS --num_classes 10 --num_classes_eval 4    --clf LR 
python data_transfer.py --attack df    --net cif10  --net_eval imagenet32 --detector LayerMFS --num_classes 10 --num_classes_eval 1000 --clf LR 
python data_transfer.py --attack df    --net cif10  --net_eval celebaHQ32 --detector LayerMFS --num_classes 10 --num_classes_eval 4    --clf LR 
python data_transfer.py --attack cw    --net cif10  --net_eval imagenet32 --detector LayerMFS --num_classes 10 --num_classes_eval 1000 --clf LR 
python data_transfer.py --attack cw    --net cif10  --net_eval celebaHQ32 --detector LayerMFS --num_classes 10 --num_classes_eval 4    --clf LR 

python data_transfer.py --attack fgsm  --net cif10  --net_eval imagenet32 --detector LayerMFS --num_classes 10 --num_classes_eval 1000 --clf RF 
python data_transfer.py --attack fgsm  --net cif10  --net_eval celebaHQ32 --detector LayerMFS --num_classes 10 --num_classes_eval 4    --clf RF 
python data_transfer.py --attack bim   --net cif10  --net_eval imagenet32 --detector LayerMFS --num_classes 10 --num_classes_eval 1000 --clf RF 
python data_transfer.py --attack bim   --net cif10  --net_eval celebaHQ32 --detector LayerMFS --num_classes 10 --num_classes_eval 4    --clf RF 
python data_transfer.py --attack pgd   --net cif10  --net_eval imagenet32 --detector LayerMFS --num_classes 10 --num_classes_eval 1000 --clf RF 
python data_transfer.py --attack pgd   --net cif10  --net_eval celebaHQ32 --detector LayerMFS --num_classes 10 --num_classes_eval 4    --clf RF 
python data_transfer.py --attack std   --net cif10  --net_eval imagenet32 --detector LayerMFS --num_classes 10 --num_classes_eval 1000 --clf RF 
python data_transfer.py --attack std   --net cif10  --net_eval celebaHQ32 --detector LayerMFS --num_classes 10 --num_classes_eval 4    --clf RF 
python data_transfer.py --attack df    --net cif10  --net_eval imagenet32 --detector LayerMFS --num_classes 10 --num_classes_eval 1000 --clf RF 
python data_transfer.py --attack df    --net cif10  --net_eval celebaHQ32 --detector LayerMFS --num_classes 10 --num_classes_eval 4    --clf RF 
python data_transfer.py --attack cw    --net cif10  --net_eval imagenet32 --detector LayerMFS --num_classes 10 --num_classes_eval 1000 --clf RF 
python data_transfer.py --attack cw    --net cif10  --net_eval celebaHQ32 --detector LayerMFS --num_classes 10 --num_classes_eval 4    --clf RF 


#imagenet32
python data_transfer.py --attack fgsm  --net imagenet32  --net_eval cif10      --detector InputMFS --num_classes 1000 --num_classes_eval 10 --clf LR 
python data_transfer.py --attack fgsm  --net imagenet32  --net_eval celebaHQ32 --detector InputMFS --num_classes 1000 --num_classes_eval 4  --clf LR 
python data_transfer.py --attack bim   --net imagenet32  --net_eval cif10      --detector InputMFS --num_classes 1000 --num_classes_eval 10 --clf LR 
python data_transfer.py --attack bim   --net imagenet32  --net_eval celebaHQ32 --detector InputMFS --num_classes 1000 --num_classes_eval 4  --clf LR 
python data_transfer.py --attack pgd   --net imagenet32  --net_eval cif10      --detector InputMFS --num_classes 1000 --num_classes_eval 10 --clf LR 
python data_transfer.py --attack pgd   --net imagenet32  --net_eval celebaHQ32 --detector InputMFS --num_classes 1000 --num_classes_eval 4  --clf LR 
python data_transfer.py --attack std   --net imagenet32  --net_eval cif10      --detector InputMFS --num_classes 1000 --num_classes_eval 10 --clf LR 
python data_transfer.py --attack std   --net imagenet32  --net_eval celebaHQ32 --detector InputMFS --num_classes 1000 --num_classes_eval 4  --clf LR 
python data_transfer.py --attack df    --net imagenet32  --net_eval cif10      --detector InputMFS --num_classes 1000 --num_classes_eval 10 --clf LR 
python data_transfer.py --attack df    --net imagenet32  --net_eval celebaHQ32 --detector InputMFS --num_classes 1000 --num_classes_eval 4  --clf LR 
python data_transfer.py --attack cw    --net imagenet32  --net_eval cif10      --detector InputMFS --num_classes 1000 --num_classes_eval 10 --clf LR 
python data_transfer.py --attack cw    --net imagenet32  --net_eval celebaHQ32 --detector InputMFS --num_classes 1000 --num_classes_eval 4  --clf LR 

python data_transfer.py --attack fgsm  --net imagenet32  --net_eval cif10      --detector InputMFS --num_classes 1000 --num_classes_eval 10 --clf RF 
python data_transfer.py --attack fgsm  --net imagenet32  --net_eval celebaHQ32 --detector InputMFS --num_classes 1000 --num_classes_eval 4  --clf RF 
python data_transfer.py --attack bim   --net imagenet32  --net_eval cif10      --detector InputMFS --num_classes 1000 --num_classes_eval 10 --clf RF 
python data_transfer.py --attack bim   --net imagenet32  --net_eval celebaHQ32 --detector InputMFS --num_classes 1000 --num_classes_eval 4  --clf RF 
python data_transfer.py --attack pgd   --net imagenet32  --net_eval cif10      --detector InputMFS --num_classes 1000 --num_classes_eval 10 --clf RF 
python data_transfer.py --attack pgd   --net imagenet32  --net_eval celebaHQ32 --detector InputMFS --num_classes 1000 --num_classes_eval 4  --clf RF 
python data_transfer.py --attack std   --net imagenet32  --net_eval cif10      --detector InputMFS --num_classes 1000 --num_classes_eval 10 --clf RF 
python data_transfer.py --attack std   --net imagenet32  --net_eval celebaHQ32 --detector InputMFS --num_classes 1000 --num_classes_eval 4  --clf RF 
python data_transfer.py --attack df    --net imagenet32  --net_eval cif10      --detector InputMFS --num_classes 1000 --num_classes_eval 10 --clf RF 
python data_transfer.py --attack df    --net imagenet32  --net_eval celebaHQ32 --detector InputMFS --num_classes 1000 --num_classes_eval 4  --clf RF 
python data_transfer.py --attack cw    --net imagenet32  --net_eval cif10      --detector InputMFS --num_classes 1000 --num_classes_eval 10 --clf RF 
python data_transfer.py --attack cw    --net imagenet32  --net_eval celebaHQ32 --detector InputMFS --num_classes 1000 --num_classes_eval 4  --clf RF 


python data_transfer.py --attack fgsm  --net imagenet32  --net_eval cif10      --detector LayerMFS --num_classes 1000 --num_classes_eval 10 --clf LR 
python data_transfer.py --attack fgsm  --net imagenet32  --net_eval celebaHQ32 --detector LayerMFS --num_classes 1000 --num_classes_eval 4  --clf LR 
python data_transfer.py --attack bim   --net imagenet32  --net_eval cif10      --detector LayerMFS --num_classes 1000 --num_classes_eval 10 --clf LR 
python data_transfer.py --attack bim   --net imagenet32  --net_eval celebaHQ32 --detector LayerMFS --num_classes 1000 --num_classes_eval 4  --clf LR 
python data_transfer.py --attack pgd   --net imagenet32  --net_eval cif10      --detector LayerMFS --num_classes 1000 --num_classes_eval 10 --clf LR 
python data_transfer.py --attack pgd   --net imagenet32  --net_eval celebaHQ32 --detector LayerMFS --num_classes 1000 --num_classes_eval 4  --clf LR 
python data_transfer.py --attack std   --net imagenet32  --net_eval cif10      --detector LayerMFS --num_classes 1000 --num_classes_eval 10 --clf LR 
python data_transfer.py --attack std   --net imagenet32  --net_eval celebaHQ32 --detector LayerMFS --num_classes 1000 --num_classes_eval 4  --clf LR 
python data_transfer.py --attack df    --net imagenet32  --net_eval cif10      --detector LayerMFS --num_classes 1000 --num_classes_eval 10 --clf LR 
python data_transfer.py --attack df    --net imagenet32  --net_eval celebaHQ32 --detector LayerMFS --num_classes 1000 --num_classes_eval 4  --clf LR 
python data_transfer.py --attack cw    --net imagenet32  --net_eval cif10      --detector LayerMFS --num_classes 1000 --num_classes_eval 10 --clf LR 
python data_transfer.py --attack cw    --net imagenet32  --net_eval celebaHQ32 --detector LayerMFS --num_classes 1000 --num_classes_eval 4  --clf LR 

python data_transfer.py --attack fgsm  --net imagenet32  --net_eval cif10      --detector LayerMFS --num_classes 1000 --num_classes_eval 10 --clf RF 
python data_transfer.py --attack fgsm  --net imagenet32  --net_eval celebaHQ32 --detector LayerMFS --num_classes 1000 --num_classes_eval 4  --clf RF 
python data_transfer.py --attack bim   --net imagenet32  --net_eval cif10      --detector LayerMFS --num_classes 1000 --num_classes_eval 10 --clf RF 
python data_transfer.py --attack bim   --net imagenet32  --net_eval celebaHQ32 --detector LayerMFS --num_classes 1000 --num_classes_eval 4  --clf RF 
python data_transfer.py --attack pgd   --net imagenet32  --net_eval cif10      --detector LayerMFS --num_classes 1000 --num_classes_eval 10 --clf RF 
python data_transfer.py --attack pgd   --net imagenet32  --net_eval celebaHQ32 --detector LayerMFS --num_classes 1000 --num_classes_eval 4  --clf RF 
python data_transfer.py --attack std   --net imagenet32  --net_eval cif10      --detector LayerMFS --num_classes 1000 --num_classes_eval 10 --clf RF 
python data_transfer.py --attack std   --net imagenet32  --net_eval celebaHQ32 --detector LayerMFS --num_classes 1000 --num_classes_eval 4  --clf RF 
python data_transfer.py --attack df    --net imagenet32  --net_eval cif10      --detector LayerMFS --num_classes 1000 --num_classes_eval 10 --clf RF 
python data_transfer.py --attack df    --net imagenet32  --net_eval celebaHQ32 --detector LayerMFS --num_classes 1000 --num_classes_eval 4  --clf RF 
python data_transfer.py --attack cw    --net imagenet32  --net_eval cif10      --detector LayerMFS --num_classes 1000 --num_classes_eval 10 --clf RF 
python data_transfer.py --attack cw    --net imagenet32  --net_eval celebaHQ32 --detector LayerMFS --num_classes 1000 --num_classes_eval 4  --clf RF 


#celebaHQ32
python data_transfer.py --attack fgsm  --net celebaHQ32  --net_eval cif10      --detector InputMFS --num_classes 4 --num_classes_eval 10    --clf LR 
python data_transfer.py --attack fgsm  --net celebaHQ32  --net_eval imagenet32 --detector InputMFS --num_classes 4 --num_classes_eval 1000  --clf LR 
python data_transfer.py --attack bim   --net celebaHQ32  --net_eval cif10      --detector InputMFS --num_classes 4 --num_classes_eval 10    --clf LR 
python data_transfer.py --attack bim   --net celebaHQ32  --net_eval imagenet32 --detector InputMFS --num_classes 4 --num_classes_eval 1000  --clf LR 
python data_transfer.py --attack pgd   --net celebaHQ32  --net_eval cif10      --detector InputMFS --num_classes 4 --num_classes_eval 10    --clf LR 
python data_transfer.py --attack pgd   --net celebaHQ32  --net_eval imagenet32 --detector InputMFS --num_classes 4 --num_classes_eval 1000  --clf LR 
python data_transfer.py --attack std   --net celebaHQ32  --net_eval cif10      --detector InputMFS --num_classes 4 --num_classes_eval 10    --clf LR 
python data_transfer.py --attack std   --net celebaHQ32  --net_eval imagenet32 --detector InputMFS --num_classes 4 --num_classes_eval 1000  --clf LR 
python data_transfer.py --attack df    --net celebaHQ32  --net_eval cif10      --detector InputMFS --num_classes 4 --num_classes_eval 10    --clf LR 
python data_transfer.py --attack df    --net celebaHQ32  --net_eval imagenet32 --detector InputMFS --num_classes 4 --num_classes_eval 1000  --clf LR 
python data_transfer.py --attack cw    --net celebaHQ32  --net_eval cif10      --detector InputMFS --num_classes 4 --num_classes_eval 10    --clf LR 
python data_transfer.py --attack cw    --net celebaHQ32  --net_eval imagenet32 --detector InputMFS --num_classes 4 --num_classes_eval 1000  --clf LR 

python data_transfer.py --attack fgsm  --net celebaHQ32  --net_eval cif10      --detector InputMFS --num_classes 4 --num_classes_eval 10    --clf RF 
python data_transfer.py --attack fgsm  --net celebaHQ32  --net_eval imagenet32 --detector InputMFS --num_classes 4 --num_classes_eval 1000  --clf RF 
python data_transfer.py --attack bim   --net celebaHQ32  --net_eval cif10      --detector InputMFS --num_classes 4 --num_classes_eval 10    --clf RF 
python data_transfer.py --attack bim   --net celebaHQ32  --net_eval imagenet32 --detector InputMFS --num_classes 4 --num_classes_eval 1000  --clf RF 
python data_transfer.py --attack pgd   --net celebaHQ32  --net_eval cif10      --detector InputMFS --num_classes 4 --num_classes_eval 10    --clf RF 
python data_transfer.py --attack pgd   --net celebaHQ32  --net_eval imagenet32 --detector InputMFS --num_classes 4 --num_classes_eval 1000  --clf RF 
python data_transfer.py --attack std   --net celebaHQ32  --net_eval cif10      --detector InputMFS --num_classes 4 --num_classes_eval 10    --clf RF 
python data_transfer.py --attack std   --net celebaHQ32  --net_eval imagenet32 --detector InputMFS --num_classes 4 --num_classes_eval 1000  --clf RF 
python data_transfer.py --attack df    --net celebaHQ32  --net_eval cif10      --detector InputMFS --num_classes 4 --num_classes_eval 10    --clf RF 
python data_transfer.py --attack df    --net celebaHQ32  --net_eval imagenet32 --detector InputMFS --num_classes 4 --num_classes_eval 1000  --clf RF 
python data_transfer.py --attack cw    --net celebaHQ32  --net_eval cif10      --detector InputMFS --num_classes 4 --num_classes_eval 10    --clf RF 
python data_transfer.py --attack cw    --net celebaHQ32  --net_eval imagenet32 --detector InputMFS --num_classes 4 --num_classes_eval 1000  --clf RF 


python data_transfer.py --attack fgsm  --net celebaHQ32  --net_eval cif10      --detector LayerMFS --num_classes 4 --num_classes_eval 10    --clf LR 
python data_transfer.py --attack fgsm  --net celebaHQ32  --net_eval imagenet32 --detector LayerMFS --num_classes 4 --num_classes_eval 1000  --clf LR 
python data_transfer.py --attack bim   --net celebaHQ32  --net_eval cif10      --detector LayerMFS --num_classes 4 --num_classes_eval 10    --clf LR 
python data_transfer.py --attack bim   --net celebaHQ32  --net_eval imagenet32 --detector LayerMFS --num_classes 4 --num_classes_eval 1000  --clf LR 
python data_transfer.py --attack pgd   --net celebaHQ32  --net_eval cif10      --detector LayerMFS --num_classes 4 --num_classes_eval 10    --clf LR 
python data_transfer.py --attack pgd   --net celebaHQ32  --net_eval imagenet32 --detector LayerMFS --num_classes 4 --num_classes_eval 1000  --clf LR 
python data_transfer.py --attack std   --net celebaHQ32  --net_eval cif10      --detector LayerMFS --num_classes 4 --num_classes_eval 10    --clf LR 
python data_transfer.py --attack std   --net celebaHQ32  --net_eval imagenet32 --detector LayerMFS --num_classes 4 --num_classes_eval 1000  --clf LR 
python data_transfer.py --attack df    --net celebaHQ32  --net_eval cif10      --detector LayerMFS --num_classes 4 --num_classes_eval 10    --clf LR 
python data_transfer.py --attack df    --net celebaHQ32  --net_eval imagenet32 --detector LayerMFS --num_classes 4 --num_classes_eval 1000  --clf LR 
python data_transfer.py --attack cw    --net celebaHQ32  --net_eval cif10      --detector LayerMFS --num_classes 4 --num_classes_eval 10    --clf LR 
python data_transfer.py --attack cw    --net celebaHQ32  --net_eval imagenet32 --detector LayerMFS --num_classes 4 --num_classes_eval 1000  --clf LR 

python data_transfer.py --attack fgsm  --net celebaHQ32  --net_eval cif10      --detector LayerMFS --num_classes 4 --num_classes_eval 10    --clf RF 
python data_transfer.py --attack fgsm  --net celebaHQ32  --net_eval imagenet32 --detector LayerMFS --num_classes 4 --num_classes_eval 1000  --clf RF 
python data_transfer.py --attack bim   --net celebaHQ32  --net_eval cif10      --detector LayerMFS --num_classes 4 --num_classes_eval 10    --clf RF 
python data_transfer.py --attack bim   --net celebaHQ32  --net_eval imagenet32 --detector LayerMFS --num_classes 4 --num_classes_eval 1000  --clf RF 
python data_transfer.py --attack pgd   --net celebaHQ32  --net_eval cif10      --detector LayerMFS --num_classes 4 --num_classes_eval 10    --clf RF 
python data_transfer.py --attack pgd   --net celebaHQ32  --net_eval imagenet32 --detector LayerMFS --num_classes 4 --num_classes_eval 1000  --clf RF 
python data_transfer.py --attack std   --net celebaHQ32  --net_eval cif10      --detector LayerMFS --num_classes 4 --num_classes_eval 10    --clf RF 
python data_transfer.py --attack std   --net celebaHQ32  --net_eval imagenet32 --detector LayerMFS --num_classes 4 --num_classes_eval 1000  --clf RF 
python data_transfer.py --attack df    --net celebaHQ32  --net_eval cif10      --detector LayerMFS --num_classes 4 --num_classes_eval 10    --clf RF 
python data_transfer.py --attack df    --net celebaHQ32  --net_eval imagenet32 --detector LayerMFS --num_classes 4 --num_classes_eval 1000  --clf RF 
python data_transfer.py --attack cw    --net celebaHQ32  --net_eval cif10      --detector LayerMFS --num_classes 4 --num_classes_eval 10    --clf RF 
python data_transfer.py --attack cw    --net celebaHQ32  --net_eval imagenet32 --detector LayerMFS --num_classes 4 --num_classes_eval 1000  --clf RF 

# imagenet64
python data_transfer.py --attack fgsm  --net imagenet64  --net_eval celebaHQ64 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack bim   --net imagenet64  --net_eval celebaHQ64 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack pgd   --net imagenet64  --net_eval celebaHQ64 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack std   --net imagenet64  --net_eval celebaHQ64 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack df    --net imagenet64  --net_eval celebaHQ64 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack cw    --net imagenet64  --net_eval celebaHQ64 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf LR 

python data_transfer.py --attack fgsm  --net imagenet64  --net_eval celebaHQ64 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack bim   --net imagenet64  --net_eval celebaHQ64 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack pgd   --net imagenet64  --net_eval celebaHQ64 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack std   --net imagenet64  --net_eval celebaHQ64 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack df    --net imagenet64  --net_eval celebaHQ64 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack cw    --net imagenet64  --net_eval celebaHQ64 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf RF 


python data_transfer.py --attack fgsm  --net imagenet64  --net_eval celebaHQ64 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack bim   --net imagenet64  --net_eval celebaHQ64 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack pgd   --net imagenet64  --net_eval celebaHQ64 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack std   --net imagenet64  --net_eval celebaHQ64 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack df    --net imagenet64  --net_eval celebaHQ64 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack cw    --net imagenet64  --net_eval celebaHQ64 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf LR 

python data_transfer.py --attack fgsm  --net imagenet64  --net_eval celebaHQ64 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack bim   --net imagenet64  --net_eval celebaHQ64 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack pgd   --net imagenet64  --net_eval celebaHQ64 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack std   --net imagenet64  --net_eval celebaHQ64 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack df    --net imagenet64  --net_eval celebaHQ64 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack cw    --net imagenet64  --net_eval celebaHQ64 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf RF


python data_transfer.py --attack fgsm  --net imagenet128  --net_eval celebaHQ128 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack bim   --net imagenet128  --net_eval celebaHQ128 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack pgd   --net imagenet128  --net_eval celebaHQ128 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack std   --net imagenet128  --net_eval celebaHQ128 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack df    --net imagenet128  --net_eval celebaHQ128 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack cw    --net imagenet128  --net_eval celebaHQ128 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf LR 

python data_transfer.py --attack fgsm  --net imagenet128  --net_eval celebaHQ128 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack bim   --net imagenet128  --net_eval celebaHQ128 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack pgd   --net imagenet128  --net_eval celebaHQ128 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack std   --net imagenet128  --net_eval celebaHQ128 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack df    --net imagenet128  --net_eval celebaHQ128 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack cw    --net imagenet128  --net_eval celebaHQ128 --detector InputMFS --num_classes 1000 --num_classes_eval 4 --clf RF 



python data_transfer.py --attack fgsm  --net imagenet128  --net_eval celebaHQ128 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack bim   --net imagenet128  --net_eval celebaHQ128 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack pgd   --net imagenet128  --net_eval celebaHQ128 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack std   --net imagenet128  --net_eval celebaHQ128 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack df    --net imagenet128  --net_eval celebaHQ128 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf LR 
python data_transfer.py --attack cw    --net imagenet128  --net_eval celebaHQ128 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf LR 

python data_transfer.py --attack fgsm  --net imagenet128  --net_eval celebaHQ128 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack bim   --net imagenet128  --net_eval celebaHQ128 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack pgd   --net imagenet128  --net_eval celebaHQ128 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack std   --net imagenet128  --net_eval celebaHQ128 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack df    --net imagenet128  --net_eval celebaHQ128 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf RF 
python data_transfer.py --attack cw    --net imagenet128  --net_eval celebaHQ128 --detector LayerMFS --num_classes 1000 --num_classes_eval 4 --clf RF 


python data_transfer.py --attack fgsm  --net celebaHQ64    --net_eval  imagenet64 --detector InputMFS --num_classes 4 --num_classes_eval  1000 --clf LR 
python data_transfer.py --attack bim   --net celebaHQ64    --net_eval  imagenet64 --detector InputMFS --num_classes 4 --num_classes_eval  1000 --clf LR 
python data_transfer.py --attack pgd   --net celebaHQ64    --net_eval  imagenet64 --detector InputMFS --num_classes 4 --num_classes_eval  1000 --clf LR 
python data_transfer.py --attack std   --net celebaHQ64    --net_eval  imagenet64 --detector InputMFS --num_classes 4 --num_classes_eval  1000 --clf LR 
python data_transfer.py --attack df    --net celebaHQ64    --net_eval  imagenet64 --detector InputMFS --num_classes 4 --num_classes_eval  1000 --clf LR 
python data_transfer.py --attack cw    --net celebaHQ64    --net_eval  imagenet64 --detector InputMFS --num_classes 4 --num_classes_eval  1000 --clf LR 
python data_transfer.py --attack fgsm  --net celebaHQ64    --net_eval  imagenet64 --detector InputMFS --num_classes 4 --num_classes_eval  1000 --clf RF 
python data_transfer.py --attack bim   --net celebaHQ64    --net_eval  imagenet64 --detector InputMFS --num_classes 4 --num_classes_eval  1000 --clf RF 
python data_transfer.py --attack pgd   --net celebaHQ64    --net_eval  imagenet64 --detector InputMFS --num_classes 4 --num_classes_eval  1000 --clf RF 
python data_transfer.py --attack std   --net celebaHQ64    --net_eval  imagenet64 --detector InputMFS --num_classes 4 --num_classes_eval  1000 --clf RF 
python data_transfer.py --attack df    --net celebaHQ64    --net_eval  imagenet64 --detector InputMFS --num_classes 4 --num_classes_eval  1000 --clf RF 
python data_transfer.py --attack cw    --net celebaHQ64    --net_eval  imagenet64 --detector InputMFS --num_classes 4 --num_classes_eval  1000 --clf RF 
python data_transfer.py --attack fgsm  --net celebaHQ64    --net_eval  imagenet64 --detector LayerMFS --num_classes 4 --num_classes_eval  1000 --clf LR 
python data_transfer.py --attack bim   --net celebaHQ64    --net_eval  imagenet64 --detector LayerMFS --num_classes 4 --num_classes_eval  1000 --clf LR 
python data_transfer.py --attack pgd   --net celebaHQ64    --net_eval  imagenet64 --detector LayerMFS --num_classes 4 --num_classes_eval  1000 --clf LR 
python data_transfer.py --attack std   --net celebaHQ64    --net_eval  imagenet64 --detector LayerMFS --num_classes 4 --num_classes_eval  1000 --clf LR 
python data_transfer.py --attack df    --net celebaHQ64    --net_eval  imagenet64 --detector LayerMFS --num_classes 4 --num_classes_eval  1000 --clf LR 
python data_transfer.py --attack cw    --net celebaHQ64    --net_eval  imagenet64 --detector LayerMFS --num_classes 4 --num_classes_eval  1000 --clf LR 
python data_transfer.py --attack fgsm  --net celebaHQ64    --net_eval  imagenet64 --detector LayerMFS --num_classes 4 --num_classes_eval  1000 --clf RF 
python data_transfer.py --attack bim   --net celebaHQ64    --net_eval  imagenet64 --detector LayerMFS --num_classes 4 --num_classes_eval  1000 --clf RF 
python data_transfer.py --attack pgd   --net celebaHQ64    --net_eval  imagenet64 --detector LayerMFS --num_classes 4 --num_classes_eval  1000 --clf RF 
python data_transfer.py --attack std   --net celebaHQ64    --net_eval  imagenet64 --detector LayerMFS --num_classes 4 --num_classes_eval  1000 --clf RF 
python data_transfer.py --attack df    --net celebaHQ64    --net_eval  imagenet64 --detector LayerMFS --num_classes 4 --num_classes_eval  1000 --clf RF 
python data_transfer.py --attack cw    --net celebaHQ64    --net_eval  imagenet64 --detector LayerMFS --num_classes 4 --num_classes_eval  1000 --clf RF


python data_transfer.py --attack fgsm  --net celebaHQ128  --net_eval imagenet128  --detector InputMFS  --num_classes 4  --num_classes_eval  1000  --clf LR 
python data_transfer.py --attack bim   --net celebaHQ128  --net_eval imagenet128  --detector InputMFS  --num_classes 4  --num_classes_eval  1000  --clf LR 
python data_transfer.py --attack pgd   --net celebaHQ128  --net_eval imagenet128  --detector InputMFS  --num_classes 4  --num_classes_eval  1000  --clf LR 
python data_transfer.py --attack std   --net celebaHQ128  --net_eval imagenet128  --detector InputMFS  --num_classes 4  --num_classes_eval  1000  --clf LR 
python data_transfer.py --attack df    --net celebaHQ128  --net_eval imagenet128  --detector InputMFS  --num_classes 4  --num_classes_eval  1000  --clf LR 
python data_transfer.py --attack cw    --net celebaHQ128  --net_eval imagenet128  --detector InputMFS  --num_classes 4  --num_classes_eval  1000  --clf LR 
python data_transfer.py --attack fgsm  --net celebaHQ128  --net_eval imagenet128  --detector InputMFS  --num_classes 4  --num_classes_eval  1000  --clf RF 
python data_transfer.py --attack bim   --net celebaHQ128  --net_eval imagenet128  --detector InputMFS  --num_classes 4  --num_classes_eval  1000  --clf RF 
python data_transfer.py --attack pgd   --net celebaHQ128  --net_eval imagenet128  --detector InputMFS  --num_classes 4  --num_classes_eval  1000  --clf RF 
python data_transfer.py --attack std   --net celebaHQ128  --net_eval imagenet128  --detector InputMFS  --num_classes 4  --num_classes_eval  1000  --clf RF 
python data_transfer.py --attack df    --net celebaHQ128  --net_eval imagenet128  --detector InputMFS  --num_classes 4  --num_classes_eval  1000  --clf RF 
python data_transfer.py --attack cw    --net celebaHQ128  --net_eval imagenet128  --detector InputMFS  --num_classes 4  --num_classes_eval  1000  --clf RF 
python data_transfer.py --attack fgsm  --net celebaHQ128  --net_eval imagenet128  --detector LayerMFS  --num_classes 4  --num_classes_eval  1000  --clf LR 
python data_transfer.py --attack bim   --net celebaHQ128  --net_eval imagenet128  --detector LayerMFS  --num_classes 4  --num_classes_eval  1000  --clf LR 
python data_transfer.py --attack pgd   --net celebaHQ128  --net_eval imagenet128  --detector LayerMFS  --num_classes 4  --num_classes_eval  1000  --clf LR 
python data_transfer.py --attack std   --net celebaHQ128  --net_eval imagenet128  --detector LayerMFS  --num_classes 4  --num_classes_eval  1000  --clf LR 
python data_transfer.py --attack df    --net celebaHQ128  --net_eval imagenet128  --detector LayerMFS  --num_classes 4  --num_classes_eval  1000  --clf LR 
python data_transfer.py --attack cw    --net celebaHQ128  --net_eval imagenet128  --detector LayerMFS  --num_classes 4  --num_classes_eval  1000  --clf LR 
python data_transfer.py --attack fgsm  --net celebaHQ128  --net_eval imagenet128  --detector LayerMFS  --num_classes 4  --num_classes_eval  1000  --clf RF 
python data_transfer.py --attack bim   --net celebaHQ128  --net_eval imagenet128  --detector LayerMFS  --num_classes 4  --num_classes_eval  1000  --clf RF 
python data_transfer.py --attack pgd   --net celebaHQ128  --net_eval imagenet128  --detector LayerMFS  --num_classes 4  --num_classes_eval  1000  --clf RF 
python data_transfer.py --attack std   --net celebaHQ128  --net_eval imagenet128  --detector LayerMFS  --num_classes 4  --num_classes_eval  1000  --clf RF 
python data_transfer.py --attack df    --net celebaHQ128  --net_eval imagenet128  --detector LayerMFS  --num_classes 4  --num_classes_eval  1000  --clf RF 
python data_transfer.py --attack cw    --net celebaHQ128  --net_eval imagenet128  --detector LayerMFS  --num_classes 4  --num_classes_eval  1000  --clf RF 