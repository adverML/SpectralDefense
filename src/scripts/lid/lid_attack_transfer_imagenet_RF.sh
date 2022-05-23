#!/bin/bash

# CLF="RF"

# RUNS="1 2 3"

# for run in $RUNS; do
#     python attack_transfer.py --attack fgsm --net imagenet --num_classes 1000 --attack_eval bim --detector LID --clf "$CLF"     --run_nr  "$run"
#     python attack_transfer.py --attack fgsm --net imagenet --num_classes 1000 --attack_eval pgd --detector LID --clf "$CLF"     --run_nr  "$run"
#     python attack_transfer.py --attack fgsm --net imagenet --num_classes 1000 --attack_eval std --detector LID --clf "$CLF"     --run_nr  "$run"
#     python attack_transfer.py --attack fgsm --net imagenet --num_classes 1000 --attack_eval df  --detector LID --clf "$CLF"     --run_nr  "$run"
#     python attack_transfer.py --attack fgsm --net imagenet --num_classes 1000 --attack_eval cw  --detector LID --clf "$CLF"     --run_nr  "$run"
#     python attack_transfer.py --attack bim  --net imagenet --num_classes 1000 --attack_eval fgsm --detector LID --clf "$CLF"    --run_nr  "$run" 
#     python attack_transfer.py --attack bim  --net imagenet --num_classes 1000 --attack_eval pgd  --detector LID --clf "$CLF"    --run_nr  "$run" 
#     python attack_transfer.py --attack bim  --net imagenet --num_classes 1000 --attack_eval std  --detector LID --clf "$CLF"    --run_nr  "$run" 
#     python attack_transfer.py --attack bim  --net imagenet --num_classes 1000 --attack_eval df   --detector LID --clf "$CLF"    --run_nr  "$run" 
#     python attack_transfer.py --attack bim  --net imagenet --num_classes 1000 --attack_eval cw   --detector LID --clf "$CLF"    --run_nr  "$run" 
#     python attack_transfer.py --attack pgd  --net imagenet --num_classes 1000 --attack_eval fgsm --detector LID --clf "$CLF"    --run_nr  "$run" 
#     python attack_transfer.py --attack pgd  --net imagenet --num_classes 1000 --attack_eval bim  --detector LID --clf "$CLF"    --run_nr  "$run" 
#     python attack_transfer.py --attack pgd  --net imagenet --num_classes 1000 --attack_eval std  --detector LID --clf "$CLF"    --run_nr  "$run" 
#     python attack_transfer.py --attack pgd  --net imagenet --num_classes 1000 --attack_eval df   --detector LID --clf "$CLF"    --run_nr  "$run" 
#     python attack_transfer.py --attack pgd  --net imagenet --num_classes 1000 --attack_eval cw   --detector LID --clf "$CLF"    --run_nr  "$run" 
#     python attack_transfer.py --attack std  --net imagenet --num_classes 1000 --attack_eval fgsm  --detector LID --clf "$CLF"   --run_nr  "$run"  
#     python attack_transfer.py --attack std  --net imagenet --num_classes 1000 --attack_eval bim  --detector LID  --clf "$CLF"   --run_nr  "$run"  
#     python attack_transfer.py --attack std  --net imagenet --num_classes 1000 --attack_eval pgd  --detector LID  --clf "$CLF"   --run_nr  "$run"  
#     python attack_transfer.py --attack std  --net imagenet --num_classes 1000 --attack_eval df  --detector LID   --clf "$CLF"   --run_nr  "$run"  
#     python attack_transfer.py --attack std  --net imagenet --num_classes 1000 --attack_eval cw  --detector LID   --clf "$CLF"   --run_nr  "$run"  
#     python attack_transfer.py --attack df   --net imagenet --num_classes 1000 --attack_eval fgsm  --detector LID --clf "$CLF"   --run_nr  "$run"  
#     python attack_transfer.py --attack df   --net imagenet --num_classes 1000 --attack_eval bim  --detector LID  --clf "$CLF"   --run_nr  "$run"  
#     python attack_transfer.py --attack df   --net imagenet --num_classes 1000 --attack_eval pgd  --detector LID  --clf "$CLF"   --run_nr  "$run"  
#     python attack_transfer.py --attack df   --net imagenet --num_classes 1000 --attack_eval std  --detector LID  --clf "$CLF"   --run_nr  "$run"  
#     python attack_transfer.py --attack df   --net imagenet --num_classes 1000 --attack_eval cw  --detector LID   --clf "$CLF"   --run_nr  "$run"  
#     python attack_transfer.py --attack cw   --net imagenet --num_classes 1000 --attack_eval fgsm  --detector LID --clf "$CLF"   --run_nr  "$run"  
#     python attack_transfer.py --attack cw   --net imagenet --num_classes 1000 --attack_eval bim  --detector LID --clf "$CLF"    --run_nr  "$run" 
#     python attack_transfer.py --attack cw   --net imagenet --num_classes 1000 --attack_eval pgd  --detector LID --clf "$CLF"    --run_nr  "$run" 
#     python attack_transfer.py --attack cw   --net imagenet --num_classes 1000 --attack_eval std  --detector LID --clf "$CLF"    --run_nr  "$run" 
#     python attack_transfer.py --attack cw   --net imagenet --num_classes 1000 --attack_eval df  --detector LID --clf "$CLF"     --run_nr  "$run"  

# done


python attack_transfer.py --attack bim  --net imagenet --num_classes 1000 --attack_eval df  --detector LID   --clf RF   --run_nr  2
python attack_transfer.py --attack std  --net imagenet --num_classes 1000 --attack_eval df  --detector LID   --clf RF   --run_nr  2
python attack_transfer.py --attack std  --net imagenet --num_classes 1000 --attack_eval fgsm  --detector LID   --clf RF   --run_nr  2
python attack_transfer.py --attack std  --net imagenet --num_classes 1000 --attack_eval cw  --detector LID   --clf RF   --run_nr  2

python attack_transfer.py --attack pgd  --net imagenet --num_classes 1000 --attack_eval df  --detector LID   --clf RF   --run_nr  2
python attack_transfer.py --attack pgd  --net imagenet --num_classes 1000 --attack_eval cw  --detector LID   --clf RF   --run_nr  2




