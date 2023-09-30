#!/bin/bash

CLF="RF"

RUNS="1 2 3"
# RUNS="3"

for run in $RUNS; do

    python attack_transfer.py --attack fgsm --attack_eval bim --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack fgsm --attack_eval pgd --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack fgsm --attack_eval std --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack fgsm --attack_eval df  --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack fgsm --attack_eval cw  --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack bim  --attack_eval fgsm --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack bim  --attack_eval pgd  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack bim  --attack_eval std  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack bim  --attack_eval df   --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack bim  --attack_eval cw   --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack pgd  --attack_eval fgsm --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack pgd  --attack_eval bim  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack pgd  --attack_eval std  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack pgd  --attack_eval df   --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack pgd  --attack_eval cw   --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack std  --attack_eval fgsm  --detector LIDNOISE --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack std  --attack_eval bim  --detector LIDNOISE  --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack std  --attack_eval pgd  --detector LIDNOISE  --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack std  --attack_eval df  --detector LIDNOISE   --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack std  --attack_eval cw  --detector LIDNOISE   --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack df   --attack_eval fgsm  --detector LIDNOISE --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack df   --attack_eval bim  --detector LIDNOISE  --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack df   --attack_eval pgd  --detector LIDNOISE  --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack df   --attack_eval std  --detector LIDNOISE  --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack df   --attack_eval cw  --detector LIDNOISE   --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack cw   --attack_eval fgsm  --detector LIDNOISE --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack cw   --attack_eval bim  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run"  
    python attack_transfer.py --attack cw   --attack_eval pgd  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run"  
    python attack_transfer.py --attack cw   --attack_eval std  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run"  
    python attack_transfer.py --attack cw   --attack_eval df  --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"  



    python attack_transfer.py --attack fgsm --net cif10vgg --attack_eval bim --detector LIDNOISE --clf "$CLF"   --run_nr  "$run" 
    python attack_transfer.py --attack fgsm --net cif10vgg --attack_eval pgd --detector LIDNOISE --clf "$CLF"   --run_nr  "$run" 
    python attack_transfer.py --attack fgsm --net cif10vgg --attack_eval std --detector LIDNOISE --clf "$CLF"   --run_nr  "$run" 
    python attack_transfer.py --attack fgsm --net cif10vgg --attack_eval df  --detector LIDNOISE --clf "$CLF"   --run_nr  "$run" 
    python attack_transfer.py --attack fgsm --net cif10vgg --attack_eval cw  --detector LIDNOISE --clf "$CLF"   --run_nr  "$run" 
    python attack_transfer.py --attack bim  --net cif10vgg --attack_eval fgsm --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack bim  --net cif10vgg --attack_eval pgd  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack bim  --net cif10vgg --attack_eval std  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack bim  --net cif10vgg --attack_eval df   --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack bim  --net cif10vgg --attack_eval cw   --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack pgd  --net cif10vgg --attack_eval fgsm --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack pgd  --net cif10vgg --attack_eval bim  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack pgd  --net cif10vgg --attack_eval std  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack pgd  --net cif10vgg --attack_eval df   --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack pgd  --net cif10vgg --attack_eval cw   --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack std  --net cif10vgg --attack_eval fgsm  --detector LIDNOISE --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack std  --net cif10vgg --attack_eval bim  --detector LIDNOISE  --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack std  --net cif10vgg --attack_eval pgd  --detector LIDNOISE  --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack std  --net cif10vgg --attack_eval df  --detector LIDNOISE   --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack std  --net cif10vgg --attack_eval cw  --detector LIDNOISE   --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack df   --net cif10vgg --attack_eval fgsm  --detector LIDNOISE --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack df   --net cif10vgg --attack_eval bim  --detector LIDNOISE  --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack df   --net cif10vgg --attack_eval pgd  --detector LIDNOISE  --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack df   --net cif10vgg --attack_eval std  --detector LIDNOISE  --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack df   --net cif10vgg --attack_eval cw  --detector LIDNOISE   --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack cw   --net cif10vgg --attack_eval fgsm  --detector LIDNOISE --clf "$CLF" --run_nr  "$run"  
    python attack_transfer.py --attack cw   --net cif10vgg --attack_eval bim  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run"
    python attack_transfer.py --attack cw   --net cif10vgg --attack_eval pgd  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run"
    python attack_transfer.py --attack cw   --net cif10vgg --attack_eval std  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run"
    python attack_transfer.py --attack cw   --net cif10vgg --attack_eval df  --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"



    python attack_transfer.py --attack fgsm --net cif100 --num_classes 100 --attack_eval bim --detector LIDNOISE --clf "$CLF"    --run_nr  "$run"
    python attack_transfer.py --attack fgsm --net cif100 --num_classes 100 --attack_eval pgd --detector LIDNOISE --clf "$CLF"    --run_nr  "$run"
    python attack_transfer.py --attack fgsm --net cif100 --num_classes 100 --attack_eval std --detector LIDNOISE --clf "$CLF"    --run_nr  "$run"
    python attack_transfer.py --attack fgsm --net cif100 --num_classes 100 --attack_eval df  --detector LIDNOISE --clf "$CLF"    --run_nr  "$run"
    python attack_transfer.py --attack fgsm --net cif100 --num_classes 100 --attack_eval cw  --detector LIDNOISE --clf "$CLF"    --run_nr  "$run"
    python attack_transfer.py --attack bim  --net cif100 --num_classes 100 --attack_eval fgsm --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack bim  --net cif100 --num_classes 100 --attack_eval pgd  --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack bim  --net cif100 --num_classes 100 --attack_eval std  --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack bim  --net cif100 --num_classes 100 --attack_eval df   --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack bim  --net cif100 --num_classes 100 --attack_eval cw   --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack pgd  --net cif100 --num_classes 100 --attack_eval fgsm --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack pgd  --net cif100 --num_classes 100 --attack_eval bim  --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack pgd  --net cif100 --num_classes 100 --attack_eval std  --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack pgd  --net cif100 --num_classes 100 --attack_eval df   --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack pgd  --net cif100 --num_classes 100 --attack_eval cw   --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack std  --net cif100 --num_classes 100 --attack_eval fgsm  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run"
    python attack_transfer.py --attack std  --net cif100 --num_classes 100 --attack_eval bim  --detector LIDNOISE  --clf "$CLF"  --run_nr  "$run"
    python attack_transfer.py --attack std  --net cif100 --num_classes 100 --attack_eval pgd  --detector LIDNOISE  --clf "$CLF"  --run_nr  "$run"
    python attack_transfer.py --attack std  --net cif100 --num_classes 100 --attack_eval df  --detector LIDNOISE   --clf "$CLF"  --run_nr  "$run"
    python attack_transfer.py --attack std  --net cif100 --num_classes 100 --attack_eval cw  --detector LIDNOISE   --clf "$CLF"  --run_nr  "$run"
    python attack_transfer.py --attack df   --net cif100 --num_classes 100 --attack_eval fgsm  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run"
    python attack_transfer.py --attack df   --net cif100 --num_classes 100 --attack_eval bim  --detector LIDNOISE  --clf "$CLF"  --run_nr  "$run"
    python attack_transfer.py --attack df   --net cif100 --num_classes 100 --attack_eval pgd  --detector LIDNOISE  --clf "$CLF"  --run_nr  "$run"
    python attack_transfer.py --attack df   --net cif100 --num_classes 100 --attack_eval std  --detector LIDNOISE  --clf "$CLF"  --run_nr  "$run"
    python attack_transfer.py --attack df   --net cif100 --num_classes 100 --attack_eval cw  --detector LIDNOISE   --clf "$CLF"  --run_nr  "$run"
    python attack_transfer.py --attack cw   --net cif100 --num_classes 100 --attack_eval fgsm  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run"
    python attack_transfer.py --attack cw   --net cif100 --num_classes 100 --attack_eval bim  --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack cw   --net cif100 --num_classes 100 --attack_eval pgd  --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack cw   --net cif100 --num_classes 100 --attack_eval std  --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack cw   --net cif100 --num_classes 100 --attack_eval df  --detector LIDNOISE --clf "$CLF"    --run_nr  "$run"

    python attack_transfer.py --attack fgsm --net cif100vgg --num_classes 100 --attack_eval bim --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack fgsm --net cif100vgg --num_classes 100 --attack_eval pgd --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack fgsm --net cif100vgg --num_classes 100 --attack_eval std --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack fgsm --net cif100vgg --num_classes 100 --attack_eval df  --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack fgsm --net cif100vgg --num_classes 100 --attack_eval cw  --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"
    python attack_transfer.py --attack bim  --net cif100vgg --num_classes 100 --attack_eval fgsm --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack bim  --net cif100vgg --num_classes 100 --attack_eval pgd  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack bim  --net cif100vgg --num_classes 100 --attack_eval std  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack bim  --net cif100vgg --num_classes 100 --attack_eval df   --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack bim  --net cif100vgg --num_classes 100 --attack_eval cw   --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack pgd  --net cif100vgg --num_classes 100 --attack_eval fgsm --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack pgd  --net cif100vgg --num_classes 100 --attack_eval bim  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack pgd  --net cif100vgg --num_classes 100 --attack_eval std  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack pgd  --net cif100vgg --num_classes 100 --attack_eval df   --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack pgd  --net cif100vgg --num_classes 100 --attack_eval cw   --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack std  --net cif100vgg --num_classes 100 --attack_eval fgsm  --detector LIDNOISE --clf "$CLF" --run_nr  "$run" 
    python attack_transfer.py --attack std  --net cif100vgg --num_classes 100 --attack_eval bim  --detector LIDNOISE  --clf "$CLF" --run_nr  "$run" 
    python attack_transfer.py --attack std  --net cif100vgg --num_classes 100 --attack_eval pgd  --detector LIDNOISE  --clf "$CLF" --run_nr  "$run" 
    python attack_transfer.py --attack std  --net cif100vgg --num_classes 100 --attack_eval df  --detector LIDNOISE   --clf "$CLF" --run_nr  "$run" 
    python attack_transfer.py --attack std  --net cif100vgg --num_classes 100 --attack_eval cw  --detector LIDNOISE   --clf "$CLF" --run_nr  "$run" 
    python attack_transfer.py --attack df   --net cif100vgg --num_classes 100 --attack_eval fgsm  --detector LIDNOISE --clf "$CLF" --run_nr  "$run" 
    python attack_transfer.py --attack df   --net cif100vgg --num_classes 100 --attack_eval bim  --detector LIDNOISE  --clf "$CLF" --run_nr  "$run" 
    python attack_transfer.py --attack df   --net cif100vgg --num_classes 100 --attack_eval pgd  --detector LIDNOISE  --clf "$CLF" --run_nr  "$run" 
    python attack_transfer.py --attack df   --net cif100vgg --num_classes 100 --attack_eval std  --detector LIDNOISE  --clf "$CLF" --run_nr  "$run" 
    python attack_transfer.py --attack df   --net cif100vgg --num_classes 100 --attack_eval cw  --detector LIDNOISE   --clf "$CLF" --run_nr  "$run" 
    python attack_transfer.py --attack cw   --net cif100vgg --num_classes 100 --attack_eval fgsm  --detector LIDNOISE --clf "$CLF" --run_nr  "$run" 
    python attack_transfer.py --attack cw   --net cif100vgg --num_classes 100 --attack_eval bim  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack cw   --net cif100vgg --num_classes 100 --attack_eval pgd  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack cw   --net cif100vgg --num_classes 100 --attack_eval std  --detector LIDNOISE --clf "$CLF"  --run_nr  "$run" 
    python attack_transfer.py --attack cw   --net cif100vgg --num_classes 100 --attack_eval df  --detector LIDNOISE --clf "$CLF"   --run_nr  "$run"

done