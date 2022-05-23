#!/bin/bash

RUNS="1 2 3"

for run in $RUNS; do

    ######## cif10

    python attack_transfer.py --attack fgsm --attack_eval bim --detector InputMFS --clf LR  --run_nr "$run" 
    python attack_transfer.py --attack fgsm --attack_eval pgd --detector InputMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack fgsm --attack_eval std --detector InputMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack fgsm --attack_eval df  --detector InputMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack fgsm --attack_eval cw  --detector InputMFS --clf LR  --run_nr "$run"

    python attack_transfer.py --attack bim  --attack_eval pgd --detector InputMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack bim  --attack_eval std --detector InputMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack bim  --attack_eval df  --detector InputMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack bim  --attack_eval cw  --detector InputMFS --clf LR  --run_nr "$run"

    python attack_transfer.py --attack pgd  --attack_eval std --detector InputMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack pgd  --attack_eval df  --detector InputMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack pgd  --attack_eval cw  --detector InputMFS --clf LR  --run_nr "$run"

    python attack_transfer.py --attack std  --attack_eval df  --detector InputMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack std  --attack_eval cw  --detector InputMFS --clf LR  --run_nr "$run"

    python attack_transfer.py --attack df   --attack_eval cw  --detector InputMFS --clf LR  --run_nr "$run"


    # Random Forest
    python attack_transfer.py --attack fgsm --attack_eval bim --detector InputMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack fgsm --attack_eval pgd --detector InputMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack fgsm --attack_eval std --detector InputMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack fgsm --attack_eval df  --detector InputMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack fgsm --attack_eval cw  --detector InputMFS --clf RF  --run_nr "$run"

    python attack_transfer.py --attack bim  --attack_eval pgd --detector InputMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack bim  --attack_eval std --detector InputMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack bim  --attack_eval df  --detector InputMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack bim  --attack_eval cw  --detector InputMFS --clf RF  --run_nr "$run"

    python attack_transfer.py --attack pgd  --attack_eval std --detector InputMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack pgd  --attack_eval df  --detector InputMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack pgd  --attack_eval cw  --detector InputMFS --clf RF  --run_nr "$run"

    python attack_transfer.py --attack std  --attack_eval df  --detector InputMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack std  --attack_eval cw  --detector InputMFS --clf RF  --run_nr "$run"

    python attack_transfer.py --attack df   --attack_eval cw  --detector InputMFS --clf RF  --run_nr "$run"




    ######## cif10 - WB

    python attack_transfer.py --attack fgsm --attack_eval bim --detector LayerMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack fgsm --attack_eval pgd --detector LayerMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack fgsm --attack_eval std --detector LayerMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack fgsm --attack_eval df  --detector LayerMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack fgsm --attack_eval cw  --detector LayerMFS --clf LR  --run_nr "$run"

    python attack_transfer.py --attack bim  --attack_eval pgd --detector LayerMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack bim  --attack_eval std --detector LayerMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack bim  --attack_eval df  --detector LayerMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack bim  --attack_eval cw  --detector LayerMFS --clf LR  --run_nr "$run"

    python attack_transfer.py --attack pgd  --attack_eval std --detector LayerMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack pgd  --attack_eval df  --detector LayerMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack pgd  --attack_eval cw  --detector LayerMFS --clf LR  --run_nr "$run"

    python attack_transfer.py --attack std  --attack_eval df  --detector LayerMFS --clf LR  --run_nr "$run"
    python attack_transfer.py --attack std  --attack_eval cw  --detector LayerMFS --clf LR  --run_nr "$run"

    python attack_transfer.py --attack df   --attack_eval cw  --detector LayerMFS --clf LR  --run_nr "$run"


    # Random Forest
    python attack_transfer.py --attack fgsm --attack_eval bim --detector LayerMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack fgsm --attack_eval pgd --detector LayerMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack fgsm --attack_eval std --detector LayerMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack fgsm --attack_eval df  --detector LayerMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack fgsm --attack_eval cw  --detector LayerMFS --clf RF  --run_nr "$run"

    python attack_transfer.py --attack bim  --attack_eval pgd --detector LayerMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack bim  --attack_eval std --detector LayerMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack bim  --attack_eval df  --detector LayerMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack bim  --attack_eval cw  --detector LayerMFS --clf RF  --run_nr "$run"

    python attack_transfer.py --attack pgd  --attack_eval std --detector LayerMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack pgd  --attack_eval df  --detector LayerMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack pgd  --attack_eval cw  --detector LayerMFS --clf RF  --run_nr "$run"

    python attack_transfer.py --attack std  --attack_eval df  --detector LayerMFS --clf RF  --run_nr "$run"
    python attack_transfer.py --attack std  --attack_eval cw  --detector LayerMFS --clf RF  --run_nr "$run"

    python attack_transfer.py --attack df   --attack_eval cw  --detector LayerMFS --clf RF  --run_nr "$run"



done