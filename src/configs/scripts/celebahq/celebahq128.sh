#!/bin/bash

#  bash main_celebahq.sh &> log_evaluation/celebAHQ/all.log

function log_msg {
  echo  "`date` $@"
}

# DATASETS=(cif10 cif10vgg cif100 cif100vgg imagenet imagenet32 imagenet64 imagenet128 celebaHQ32 celebaHQ64 celebaHQ128)
# DATASETS="celebaHQ32"
DATASETS="celebaHQ128"
# DATASETS="celebaHQ32"
# DATASETS="celebaHQ256"

RUNS="1 2 3"
# RUNS="1"

NUMCLASSES=4
VERSION="standard_4"

# ATTACKS="fgsm bim pgd df cw"
ATTACKS="gauss fgsm bim pgd std"
# ATTACKS="gauss"
# ATTACKS="cw"

DETECTORS="LayerMFS"
# DETECTORS="InputMFS LayerMFS"

# EPSILONS="4./255. 2./255. 1./255. 0.5/255."
# EPSILONS="8./255. 4./255. 2./255. 1./255. 0.5/255."
EPSILONS="8./255."
CLF="LR RF"

IMAGENET32CLASSES="25 50 100 250 1000"
# NRSAMPLES="300 500 1000 1200 1500 2000"
NRSAMPLES="1500"
# NRRUN=1..4
NRRUN=3

WANTEDSAMPLES="2000"
ALLSAMPLES="4500"
NRSAMPLES="2000" 

#-----------------------------------------------------------------------------------------------------------------------------------
log_msg "Networks are already trained!"
#-----------------------------------------------------------------------------------------------------------------------------------

genereratecleandata ()
{
    log_msg "Generate Clean Data for Foolbox Attacks and Autoattack!"
    for run in $RUNS; do
        for net in $DATASETS; do
            if [ "$net" == celebaHQ32 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes $NUMCLASSES  --img_size 32 --run_nr "$run" --wanted_samples "$ALLSAMPLES"
            fi 

            if [ "$net" == celebaHQ64 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes $NUMCLASSES  --img_size 64 --run_nr "$run" --wanted_samples "$ALLSAMPLES"
            fi 

            if [ "$net" == celebaHQ128 ]; then 
                python -u generate_clean_data.py --net "$net" --num_classes $NUMCLASSES  --img_size 128 --run_nr "$run" --wanted_samples "$ALLSAMPLES"
            fi 
            if [ "$net" == celebaHQ256 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes $NUMCLASSES  --img_size 256 --run_nr "$run" --wanted_samples "$ALLSAMPLES"
            fi 
        done
    done
}

#-----------------------------------------------------------------------------------------------------------------------------------
attacks ()
{
    log_msg "Attack Clean Data with Foolbox Attacks and Autoattack!"
    for run in $RUNS; do
        for net in $DATASETS; do
            for att in $ATTACKS; do
                for eps in $EPSILONS; do
                    if [ "$net" == celebaHQ32 ]; then
                        python -u attacks.py --net "$net" --num_classes $NUMCLASSES --attack "$att" --img_size 32 --batch_size 500  --eps "$eps" --version "$VERSION" --run_nr "$run" --wanted_samples "$WANTEDSAMPLES"
                    fi 

                    if [ "$net" == celebaHQ64 ]; then
                        python -u attacks.py --net "$net" --num_classes $NUMCLASSES --attack "$att" --img_size 64 --batch_size 500  --eps "$eps" --version "$VERSION" --run_nr "$run" --wanted_samples "$WANTEDSAMPLES"
                    fi 

                    if [ "$net" == celebaHQ128 ]; then
                        python -u attacks.py --net "$net" --num_classes $NUMCLASSES --attack "$att" --img_size 128 --batch_size 48  --eps "$eps" --version "$VERSION" --run_nr "$run" --wanted_samples "$WANTEDSAMPLES"
                    fi 

                    if [ "$net" == celebaHQ256 ]; then 
                        python -u attacks.py --net "$net" --num_classes $NUMCLASSES --attack "$att" --img_size 256 --batch_size 12  --eps "$eps" --version "$VERSION" --run_nr "$run" --wanted_samples "$WANTEDSAMPLES"
                    fi 
                done
            done
        done
    done
}

#-----------------------------------------------------------------------------------------------------------------------------------
extractcharacteristics ()
{
    log_msg "Extract Characteristics"
    for run in $RUNS; do
        for net in $DATASETS; do
            for att in $ATTACKS; do  
                for eps in $EPSILONS; do
                    for det in $DETECTORS; do
                        if [ "$net" == celebaHQ32 ]; then
                            python -u extract_characteristics.py --net "$net"  --num_classes $NUMCLASSES --attack "$att" --detector "$det" --eps "$eps" --run_nr "$run" --take_inputimage_off
                        fi 
                        if [ "$net" == celebaHQ64 ]; then
                            python -u extract_characteristics.py --net "$net"  --num_classes $NUMCLASSES --attack "$att" --detector "$det" --img_size 64 --eps "$eps" --run_nr "$run" --take_inputimage_off
                        fi 
                        if [ "$net" == celebaHQ128 ]; then
                            python -u extract_characteristics.py --net "$net"  --num_classes $NUMCLASSES --attack "$att" --detector "$det" --img_size 128 --eps "$eps" --wanted_samples 1500 --run_nr "$run" --take_inputimage_off
                        fi 

                        if [ "$net" == celebaHQ256 ]; then
                            python -u extract_characteristics.py --net "$net"  --num_classes $NUMCLASSES --attack "$att" --detector "$det" --img_size 256 --eps "$eps" --wanted_samples 1500 --run_nr "$run" --take_inputimage_off
                        fi 
                    done
                done
            done
        done
    done
}

# #-----------------------------------------------------------------------------------------------------------------------------------
detectadversarials ()
{
    log_msg "Detect Adversarials!"
    for run in $RUNS; do
        for net in $DATASETS; do
            for att in $ATTACKS; do
                for det in $DETECTORS; do
                    for nrsamples in $NRSAMPLES; do
                        for classifier in $CLF; do
                            for eps in $EPSILONS; do
                                python -u detect_adversarials.py --net "$net" --num_classes $NUMCLASSES --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier"  --eps "$eps" --run_nr "$run"
                            done
                        done
                    done
                done
            done 
        done
    done
}

# genereratecleandata
# attacks
extractcharacteristics
detectadversarials


# #-----------------------------------------------------------------------------------------------------------------------------------
log_msg "finished"
exit 0

# : <<'END'
#   just a comment!
# END


