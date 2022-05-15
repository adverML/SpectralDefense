#!/bin/bash

# bash main_imagenet3264.sh  &> log_evaluation/imagenet3264/all.log

function log_msg {
  echo  "`date` $@"
}

# DATASETS=(cif10 cif10vgg cif100 cif100vgg imagenet imagenet32 imagenet64 imagenet128 celebaHQ32 celebaHQ64 celebaHQ128)
DATASETS="imagenet64"
RUNS="1"
# RUNS="8"

# ATTACKS="cw"
# ATTACKS="df cw"
ATTACKS="gauss"

# DETECTORS="InputPFS LayerPFS LID Mahalanobis"
# DETECTORS="InputPFS LayerPFS InputMFS LayerMFS LID Mahalanobis"
DETECTORS="LayerMFS"
# EPSILONS="8./255. 4./255. 2./255. 1./255. 0.5/255."
EPSILONS="8./255."

CLF="LR RF"
# CLF="LR"

IMAGENET32CLASSES="25 50 100 250 1000"
# NRSAMPLES="300 500 1000 1200 1500 2000" # only at detectadversarialslayer
WANTEDSAMPLES="24000"
ALLSAMPLES="24000"
NRSAMPLES="24000" # detect



DATASETSLAYERNR="imagenet32"
ATTACKSLAYERNR="df"

LAYERNR="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
DETECTORSLAYERNR="LayerMFS"


#-----------------------------------------------------------------------------------------------------------------------------------
log_msg "Networks are already trained!"
#-----------------------------------------------------------------------------------------------------------------------------------

genereratecleandata ()
{
    log_msg "Generate Clean Data for Foolbox Attacks and Autoattack!"
    for run in $RUNS; do
        for net in $DATASETS; do
            if [ "$net" == imagenet32 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 1000   --img_size 32  --run_nr "$run" --wanted_samples "$ALLSAMPLES"
            fi 

            if [ "$net" == imagenet64 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 1000   --img_size 64  --run_nr "$run" --wanted_samples "$ALLSAMPLES"
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
                    if [ "$net" == imagenet32 ]; then
                        python -u attacks_trte.py --net "$net" --num_classes 1000 --attack "$att" --img_size 32 --batch_size 500  --eps "$eps"  --run_nr "$run"   --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                    fi 

                    if [ "$net" == imagenet64 ]; then
                        python -u attacks_trte.py --net "$net" --num_classes 1000 --attack "$att" --img_size 64 --batch_size 500  --run_nr "$run"   --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
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
                        if [ "$net" == imagenet32 ]; then
                            python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000  --img_size 32  --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"   --take_inputimage_off
                        fi
                        if [ "$net" == imagenet64 ]; then
                            if  [ "$att" == std ]; then
                                    python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000  --img_size 64  --eps "$eps"  --wanted_samples "$WANTEDSAMPLES"  --run_nr "$run" --take_inputimage_off
                            else
                                python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000  --img_size 64   --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off
                            fi
                        fi
                        
                    done
                done
            done
        done
    done
}

#-----------------------------------------------------------------------------------------------------------------------------------
detectadversarials ()
{
    log_msg "Detect Adversarials!"
    for run in $RUNS; do
        for net in $DATASETS; do
            for att in $ATTACKS; do
                for eps in $EPSILONS; do
                    for det in $DETECTORS; do
                        for nrsamples in $NRSAMPLES; do
                            for classifier in $CLF; do
                                if  [ "$att" == std ]; then
                                    python -u detect_adversarials.py_trte --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 1000 --eps "$eps"  --run_nr "$run"  
                                else
                                    python -u detect_adversarials.py_trte --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 1000   --run_nr "$run"  
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
}


extractcharacteristicslayer ()
{
    log_msg "Extract Characteristics Layer By Layer for WhiteBox"
    for run in $RUNS; do
        for net in $DATASETSLAYERNR; do
            for att in $ATTACKSLAYERNR; do
                for det in $DETECTORSLAYERNR; do
                    for nr in $LAYERNR; do 
                        log_msg "Layer Nr. $nr; attack $att; detectors $det"
                        if [ "$net" == imagenet32 ]; then
                            python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000 --nr "$nr" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"   --take_inputimage_off
                        fi

                        if [ "$net" == imagenet64 ]; then
                            python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000 --nr "$nr" --run_nr "$run" --img_size 64   --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off
                        fi
                    done
                done
            done
        done
    done
}



detectadversarialslayer ()
{
    log_msg "Detect Adversarials Layer By Layer!"
    for run in $RUNS; do
        for net in $DATASETSLAYERNR; do
            for att in $ATTACKSLAYERNR; do
                for det in $DETECTORSLAYERNR; do
                    for nrsamples in $NRSAMPLES; do
                        for classifier in $CLF; do
                            for nr in $LAYERNR; do 
                                log_msg "Layer Nr. $nr"
                                if [ "$net" == imagenet32 ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 1000 --nr "$nr" --run_nr  "$run" 
                                fi
                                if [ "$net" == imagenet64 ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 1000 --nr "$nr" --run_nr  "$run" 
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
}


genereratecleandata
attacks
extractcharacteristics
detectadversarials

# extractcharacteristicslayer
# detectadversarialslayer

# #-----------------------------------------------------------------------------------------------------------------------------------
log_msg "finished"
exit 0



python -u detect_adversarials.py --net imagenet32 --num_classes 1000  --wanted_samples 1500 --clf LR --detector InputMFS --attack df

