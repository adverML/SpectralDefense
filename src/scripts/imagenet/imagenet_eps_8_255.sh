#!/bin/bash

function log_msg {
  echo  "`date` $@"
}

# DATASETS=(cif10 cif10vgg cif100 cif100vgg imagenet imagenet32 imagenet64 imagenet128 celebaHQ32 celebaHQ64 celebaHQ128)
DATASETS="imagenet"
# DATASETS="imagenet_hierarchy"
# DATASETS="restricted_imagenet"

RUNS="4"
# RUNS="1 2 3"
# RUNS="8"
# RUNS="7"
# RUNS="10"


# ATTACKS="fgsm bim pgd"
# ATTACKS="gauss"
# ATTACKS="fgsm bim pgd std"
# ATTACKS="aa+"

# ATTACKS="apgd-dlr+ fab+ square+ apgd-t+ fab-t+"
ATTACKS="apgd-ce"
# ATTACKS="gauss fgsm bim std pgd df cw"
# ATTACKS="apgd-ce apgd-t fab-t square"
# ATTACKS="fab-t"


# DETECTORS="InputMFS LayerMFS LID Mahalanobis"
# DETECTORS="LayerMFS"
# DETECTORS="InputMFS LayerMFS"
# DETECTORS="InputPFS LayerPFS InputMFS LayerMFS LID Mahalanobis"

DETECTORS="LID"


# EPSILONS="8./255. 4./255. 2./255. 1./255. 0.5/255."
EPSILONS="8./255. 4./255. 2./255."

CLF="LR RF"
# CLF="RF"
# CLF="LR"
# CLF="IF"

DATASETSLAYERNR="imagenet"
ATTACKSLAYERNR="std df"
# ATTACKSLAYERNR="bim df"

# ATTACKSLAYERNR="fgsm bim pgd std df cw"
LAYERNR="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
# LAYERNR="16"

# LAYERNR="3 5 6 7 8 9 10 11"

DETECTORSLAYERNR="LayerMFS LayerPFS"
# DETECTORSLAYERNR="LayerMFS LayerPFS"

# NRSAMPLES="300 500 1000 1200 1500 2000"
ALLSAMPLES="10000"
WANTEDSAMPLES="1500"
WANTEDSAMPLES_TR="18000"
WANTEDSAMPLES_TE="18000"

NRSAMPLES="1500" # detect

#-----------------------------------------------------------------------------------------------------------------------------------
log_msg "Networks are already trained!"
#-----------------------------------------------------------------------------------------------------------------------------------

genereratecleandata ()
{
    log_msg "Generate Clean Data for Foolbox Attacks and Autoattack!"
    for run in $RUNS; do
        for net in $DATASETS; do
            python -u generate_clean_data.py --net "$net" --num_classes 1000   --run_nr "$run" --wanted_samples "$ALLSAMPLES" #--shuffle_off
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
                    python -u attacks.py --net "$net" --attack "$att"  --batch_size 500 --all_samples 8000 --all_samples $ALLSAMPLES --wanted_samples $WANTEDSAMPLES --eps "$eps" --run_nr "$run"
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
                        python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --run_nr "$run"  --eps "$eps" --wanted_samples $WANTEDSAMPLES --take_inputimage_off
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
                                python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier"   --eps "$eps"  --run_nr "$run" --pca_features 0
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
                        if [ "$run" == 7 ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000 --nr "$nr" --run_nr "$run" #--take_inputimage_off
                        else
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000 --nr "$nr" --run_nr "$run" --take_inputimage_off
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
                                python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 1000 --nr "$nr" --run_nr "$run" 
                                # python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 1000 --nr "$nr" --run_nr "$run" --pca_features -1
                            done
                        done
                    done
                done
            done
        done
    done
}

# genereratecleandata
attacks
extractcharacteristics
detectadversarials

# extractcharacteristicslayer
# detectadversarialslayer

# #------------------------------------------------------------------------------------------------------------
log_msg "finished"
exit 0
