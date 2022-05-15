#!/bin/bash

function log_msg {
  echo  "`date` $@"
}

# DATASETS=(cif10 cif10vgg cif100 cif100vgg imagenet imagenet32 imagenet64 imagenet128 celebaHQ32 celebaHQ64 celebaHQ128)
DATASETS="imagenet"

RUNS="8"
# RUNS="8"
# RUNS="7"


ATTACKS="fgsm"
# ATTACKS="fgsm bim std pgd df cw"
# ATTACKS="apgd-ce apgd-t fab-t square"
# DETECTORS="InputMFS LayerMFS LID Mahalanobis"
DETECTORS="LayerMFS"
# DETECTORS="InputPFS LayerPFS InputMFS LayerMFS LID Mahalanobis"

# EPSILONS="8./255. 4./255. 2./255. 1./255. 0.5/255."
EPSILONS="8./255."

NRSAMPLES="4000 5000 6000 7000"
# NRSAMPLES="2000"
# CLF="LR RF"
CLF="LR RF"

DATASETSLAYERNR="imagenet"
ATTACKSLAYERNR="df"
# ATTACKSLAYERNR="bim df"

# ATTACKSLAYERNR="fgsm bim pgd std df cw"
LAYERNR="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
# LAYERNR="3 5 6 7 8 9 10 11"

DETECTORSLAYERNR="LayerPFS"
# DETECTORSLAYERNR="LayerMFS LayerPFS"

NRWANTEDSAMPLES="7000"
NRSAMPLES="4000 5000 6000 7000"
WANTEDSAMPLES_TR="1000"
WANTEDSAMPLES_TE="1000"


#-----------------------------------------------------------------------------------------------------------------------------------
log_msg "Networks are already trained!"
#-----------------------------------------------------------------------------------------------------------------------------------

genereratecleandata ()
{
    log_msg "Generate Clean Data for Foolbox Attacks and Autoattack!"
    for run in $RUNS; do
        for net in $DATASETS; do
            if [ "$net" == imagenet ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 1000 --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE" --run_nr "$run" #--shuffle_off
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
                    if [ "$net" == imagenet ]; then
                        python -u attacks.py --net "$net" --attack "$att" --img_size 32 --num_classes 1000 --wanted_samples_tr "$WANTEDSAMPLES_TR" --wanted_samples_te "$WANTEDSAMPLES_TE" --eps "$eps" --run_nr "$run" 
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
                        python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000 --run_nr "$run"  --eps "$eps" --wanted_samples_tr "$WANTEDSAMPLES_TR" --wanted_samples_te "$WANTEDSAMPLES_TE" --take_inputimage_off
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
                                python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det"  --wanted_samples_tr "$nrsamples"  --wanted_samples_te "$WANTEDSAMPLES_TE"  --clf "$classifier" --num_classes 1000  --eps "$eps"  --run_nr "$run"
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
# extractcharacteristics
# detectadversarials

# extractcharacteristicslayer
# detectadversarialslayer

# #------------------------------------------------------------------------------------------------------------
log_msg "finished"
exit 0



# python -u generate_clean_data.py --net imagenet --num_classes 1000 --wanted_samples 8000 --run_nr 4
# python -u attacks.py --net imagenet --attack "$att" --img_size 32 --batch_size 128 --num_classes 1000 --wanted_samples $NRWANTEDSAMPLES --all_samples 8000 --eps "$eps" --run_nr "$run"

# python -u attacks.py --net cif10 --attack  LayerMFS --img_size 32 --batch_size 128 --num_classes 1000 --wanted_samples 7000 --all_samples 8000  --run_nr 4


# python -u extract_characteristics.py --net imagenet --attack df --detector LayerMFS --num_classes 1000 --run_nr 4 --wanted_samples 4000 --take_inputimage_off
python -u extract_characteristics.py --net imagenet --attack df --detector LayerMFS --num_classes 1000 --run_nr 4 --wanted_samples 6000 --take_inputimage_off
