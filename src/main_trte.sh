#!/bin/bash

function log_msg {
  echo  "`date` $@"
}

# DATASETS=(cif10 cif10vgg cif100 cif100vgg imagenet imagenet32 imagenet64 imagenet128 celebaHQ32 celebaHQ64 celebaHQ128)
# DATASETS="cif10 cif10vgg cif100 cif100vgg"
# DATASETS="cif10 cif100"
# DATASETS="cif100vgg"
# DATASETS="cif10rn34 cif100rn34"
# DATASETS="cif100rn34"
# DATASETS="cif10_m"
# DATASETS="cif10rn34sota"
# DATASETS="cif10vgg cif100vgg"
# DATASETS="cif100"
# DATASETS="imagenet"
# DATASETS="cif100"

DATASETS="restricted_imagenet"

# DATASETS="imagenet64 celebaHQ64 imagenet128 celebaHQ128"
# RUNS="1 2 3"
RUNS="8"

# ATTACKS="fgsm bim pgd std df cw"
# ATTACKS="apgd-ce apgd-t fab-t square"
# ATTACKS="std df"
# ATTACKS="gauss"
ATTACKS="std df"



DETECTORS="InputMFS LayerMFS"
# DETECTORS="InputMFS LayerMFS LID Mahalanobis"
# DETECTORS="LayerMFS"
# DETECTORS="InputPFS LayerPFS InputMFS LayerMFS LID Mahalanobis"
# DETECTORS="LID Mahalanobis"
# EPSILONS="8./255. 4./255. 2./255. 1./255. 0.5/255."
EPSILONS="8./255."

# CLF="LR RF"
CLF="LR"

IMAGENET32CLASSES="25 50 100 250 1000"
# NRSAMPLES="300 500 1000 1200 1500 2000" # only at detectadversarialslayer

DATASETSLAYERNR="cif10rn34sota"
ATTACKSLAYERNR="bim"

# LAYERNR="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
LAYERNR="13 14"

# DETECTORSLAYERNR="LayerMFS LayerPFS"
DETECTORSLAYERNR="LayerMFS"

NRWANTEDSAMPLES="0"
NRSAMPLES="0"
# WANTEDSAMPLES_TR="5000"

# cif10 100
WANTEDSAMPLES_TR="30000"
WANTEDSAMPLES_TE="8600"
# WANTEDSAMPLES_TR="44000"
# WANTEDSAMPLES_TE="0"

# WANTEDSAMPLES_TE="2600"
# WANTEDSAMPLES_TR="5000"


#-----------------------------------------------------------------------------------------------------------------------------------
log_msg "Networks are already trained!"
#-----------------------------------------------------------------------------------------------------------------------------------

printn()
{
    for index in $RUNS; do
        echo "$index"
    done 
}


genereratecleandata ()
{
    log_msg "Generate Clean Data for Foolbox Attacks and Autoattack!"
    for run in $RUNS; do
        for net in $DATASETS; do
            if [ "$net" == cif10 ]; then
                python -u generate_clean_data_trte.py --net "$net"  --run_nr "$run" --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"  --wanted_samples_te "$WANTEDSAMPLES_TE" --shuffle_off
            fi 

            if [ "$net" == cif10_m ]; then
                python -u generate_clean_data_trte.py --net "$net"  --run_nr "$run" --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"  --wanted_samples_te "$WANTEDSAMPLES_TE"  --shuffle_off
            fi           

            if [ "$net" == cif10vgg ]; then
                python -u generate_clean_data_trte.py --net "$net"  --run_nr "$run" --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE" --shuffle_off
            fi 

            if [ "$net" == cif10rn34sota ]; then
                python -u generate_clean_data_trte.py --net "$net"  --run_nr "$run" --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE" --shuffle_off
            fi 

            if [ "$net" == cif10rn34 ]; then
                python -u generate_clean_data_trte.py --net "$net"  --run_nr "$run" --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE" --shuffle_off
            fi 

            if [ "$net" == cif100rn34 ]; then
                python -u generate_clean_data_trte.py --net "$net"  --run_nr "$run" --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE" --shuffle_off
            fi 

            if [ "$net" == cif100 ]; then
                python -u generate_clean_data_trte.py --net "$net"   --run_nr "$run" --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE" --shuffle_off
            fi 

            if [ "$net" == cif100vgg ]; then 
                python -u generate_clean_data_trte.py --net "$net"   --run_nr "$run" --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE" --shuffle_off
            fi 

            if [ "$net" == imagenet ]; then
                python -u generate_clean_data_trte.py --net "$net"  --run_nr "$run" --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE"
            fi      

            if [ "$net" == restricted_imagenet ]; then
                python -u generate_clean_data_trte.py --net "$net"  --run_nr "$run" --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE"
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
                    if [ "$net" == cif10 ]; then      
                       python -u attacks_trte.py --net "$net" --attack "$att"  --batch_size 500 --eps "$eps"  --run_nr "$run"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE"
                    fi

                    if [ "$net" == cif10vgg ]; then                             
                        python -u attacks_trte.py --net "$net" --attack "$att"  --batch_size 500 --eps "$eps"  --run_nr "$run"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE"
                    fi 

                    if [ "$net" == cif10rn34sota ]; then
                        python -u attacks_trte_trte.py --net "$net" --attack "$att"  --batch_size 500 --eps "$eps"  --run_nr "$run" --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE"
                    fi  

                    if [ "$net" == cif10rn34 ]; then                          
                        python -u attacks_trte.py --net "$net" --attack "$att"  --batch_size 500 --eps "$eps"  --run_nr "$run"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE"
                    fi                     

                    if [ "$net" == cif100 ]; then
                        python -u attacks_trte.py --net "$net"  --attack "$att"  --batch_size 1000  --run_nr "$run"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE"
                    fi 

                    if [ "$net" == cif100vgg ]; then
                        python -u attacks_trte.py --net "$net"  --attack "$att"  --batch_size 1000  --run_nr "$run"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE"
                    fi 

                    if [ "$net" == cif100rn34 ]; then
                        python -u attacks_trte.py --net "$net"  --attack "$att" --batch_size 1000  --run_nr "$run"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE"
                    fi 

                    if [ "$net" == imagenet ]; then
                        python -u attacks_trte.py --net "$net" --attack "$att"  --batch_size 500  --eps "$eps" --run_nr "$run"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE"
                    fi 

                    if [ "$net" == restricted_imagenet ]; then
                        python -u attacks_trte.py --net "$net" --attack "$att"  --batch_size 500  --eps "$eps" --run_nr "$run"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE"
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
                        if [ "$net" == cif10 ]; then
                            # python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det"  --eps "$eps" --run_nr "$run" --take_inputimage 
                            python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det"  --eps "$eps" --run_nr "$run" --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE" --take_inputimage_off
                        fi

                        if [ "$net" == cif10vgg ]; then
                            python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det" --eps "$eps"  --run_nr "$run"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE" --take_inputimage_off 
                        fi 

                        if [ "$net" == cif10rn34 ]; then
                            python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det" --eps "$eps"  --run_nr "$run"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE"--take_inputimage_off 
                        fi 

                        if [ "$net" == cif10rn34sota ]; then
                            python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det"  --eps "$eps" --run_nr "$run"   --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE" --take_inputimage_off 
                        fi 

                        if [ "$net" == cif100 ]; then                              
                            python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det" --num_classes 100  --eps "$eps"  --run_nr "$run"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE" --take_inputimage_off 
                        fi

                        if [ "$net" == cif100vgg ]; then
                            python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det" --num_classes 100   --eps "$eps" --run_nr "$run"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE"  --take_inputimage_off 
                        fi 

                        if [ "$net" == cif100rn34 ]; then
                            python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det" --num_classes 100  --eps "$eps"  --run_nr "$run"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE"  --take_inputimage_off 
                        fi

                        if [ "$net" == imagenet ]; then
                            python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000 --run_nr "$run"  --eps "$eps" --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE" --take_inputimage_off
                        fi

                        if [ "$net" == restricted_imagenet ]; then
                            python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000 --run_nr "$run"  --eps "$eps" --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE" --take_inputimage_off
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
                                python -u detect_adversarials_trte.py --net "$net" --attack "$att" --detector "$det"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR" --wanted_samples_te "$WANTEDSAMPLES_TE" --clf "$classifier" --eps "$eps" --num_classes 1000  --run_nr "$run"
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
                        if [ "$net" == cif10 ]; then
                            python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det" --nr "$nr" --run_nr "$run"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE" --take_inputimage_off
                        fi

                        if [ "$net" == cif10vgg ]; then
                            python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det" --nr "$nr" --run_nr "$run"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE" --take_inputimage_off
                        fi 

                        if [ "$net" == cif10rn34sota ]; then
                            python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det" --nr "$nr" --run_nr "$run"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE" --take_inputimage_off 
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
                                if [ "$net" == cif10 ]; then
                                    python -u detect_adversarials_trte.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --nr "$nr" --run_nr  "$run" 
                                fi

                                if [ "$net" == cif10vgg ]; then
                                    python -u detect_adversarials_trte.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --nr "$nr" --run_nr  "$run" 
                                fi 

                                if [ "$net" == cif10rn34sota ]; then
                                    python -u detect_adversarials_trte.py --net "$net" --attack "$att" --detector "$det"  --clf "$classifier" --nr "$nr" --run_nr  "$run"   --wanted_samples 0 --wanted_samples_tr 2000   --wanted_samples_te 500 
                                fi 
                            done
                        done
                    done
                done
            done
        done
    done
}


# extractcharacteristicslayer
# detectadversarialslayer

# printn
# genereratecleandata
# attacks
extractcharacteristics
detectadversarials

#-----------------------------------------------------------------------------------------------------------------------------------
log_msg "finished"
exit 0
