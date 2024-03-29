#!/bin/bash

# SD_blackbox = InputMFS
# SD_whitebox = LayerMFS

function log_msg {
  echo  "`date` $@"
}

DATASETS="cif10"
RUNS="1"
ATTACKS="fgsm bim pgd std df cw"
DETECTORS="InputMFS LayerMFS"
# DETECTORS="LID Mahalanobis"
EPSILONS="8./255."

CLF="RF"
# CLF="LR"
# CLF="LR RF"

IMAGENET32CLASSES="25 50 100 250 1000"

WANTEDSAMPLES="2000"
ALLSAMPLES="5000"
# NRSAMPLES="1500" # detect
NRSAMPLES="2000" # detect

DATASETSLAYERNR="cif100 cif10vgg cif100vgg"
ATTACKSLAYERNR="df"

# LAYERNR="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
LAYERNR="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14"
DETECTORSLAYERNR="LayerMFS LayerPFS"
PCA_FEATURES="0"
LID_K="5 10 20 50"


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
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run" --wanted_samples "$ALLSAMPLES" #--shuffle_off
            fi 

            if [ "$net" == cif10_rb ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run" --wanted_samples "$ALLSAMPLES" #--shuffle_off
            fi 

            if [ "$net" == cif10vgg ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run" --wanted_samples "$ALLSAMPLES" #--shuffle_off
            fi 

            if [ "$net" == cif10rn34 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run" --wanted_samples "$ALLSAMPLES" #--shuffle_off
            fi 

            if [ "$net" == cif10rn34sota ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run" --wanted_samples "$ALLSAMPLES" #--shuffle_off
            fi 

            if [ "$net" == cif100rn34 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 100  --run_nr "$run" --wanted_samples "$ALLSAMPLES" #--shuffle_off
            fi 

            if [ "$net" == cif100 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 100  --run_nr "$run" --wanted_samples "$ALLSAMPLES" #--shuffle_off
            fi 

            if [ "$net" == cif100vgg ]; then 
                python -u generate_clean_data.py --net "$net" --num_classes 100  --run_nr "$run" --wanted_samples "$ALLSAMPLES" #--shuffle_off
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
                        python -u attacks.py --net "$net" --attack "$att"  --batch_size 500  --eps "$eps" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES" #--use_clean_train_data
                    fi

                    if [ "$net" == cif10_rb ]; then     
                        python -u attacks.py --net "$net" --attack "$att"  --batch_size 500  --eps "$eps" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                    fi

                    if [ "$net" == cif10vgg ]; then                            
                        python -u attacks.py --net "$net" --attack "$att"  --batch_size 500  --eps "$eps"  --run_nr "$run" --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                    fi 

                    if [ "$net" == cif10rn34 ]; then
                        python -u attacks.py --net "$net" --attack "$att"  --batch_size 500   --eps "$eps"  --run_nr "$run" --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                    fi      

                    if [ "$net" == cif10rn34sota ]; then
                        python -u attacks.py --net "$net" --attack "$att"  --batch_size 500   --eps "$eps"  --run_nr "$run" --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                    fi                    

                    if [ "$net" == cif100 ]; then
                        python -u attacks.py --net "$net" --attack "$att"  --batch_size 1000   --eps "$eps"  --run_nr "$run" --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                    fi 

                    if [ "$net" == cif100vgg ]; then
                        python -u attacks.py --net "$net" --attack "$att"  --batch_size 1000   --eps "$eps"  --run_nr "$run" --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                    fi 

                    if [ "$net" == cif100rn34 ]; then
                        python -u attacks.py --net "$net" --attack "$att"  --batch_size 1000   --eps "$eps"  --run_nr "$run" --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
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
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det"  --eps "$eps" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off
                        fi

                        if [ "$net" == cif10_rb ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --eps "$eps" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off 
                        fi 

                        if [ "$net" == cif10vgg ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --eps "$eps" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off 
                        fi 

                        if [ "$net" == cif10rn34 ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --eps "$eps" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off 
                        fi 

                        if [ "$net" == cif10rn34sota ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --eps "$eps"--run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off 
                        fi  

                        if [ "$net" == cif100 ]; then                            
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --eps "$eps"  --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off 
                        fi

                        if [ "$net" == cif100vgg ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --eps "$eps" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off 
                        fi 

                        if [ "$net" == cif100rn34 ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --eps "$eps" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off 
                        fi 
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
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --nr "$nr" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off
                        fi

                        if [ "$net" == cif10_rb ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --nr "$nr" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off
                        fi

                        if [ "$net" == cif10vgg ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --nr "$nr" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off
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
                for eps in $EPSILONS; do
                    for det in $DETECTORS; do
                        for nrsamples in $NRSAMPLES; do
                            for classifier in $CLF; do
                                if [ "$net" == cif10 ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --eps "$eps" --num_classes 10  --run_nr "$run"  --pca_features "$PCA_FEATURES"
                                fi

                                if [ "$net" == cif10_rb ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --eps "$eps" --num_classes 10  --run_nr "$run"  --pca_features "$PCA_FEATURES"
                                fi

                                if [ "$net" == cif10vgg ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10  --run_nr "$run"  --pca_features "$PCA_FEATURES"
                                fi 

                                if [ "$net" == cif10rn34 ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10  --run_nr "$run"  --pca_features "$PCA_FEATURES"
                                fi 

                                if [ "$net" == cif10rn34sota ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10  --run_nr "$run"  --pca_features "$PCA_FEATURES"
                                fi 


                                if [ "$net" == cif100 ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 100 --eps "$eps"  --run_nr "$run"  --pca_features "$PCA_FEATURES"
                                fi

                                if [ "$net" == cif100vgg ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 100   --run_nr "$run"  --pca_features "$PCA_FEATURES"
                                fi 
                                
                                if [ "$net" == cif100rn34 ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 100   --run_nr "$run"  --pca_features "$PCA_FEATURES"
                                fi 
                            done
                        done
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
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10 --nr "$nr" --run_nr  "$run" 
                                fi

                                if [ "$net" == cif10_rb ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10 --nr "$nr" --run_nr  "$run" 
                                fi

                                if [ "$net" == cif10vgg ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10 --nr "$nr" --run_nr  "$run" 
                                fi 
                            done
                        done
                    done
                done
            done
        done
    done
}

# printn
genereratecleandata
attacks
extractcharacteristics
detectadversarials

# extractcharacteristicslayer
# detectadversarialslayer

#-----------------------------------------------------------------------------------------------------------------------------------
log_msg "finished"
exit 0
