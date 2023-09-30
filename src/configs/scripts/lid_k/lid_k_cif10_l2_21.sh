#!/bin/bash

function log_msg {
  echo  "`date` $@"
}

# DATASETS=(cif10 cif10vgg cif100 cif100vgg imagenet imagenet32 imagenet64 imagenet128 celebaHQ32 celebaHQ64 celebaHQ128)
DATASETS="cif10"
# DATASETS="cif10_rb"
# DATASETS="cif10 cif10vgg cif10_rb cif10rn34 cif100 cif100vgg cif100rn34"

# RUNS="1 2 3"
# RUNS="4 10 11 12"
RUNS="21"

# ATTACKS="gauss fgsm bim pgd std df cw"
ATTACKS="apgd-cel2"
# ATTACKS="apgd-ce"

# DETECTORS="LID"
DETECTORS="LIDNOISE"

# EPSILONS="8./255. 4./255. 2./255. 1./255. 0.5/255."
EPSILONS="0.5 0.4 0.3 0.2 0.1"
# EPSILONS="0.1"


# CLF="RF"
CLF="LR"
# CLF="SVC"
# CLF="cuSVC"
# CLF="IF"
# CLF="LR RF"

IMAGENET32CLASSES="25 50 100 250 1000"
# NRSAMPLES="300 500 1000 1200 1500 2000" # only at detectadversarialslayer

WANTEDSAMPLES="2000"
ALLSAMPLES="9600"
# NRSAMPLES="1500"
NRSAMPLES="2000" # detect


DATASETSLAYERNR="cif10"
ATTACKSLAYERNR="std"

# LAYERNR="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
LAYERNR="0 1 2 3 4 5 6 7 8 9 10 11 12"
DETECTORSLAYERNR="LID"
PCA_FEATURES="0"
# LID_K="3 5 10 20 50"
LID_K="3 5 10 30 50"


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
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run" --wanted_samples "$ALLSAMPLES" 
            fi 

            if [ "$net" == cif10_rb ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run" --wanted_samples "$ALLSAMPLES" 
            fi 

            if [ "$net" == cif10vgg ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run" --wanted_samples "$ALLSAMPLES" 
            fi 

            if [ "$net" == cif10rn34 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run" --wanted_samples "$ALLSAMPLES" 
            fi 

            if [ "$net" == cif10rn34sota ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run" --wanted_samples "$ALLSAMPLES" 
            fi 

            if [ "$net" == cif100rn34 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 100  --run_nr "$run" --wanted_samples "$ALLSAMPLES" 
            fi 

            if [ "$net" == cif100 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 100  --run_nr "$run" --wanted_samples "$ALLSAMPLES" 
            fi 

            if [ "$net" == cif100vgg ]; then 
                python -u generate_clean_data.py --net "$net" --num_classes 100  --run_nr "$run" --wanted_samples "$ALLSAMPLES" 
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
                        python -u attacks.py --net "$net" --attack "$att"  --batch_size 500  --eps "$eps" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
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
                        for lidk in $LID_K; do      
                            if [ "$net" == cif10 ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det"  --eps "$eps" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off --k_lid "$lidk"
                            fi

                            if [ "$net" == cif10_rb ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --eps "$eps" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off  --k_lid "$lidk"
                            fi 

                            if [ "$net" == cif10vgg ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --eps "$eps" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off  --k_lid "$lidk"
                            fi 

                            if [ "$net" == cif10rn34 ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --eps "$eps" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off  --k_lid "$lidk"
                            fi 

                            if [ "$net" == cif10rn34sota ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --eps "$eps"--run_nr "$run"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off  --k_lid "$lidk"
                            fi  

                            if [ "$net" == cif100 ]; then                            
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --eps "$eps"  --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off  --k_lid "$lidk"
                            fi

                            if [ "$net" == cif100vgg ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --eps "$eps" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off  --k_lid "$lidk"
                            fi 

                            if [ "$net" == cif100rn34 ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --eps "$eps" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off  --k_lid "$lidk"
                            fi 
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
                for eps in $EPSILONS; do
                    for det in $DETECTORSLAYERNR; do
                        for nr in $LAYERNR; do 
                            for lidk in $LID_K; do      
                                log_msg "Layer Nr. $nr; attack $att; detectors $det"
                                if [ "$net" == cif10 ]; then
                                    python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det"  --eps "$eps" --run_nr "$run" --nr "$nr" --k_lid "$lidk" --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off
                                fi

                                if [ "$net" == cif10_rb ]; then
                                    python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --nr "$nr" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off
                                fi

                                if [ "$net" == cif10vgg ]; then
                                    python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --nr "$nr" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off
                                fi 
                            done
                        done
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
                        for lidk in $LID_K; do                        
                            for nrsamples in $NRSAMPLES; do
                                for classifier in $CLF; do
                                    if [ "$net" == cif10 ]; then
                                        python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --eps "$eps" --num_classes 10  --run_nr "$run"  --pca_features "$PCA_FEATURES"  --k_lid "$lidk" --lid_k_log
                                    fi

                                    if [ "$net" == cif10_rb ]; then
                                        python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --eps "$eps" --num_classes 10  --run_nr "$run"  --pca_features "$PCA_FEATURES"  --k_lid "$lidk"
                                    fi

                                    if [ "$net" == cif10vgg ]; then
                                        python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10  --run_nr "$run"  --pca_features "$PCA_FEATURES"  --k_lid "$lidk"
                                    fi 

                                    if [ "$net" == cif10rn34 ]; then
                                        python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10  --run_nr "$run"  --pca_features "$PCA_FEATURES"  --k_lid "$lidk"
                                    fi 

                                    if [ "$net" == cif10rn34sota ]; then
                                        python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10  --run_nr "$run"  --pca_features "$PCA_FEATURES"  --k_lid "$lidk"
                                    fi 


                                    if [ "$net" == cif100 ]; then
                                        python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 100 --eps "$eps"  --run_nr "$run"  --pca_features "$PCA_FEATURES"  --k_lid "$lidk"
                                    fi

                                    if [ "$net" == cif100vgg ]; then
                                        python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 100   --run_nr "$run"  --pca_features "$PCA_FEATURES"  --k_lid "$lidk"
                                    fi 
                                    
                                    if [ "$net" == cif100rn34 ]; then
                                        python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 100   --run_nr "$run"  --pca_features "$PCA_FEATURES"  --k_lid "$lidk"
                                    fi 
                                done
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
                                for lidk in $LID_K; do 
                                    log_msg "Layer Nr. $nr"
                                    if [ "$net" == cif10 ]; then
                                        python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --k_lid "$lidk" --num_classes 10 --nr "$nr" --run_nr  "$run" 
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
    done
}


# printn
# genereratecleandata
# attacks
# extractcharacteristics
detectadversarials

# extractcharacteristicslayer
# detectadversarialslayer

# #-----------------------------------------------------------------------------------------------------------------------------------
log_msg "finished"
exit 0

