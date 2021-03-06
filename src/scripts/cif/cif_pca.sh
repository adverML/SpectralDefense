#!/bin/bash

function log_msg {
  echo  "`date` $@"
}

# DATASETS=(cif10 cif10vgg cif100 cif100vgg imagenet imagenet32 imagenet64 imagenet128 celebaHQ32 celebaHQ64 celebaHQ128)
# DATASETS="cif10 cif10vgg cif100 cif100vgg"
# DATASETS="cif10 cif100"
# DATASETS="cif100vgg"
# DATASETS="cif10rn34 cif100rn34"
# DATASETS="cif10rn34sota"
DATASETS="cif10"

# DATASETS="imagenet64 celebaHQ64 imagenet128 celebaHQ128"
# RUNS="1 2 3"
RUNS="8"

# ATTACKS="fgsm bim pgd std df cw"
# ATTACKS="apgd-ce apgd-t fab-t square"
# ATTACKS="std"
ATTACKS="df"

# ATTACKS="apgd-ce"

DETECTORS="InputMFS"
# DETECTORS="InputMFS LayerMFS LID Mahalanobis"
# DETECTORS="LayerMFS"
# DETECTORS="InputPFS LayerPFS InputMFS LayerMFS LID Mahalanobis"
# DETECTORS="LID Mahalanobis"
# EPSILONS="8./255. 4./255. 2./255. 1./255. 0.5/255."
EPSILONS="8./255."

# CLF="LR RF"
# CLF="IF"
CLF="LR"



IMAGENET32CLASSES="25 50 100 250 1000"
# NRSAMPLES="300 500 1000 1200 1500 2000" # only at detectadversarialslayer

WANTEDSAMPLES="2000"
ALLSAMPLES="4500"
NRSAMPLES="2000" # detect

DATASETSLAYERNR="cif10"
ATTACKSLAYERNR="df"

LAYERNR="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
LAYERNR="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14"
DETECTORSLAYERNR="LayerMFS LayerPFS"

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
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run" --wanted_samples "$ALLSAMPLES" --shuffle_off
            fi 

            if [ "$net" == cif10vgg ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run" --wanted_samples "$ALLSAMPLES" --shuffle_off
            fi 

            if [ "$net" == cif10rn34 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run" --wanted_samples "$ALLSAMPLES" --shuffle_off
            fi 

            if [ "$net" == cif10rn34sota ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run" --wanted_samples "$ALLSAMPLES" --shuffle_off
            fi 

            if [ "$net" == cif100rn34 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 100  --run_nr "$run" --wanted_samples "$ALLSAMPLES" --shuffle_off
            fi 

            if [ "$net" == cif100 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 100  --run_nr "$run" --wanted_samples "$ALLSAMPLES" --shuffle_off
            fi 

            if [ "$net" == cif100vgg ]; then 
                python -u generate_clean_data.py --net "$net" --num_classes 100  --run_nr "$run" --wanted_samples "$ALLSAMPLES" --shuffle_off
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
                        if  [ "$att" == std ]; then   
                            if  [ "$run" == 8 ]; then        
                                python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 500 --eps "$eps"  --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                            else
                                python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 500 --eps "$eps"  --run_nr "$run" --shuffle_on --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                            fi 
                        else 
                            python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 1500  --net_normalization --run_nr "$run" --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                        fi 
                    fi

                    if [ "$net" == cif10vgg ]; then
                        if  [ "$att" == std ]; then                                
                            python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 500 --eps "$eps"  --run_nr "$run" --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                        else
                            python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 1500  --net_normalization --run_nr "$run" --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                        fi
                    fi 

                    if [ "$net" == cif10rn34 ]; then
                        if  [ "$att" == std ]; then                                 
                            python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 500 --eps "$eps"  --run_nr "$run" --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                        else
                            python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 1500  --net_normalization --run_nr "$run" --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                        fi
                    fi      

                    if [ "$net" == cif10rn34sota ]; then
                        if  [ "$att" == std ]; then                                 
                            python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 500 --eps "$eps"  --run_nr "$run" --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                        else
                            python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 1500  --net_normalization --run_nr "$run" --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                        fi
                    fi                    

                    if [ "$net" == cif100 ]; then
                        python -u attacks.py --net "$net" --num_classes 100 --attack "$att" --img_size 32 --batch_size 1000  --run_nr "$run" --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                    fi 

                    if [ "$net" == cif100vgg ]; then
                        python -u attacks.py --net "$net" --num_classes 100 --attack "$att" --img_size 32 --batch_size 1000  --run_nr "$run" --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
                    fi 

                    if [ "$net" == cif100rn34 ]; then
                        python -u attacks.py --net "$net" --num_classes 100 --attack "$att" --img_size 32 --batch_size 1000  --run_nr "$run" --wanted_samples "$WANTEDSAMPLES" --all_samples "$ALLSAMPLES"
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
                                # python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det"  --num_classes 10 --eps "$eps" --run_nr "$run" --take_inputimage 
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det"  --num_classes 10 --eps "$eps" --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off
                            fi

                            if [ "$net" == cif10vgg ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 10   --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off 
                            fi 

                            if [ "$net" == cif10rn34 ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 10   --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off 
                            fi 

                            if [ "$net" == cif10rn34sota ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 10   --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off 
                            fi  

                            if [ "$net" == cif100 ]; then
                                if  [ "$att" == std ]; then                                
                                    python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 100  --eps "$eps"  --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off 
                                else 
                                    python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 100  --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off 
                                fi
                            fi

                            if [ "$net" == cif100vgg ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 100   --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off 
                            fi 

                            if [ "$net" == cif100rn34 ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 100   --run_nr "$run"  --wanted_samples "$WANTEDSAMPLES"  --take_inputimage_off 
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
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 10 --nr "$nr" --run_nr "$run"   --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off
                        fi

                        if [ "$net" == cif10vgg ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 10 --nr "$nr" --run_nr "$run"   --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off
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
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --eps "$eps" --num_classes 10  --run_nr "$run"  --pca_features 100
                                fi

                                if [ "$net" == cif10vgg ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10  --run_nr "$run"
                                fi 

                                if [ "$net" == cif10rn34 ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10  --run_nr "$run"
                                fi 

                                if [ "$net" == cif10rn34sota ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10  --run_nr "$run"
                                fi 


                                if [ "$net" == cif100 ]; then
                                    if  [ "$att" == std ]; then
                                            python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 100 --eps "$eps"  --run_nr "$run"
                                    else
                                        python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 100  --run_nr "$run"
                                    fi
                                fi

                                if [ "$net" == cif100vgg ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 100   --run_nr "$run"
                                fi 
                                
                                if [ "$net" == cif100rn34 ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 100   --run_nr "$run"
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


# extractcharacteristicslayer
# detectadversarialslayer

# printn
# genereratecleandata
# attacks
# extractcharacteristics
detectadversarials


# python attacks.py --net cif10 --att std --batch_size 500 --eps 4./255.
# python attacks.py --net cif10 --att std --batch_size 500 --eps 2./255.
# python attacks.py --net cif10 --att std --batch_size 500 --eps 1./255.
# python attacks.py --net cif10 --att std --batch_size 500 --eps 0.5/255.


# python attacks.py --net cif10vgg --att std --batch_size 500 --eps 4./255.
# python attacks.py --net cif10vgg --att std --batch_size 500 --eps 2./255.
# python attacks.py --net cif10vgg --att std --batch_size 500 --eps 1./255.
# python attacks.py --net cif10vgg --att std --batch_size 500 --eps 0.5/255.




# #-----------------------------------------------------------------------------------------------------------------------------------
log_msg "finished"
exit 0



python -u attacks.py --net cif10vgg --num_classes 10 --attack std --img_size 32 --batch_size 500 --eps 3./1000.  --run_nr 8

# EPSILONS="1./1. 1./10."
# EPSILONS="1./100. 1./1000."


python -u extract_characteristics.py --net cif100vgg --attack df --detector LayerMFS --num_classes 100  --run_nr 8  --wanted_samples 7210  --take_inputimage_off 

python -u extract_characteristics.py --net cif10vgg --attack std --detector InputMFS --num_classes 10 --nr -1 --run_nr 8 --eps 3./1000.   --take_inputimage_off

python -u detect_adversarials.py --net cif10vgg --attack std --detector InputMFS --wanted_samples 1500 --clf LR --num_classes 10  --run_nr 8 --eps 1./1.
python -u detect_adversarials.py --net cif10vgg --attack std --detector InputMFS --wanted_samples 1500 --clf LR --num_classes 10  --run_nr 8 --eps 1./10.
python -u detect_adversarials.py --net cif10vgg --attack std --detector InputMFS --wanted_samples 1500 --clf LR --num_classes 10  --run_nr 8 --eps 1./100.
python -u detect_adversarials.py --net cif10vgg --attack std --detector InputMFS --wanted_samples 1500 --clf LR --num_classes 10  --run_nr 8 --eps 3./1000.
python -u detect_adversarials.py --net cif10vgg --attack std --detector InputMFS --wanted_samples 1500 --clf LR --num_classes 10  --run_nr 8 --eps 1./1000.


python -u extract_characteristics.py --net cif100vgg --attack df --detector LayerMFS --num_classes 100  --run_nr 8  --wanted_samples 7210  --take_inputimage_off 


python -u detect_adversarials.py --net cif100vgg --attack df --detector LayerMFS --wanted_samples 7210 --clf "$classifier" --num_classes 10 --nr "$nr" --run_nr  "$run"


python -u extract_characteristics.py --net cif10 --attack df --detector DkNN --num_classes 10  --run_nr 8  --wanted_samples 7210  --take_inputimage_off 