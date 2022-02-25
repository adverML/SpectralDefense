#!/bin/bash

function log_msg {
  echo  "`date` $@"
}

# DATASETS=(cif10 cif10vgg cif100 cif100vgg imagenet imagenet32 imagenet64 imagenet128 celebaHQ32 celebaHQ64 celebaHQ128)
# DATASETS="cif10 cif10vgg cif100 cif100vgg"
DATASETS="cif10 cif100"
# DATASETS="cif10rn34 cif100rn34"
# DATASETS="cif100rn34"
# DATASETS="imagenet64 celebaHQ64 imagenet128 celebaHQ128"
RUNS="10"

# ATTACKS="fgsm bim pgd std df cw"
ATTACKS="df cw"
DETECTORS="LayerMFS"
# DETECTORS="InputPFS LayerPFS LID Mahalanobis"
# DETECTORS="InputPFS LayerPFS InputMFS LayerMFS LID Mahalanobis"
# DETECTORS="LID Mahalanobis"
# EPSILONS="8./255. 4./255. 2./255. 1./255. 0.5/255."
EPSILONS="8./255."

CLF="LR RF"
# CLF="LR"

IMAGENET32CLASSES="25 50 100 250 1000"
# NRSAMPLES="300 500 1000 1200 1500 2000" # only at detectadversarialslayer
NRSAMPLES="1500"


DATASETSLAYERNR="cif10 cif10vgg"
ATTACKSLAYERNR="df"
# ATTACKSLAYERNR="fgsm bim pgd std df cw"
NRRUN=5
LAYERNR={0,1,2,3,4,5,6,7,8,9,10,11,12}
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
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run"
            fi 

            if [ "$net" == cif10vgg ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run"
            fi 

            if [ "$net" == cif10rn34 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run"
            fi 

            if [ "$net" == cif100rn34 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 100  --run_nr "$run"
            fi 

            if [ "$net" == cif100 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 100  --run_nr "$run"
            fi 

            if [ "$net" == cif100vgg ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 100  --run_nr "$run"
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
                            python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 500 --eps "$eps"  --run_nr "$run"
                        else
                            python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 1500  --net_normalization --run_nr "$run"
                        fi
                    fi

                    if [ "$net" == cif10vgg ]; then
                        if  [ "$att" == std ]; then                                
                            python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 500 --eps "$eps"  --run_nr "$run"
                        else
                            python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 1500  --net_normalization --run_nr "$run"
                        fi
                    fi 

                    if [ "$net" == cif10rn34 ]; then
                        if  [ "$att" == std ]; then                                
                            python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 500 --eps "$eps"  --run_nr "$run"
                        else
                            python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 1500  --net_normalization --run_nr "$run"
                        fi
                    fi                     

                    if [ "$net" == cif100 ]; then
                        python -u attacks.py --net "$net" --num_classes 100 --attack "$att" --img_size 32 --batch_size 1000  --run_nr "$run"
                    fi 

                    if [ "$net" == cif100vgg ]; then
                        python -u attacks.py --net "$net" --num_classes 100 --attack "$att" --img_size 32 --batch_size 1000  --run_nr "$run"
                    fi 

                    if [ "$net" == cif100rn34 ]; then
                        python -u attacks.py --net "$net" --num_classes 100 --attack "$att" --img_size 32 --batch_size 1000  --run_nr "$run"
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
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det"  --num_classes 10 --eps "$eps" --run_nr "$run"  --take_inputimage_off
                            fi

                            if [ "$net" == cif10vgg ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 10   --run_nr "$run" --take_inputimage_off 
                            fi 

                            if [ "$net" == cif10rn34 ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 10   --run_nr "$run" --take_inputimage_off 
                            fi 

                            if [ "$net" == cif100 ]; then
                                if  [ "$att" == std ]; then                                
                                    python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 100  --eps "$eps"  --run_nr "$run"   --take_inputimage_off 
                                else 
                                    python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 100  --run_nr "$run"  --take_inputimage_off 
                                fi
                            fi

                            if [ "$net" == cif100vgg ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 100   --run_nr "$run"  --take_inputimage_off 
                            fi 

                            if [ "$net" == cif100rn34 ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 100   --run_nr "$run"  --take_inputimage_off 
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
                    for nr in {0,1,2,3,4,5,6,7,8,9,10,11,12}; do 
                        log_msg "Layer Nr. $nr; attack $att; detectors $det"
                        if [ "$net" == cif10 ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 10 --nr "$nr" --run_nr "$run"   --take_inputimage_off
                        fi

                        if [ "$net" == cif10vgg ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 10 --nr "$nr" --run_nr "$run"   --take_inputimage_off
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
                                            python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --eps "$eps" --num_classes 10  --run_nr "$run"
                                        fi

                                        if [ "$net" == cif10vgg ]; then
                                            python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10  --run_nr "$run"
                                        fi 

                                        if [ "$net" == cif10rn34 ]; then
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
    for net in $DATASETSLAYERNR; do
            for att in $ATTACKSLAYERNR; do
                for det in $DETECTORSLAYERNR; do
                    for nrsamples in $NRSAMPLES; do
                        for classifier in $CLF; do
                            for nr in {0,1,2,3,4,5,6,7,8,9,10,11,12}; do 
                                log_msg "Layer Nr. $nr"
                                if [ "$net" == cif10 ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10 --nr "$nr" --run_nr  "$NRRUN" 
                                fi

                                if [ "$net" == cif10vgg ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10 --nr "$nr" --run_nr  "$NRRUN" 
                                fi 
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
