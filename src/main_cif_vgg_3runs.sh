#!/bin/bash

function log_msg {
  echo  "`date` $@"
}

# DATASETS=(cif10 cif10vgg cif100 cif100vgg imagenet imagenet32 imagenet64 imagenet128 celebaHQ32 celebaHQ64 celebaHQ128)
# DATASETS="cif10 cif10vgg cif100 cif100vgg"
DATASETS="cif10vgg"

ATTACKS="bim df"
# DETECTORS="InputPFS LayerPFS LID Mahalanobis"
# DETECTORS="InputPFS LayerPFS InputMFS LayerMFS LID Mahalanobis"
DETECTORS="InputMFSAnalysis"
# EPSILONS="8./255. 4./255. 2./255. 1./255. 0.5/255."
EPSILONS="8./255."

# CLF="LR RF"
CLF="LR"

IMAGENET32CLASSES="25 50 100 250 1000"
NRSAMPLES="1500"

##### Go through each layer
# NRSAMPLES="300 500 1000 1200 1500 2000" # only at detectadversarialslayer
DATASETSLAYERNR="cif10 cif10vgg"
ATTACKSLAYERNR="bim df"
# ATTACKSLAYERNR="fgsm bim pgd std df cw"
LAYERNR={0,1,2,3,4,5,6,7,8,9,10,11,12}
DETECTORSLAYERNR="LayerMFS LayerPFS"
NRRUN="8"

#-----------------------------------------------------------------------------------------------------------------------------------
log_msg "Networks are already trained!"
#-----------------------------------------------------------------------------------------------------------------------------------

genereratecleandata ()
{
    log_msg "Generate Clean Data for Foolbox Attacks and Autoattack!"
    for net in $DATASETS; do
        if [ "$net" == cif10 ]; then
            python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$NRRUN"
        fi 

        if [ "$net" == cif10vgg ]; then
            python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$NRRUN" 
        fi 

        if [ "$net" == cif100 ]; then
            python -u generate_clean_data.py --net "$net" --num_classes 100  --run_nr "$NRRUN"
        fi 

        if [ "$net" == cif100vgg ]; then
            python -u generate_clean_data.py --net "$net" --num_classes 100  --run_nr "$NRRUN"
        fi 
    done
}

#-----------------------------------------------------------------------------------------------------------------------------------
attacks ()
{
    log_msg "Attack Clean Data with Foolbox Attacks and Autoattack!"

    for net in $DATASETS; do
        for att in $ATTACKS; do
            for eps in $EPSILONS; do
                if [ "$net" == cif10 ]; then
                    if  [ "$att" == std ]; then                                
                        python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 500  --eps "$eps" --run_nr "$NRRUN"
                    else
                        python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 1500 --eps "$eps" --run_nr "$NRRUN"
                    fi
                fi

                if [ "$net" == cif10vgg ]; then
                    python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 1500  --eps "$eps"  --run_nr "$NRRUN"
                fi 

                if [ "$net" == cif100 ]; then
                    python -u attacks.py --net "$net" --num_classes 100 --attack "$att" --img_size 32 --batch_size 1000  --eps "$eps"  --run_nr "$NRRUN"
                fi 

                if [ "$net" == cif100vgg ]; then
                    python -u attacks.py --net "$net" --num_classes 100 --attack "$att" --img_size 32 --batch_size 1000  --eps "$eps"  --run_nr "$NRRUN"
                fi 
            done
        done
    done
}

#-----------------------------------------------------------------------------------------------------------------------------------
extractcharacteristics ()
{
    log_msg "Extract Characteristics"

    for net in $DATASETS; do
        for att in $ATTACKS; do
            for eps in $EPSILONS; do
                for det in $DETECTORS; do
                        if [ "$net" == cif10 ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det"  --num_classes 10 --eps "$eps" --run_nr "$NRRUN" --wanted_samples "$NRSAMPLES" --take_inputimage_off 
                        fi

                        if [ "$net" == cif10vgg ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 10   --eps "$eps"  --run_nr "$NRRUN"  --wanted_samples "$NRSAMPLES" --take_inputimage_off 
                        fi 

                        if [ "$net" == cif100 ]; then
                            if  [ "$att" == std ]; then                                
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 100  --eps "$eps"  --run_nr "$NRRUN" 
                            else 
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 100  --run_nr "$NRRUN"
                            fi
                        fi

                        if [ "$net" == cif100vgg ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 100   --run_nr "$NRRUN" 
                        fi 
                done
            done
        done
    done
}


extractcharacteristicslayer ()
{
    log_msg "Extract Characteristics Layer By Layer for WhiteBox"

    for net in $DATASETSLAYERNR; do
        for att in $ATTACKSLAYERNR; do
            for det in $DETECTORSLAYERNR; do
                for nr in {0,1,2,3,4,5,6,7,8,9,10,11,12}; do 
                    log_msg "Layer Nr. $nr; attack $att; detectors $det"
                    if [ "$net" == cif10 ]; then
                        python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 10 --nr "$nr" --run_nr  "$NRRUN"   --take_inputimage_off 
                    fi

                    if [ "$net" == cif10vgg ]; then
                        python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 10 --nr "$nr" --run_nr  "$NRRUN"  --take_inputimage_off 
                    fi 

                done
            done
        done
    done
}

# #-----------------------------------------------------------------------------------------------------------------------------------
detectadversarials ()
{
    log_msg "Detect Adversarials!"
    for net in $DATASETS; do
        for att in $ATTACKS; do
            for eps in $EPSILONS; do
                for det in $DETECTORS; do
                    for nrsamples in $NRSAMPLES; do
                        for classifier in $CLF; do
                                if [ "$net" == cif10 ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --eps "$eps" --num_classes 10  --run_nr "$NRRUN"
                                fi

                                if [ "$net" == cif10vgg ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier"  --eps "$eps" --num_classes 10  --run_nr "$NRRUN"
                                fi 

                                if [ "$net" == cif100 ]; then
                                    if  [ "$att" == std ]; then
                                            python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 100 --eps "$eps"  --run_nr "$NRRUN"
                                    else
                                        python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 100  --run_nr "$NRRUN"
                                    fi
                                fi

                                if [ "$net" == cif100vgg ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 100   --run_nr "$NRRUN"
                                fi 
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
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10 --nr "$nr" --run_nr "$NRRUN" 
                                fi

                                if [ "$net" == cif10vgg ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10 --nr "$nr" --run_nr "$NRRUN" 
                                fi 
                            done
                        done
                    done
                done
            done
    done
}


# genereratecleandata
# attacks
# extractcharacteristicslayer
# detectadversarialslayer

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


python -u generate_clean_data.py --net cif10vgg --num_classes 10  --run_nr 8 --wanted_samples 8000 --shuffle_off

python -u attacks.py --net cif10vgg --num_classes 10 --img_size 32 --run_nr 8 --wanted_samples 3500 --all_samples 7000 --attack bim

python -u extract_characteristics.py --net cif10vgg --attack bim --detector InputMFSAnalysis  --num_classes 10  --run_nr 8  --wanted_samples 3500 --take_inputimage_off  --fr 8 --to 24


python -u detect_adversarials.py --net cif10vgg --attack bim --detector InputMFSAnalysis --wanted_samples 3500 --clf RF  --num_classes 10  --run_nr 8
python -u detect_adversarials.py --net cif10vgg --attack bim --detector InputMFSAnalysis --wanted_samples 3500 --clf LR  --num_classes 10  --run_nr 8




python -u extract_characteristics.py --net imagenet --attack df  --detector InputMFSAnalysis  --num_classes 1000    --wanted_samples 3500  --run_nr 1 --take_inputimage_off  --fr 75 --to 150
python -u detect_adversarials.py     --net imagenet --attack bim --detector InputMFSAnalysis  --wanted_samples 3500 --clf LR  --num_classes 1000  --run_nr 1