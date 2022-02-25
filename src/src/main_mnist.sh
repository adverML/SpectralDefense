#!/bin/bash

function log_msg {
  echo  "`date` $@"
}

DATASETS="mnist"

ATTACKS="fgsm bim std pgd df cw"
# DETECTORS="InputPFS LayerPFS LID Mahalanobis"
# DETECTORS="InputPFS LayerPFS InputMFS LayerMFS LID Mahalanobis"
DETECTORS="InputMFS LayerMFS"
# EPSILONS="4./255. 2./255. 1./255. 0.5/255."
EPSILONS="8./255."

CLF="LR RF"
# CLF="LR"

IMAGENET32CLASSES="25 50 100 250 1000"
# NRSAMPLES="300 500 1000 1200 1500 2000" # only at detectadversarialslayer
NRSAMPLES="1500"

NRRUN={1}
# NRRUN={1,2,3,4}


DATASETSLAYERNR="cif10"
ATTACKSLAYERNR="std"

# ATTACKSLAYERNR="fgsm bim pgd std df cw"

LAYERNR={0,1,2,3,4,5,6,7,8,9,10,11,12}
DETECTORSLAYERNR="InputMFS LayerMFS"


#-----------------------------------------------------------------------------------------------------------------------------------
log_msg "Networks are already trained!"
#-----------------------------------------------------------------------------------------------------------------------------------

genereratecleandata ()
{
    log_msg "Generate Clean Data for Foolbox Attacks and Autoattack!"
    for net in $DATASETS; do
        if [ "$net" == mnist ]; then
            python -u generate_clean_data.py --net "$net" --num_classes 10 
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
                if [ "$net" == mnist ]; then
                    if  [ "$att" == std ]; then                                
                        python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 500 --eps "$eps"
                    else
                        python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 1500 
                    fi
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
                        if [ "$net" == mnist ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det"  --num_classes 10 --eps "$eps"
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
                        python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 10 --nr "$nr" --run_nr 0
                    fi

                    if [ "$net" == cif10vgg ]; then
                        python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 10 --nr "$nr" --run_nr 0
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
                                    if [ "$net" == mnist ]; then
                                        python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --eps "$eps" --num_classes 10
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
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10 --nr "$nr" --run_nr 0
                                fi
                            done
                        done
                    done
                done
            done
    done
}

attacks
# extractcharacteristics
# detectadversarials


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