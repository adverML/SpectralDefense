#!/bin/bash


# bash main_imagnet128.sh  &> log_evaluation/imagenet128/all.log

function log_msg {
  echo "`date` $@"
}

# DATASETS=(cif10 cif10vgg cif100 cif100vgg imagenet imagenet32 imagenet64 imagenet128 celebaHQ32 celebaHQ64 celebaHQ128)
DATASETS="imagenet128"
# DATASETS="imagenet"
# ATTACKS="gauss fgsm bim pgd std df"
# ATTACKS="fgsm bim pgd std"
ATTACKS="df cw"

# ATTACKS="apgd-ce apgd-t fab-t square"
# RUNS="1 2 3"
RUNS="1"

# ATTACKS="cw"
# ATTACKS="fgsm bim pgd df cw"
# ATTACKS="fgsm bim pgd std df"

# EPSILONS="4./255. 2./255. 1./255. 0.5/255."
EPSILONS="8./255."


# DETECTORS="InputPFS LayerPFS InputMFS LayerMFS LID Mahalanobis"
DETECTORS="InputMFS"
CLF="LR RF"
IMAGENET32CLASSES="25 50 100 250 1000"
# NRSAMPLES="300 500 1000 1200 1500 2000"
NRSAMPLES="1500"

#-----------------------------------------------------------------------------------------------------------------------------------
log_msg "Networks are already trained!"
#-----------------------------------------------------------------------------------------------------------------------------------

genereratecleandata ()
{
    log_msg "Generate Clean Data for Foolbox Attacks and Autoattack!"
    for run in $RUNS; do
        for net in $DATASETS; do
            if [ "$net" == imagenet128 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 1000  --img_size 128 --run_nr "$run" 
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

                if [ "$net" == imagenet128 ]; then
                    python -u attacks.py --net "$net" --num_classes 1000 --attack "$att" --img_size 128 --batch_size 48 --run_nr "$run"
                fi 

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
                        python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000 --img_size 128 --wanted_samples 1500 --eps "$eps" --run_nr "$run"  --take_inputimage_off
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
                                python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 1000 --eps "$eps" --run_nr "$run"
                            done
                        done
                    done
                done
            done
        done
    done
}

# genereratecleandata
# attacks
# extractcharacteristics
detectadversarials


# #-----------------------------------------------------------------------------------------------------------------------------------
log_msg "finished"
exit 0




python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack fgsm
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack bim
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack std
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack pgd
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack df
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack cw


python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack fgsm
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack bim
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack std
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack pgd
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack df
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack cw