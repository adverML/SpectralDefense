#!/bin/bash

function log_msg {
  echo  "`date` $@"
}

# DATASETS=(cif10 cif10vgg cif100 cif100vgg imagenet imagenet32 imagenet64 imagenet128 celebaHQ32 celebaHQ64 celebaHQ128)
# DATASETS="cif10 cif10vgg  cif100 cif100vgg cif10_rb"
DATASETS="cif10"
# DATASETS="cif100vgg"
# DATASETS="cif10rn34 cif100rn34"
# DATASETS="cif10rn34sota"
# DATASETS="cif10_rb"
# DATASETS="cif10 cif10vgg cif10_rb cif10rn34 cif100 cif100vgg cif100rn34"

# DATASETS="cif10_rb"
# DATASETS="cif10vgg"

# DATASETS="imagenet64 celebaHQ64 imagenet128 celebaHQ128"
RUNS="1"

ATTACKS="fgsm bim pgd cw"
DETECTORS="VAEInputMFS"


# DETECTORS="DkNN"
# EPSILONS="8./255. 4./255. 2./255. 1./255. 0.5/255."
EPSILONS="8./255."

# CLF="LR"
CLF="RF"
# CLF="SVC"
# CLF="cuSVC"

# CLF="IF"
# CLF="RF"


WANTEDSAMPLES="2000"
ALLSAMPLES="3000"
# NRSAMPLES="2000"
NRSAMPLES="10000"
PCA_FEATURES="0"


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
                            done
                        done
                    done
                done
            done
        done
    done
}


detectadversarials

# #-----------------------------------------------------------------------------------------------------------------------------------
log_msg "finished"
exit 0