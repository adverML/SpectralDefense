#!/bin/bash

# bash main_imagenet3264.sh  &> log_evaluation/imagenet3264/all.log

function log_msg {
  echo  "`date` $@"
}

# DATASETS=(cif10 cif10vgg cif100 cif100vgg imagenet imagenet32 imagenet64 imagenet128 celebaHQ32 celebaHQ64 celebaHQ128)
DATASETS="imagenet32"
RUNS="8"

# ATTACKS="cw"
ATTACKS="df"

# DETECTORS="InputPFS LayerPFS LID Mahalanobis"
# DETECTORS="InputPFS LayerPFS InputMFS LayerMFS LID Mahalanobis"
DETECTORS="LayerMFS"
# EPSILONS="8./255. 4./255. 2./255. 1./255. 0.5/255."
EPSILONS="8./255."


CLF="LR RF"
# CLF="LR"

IMAGENET32CLASSES="25 50 100 250 1000"
# NRSAMPLES="300 500 1000 1200 1500 2000" # only at detectadversarialslayer
NRSAMPLES="1500"


DATASETSLAYERNR="imagenet32"
ATTACKSLAYERNR="df"

LAYERNR="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
DETECTORSLAYERNR="LayerMFS"


#-----------------------------------------------------------------------------------------------------------------------------------
log_msg "Networks are already trained!"
#-----------------------------------------------------------------------------------------------------------------------------------

genereratecleandata ()
{
    log_msg "Generate Clean Data for Foolbox Attacks and Autoattack!"
    for run in $RUNS; do
        for net in $DATASETS; do
            if [ "$net" == imagenet32 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 1000   --img_size 32  --run_nr "$run"
            fi 

            if [ "$net" == imagenet64 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 1000   --img_size 64  --run_nr "$run"
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
                    if [ "$net" == imagenet32 ]; then
                        python -u attacks.py --net "$net" --num_classes 1000 --attack "$att" --img_size 32 --batch_size 500  --eps "$eps"  --run_nr "$run"
                    fi 

                    if [ "$net" == imagenet64 ]; then
                        python -u attacks.py --net "$net" --num_classes 1000 --attack "$att" --img_size 64 --batch_size 500  --run_nr "$run"
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
                        if [ "$net" == imagenet32 ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000  --img_size 32  --run_nr "$run" --take_inputimage_off
                        fi
                        if [ "$net" == imagenet64 ]; then
                            if  [ "$att" == std ]; then
                                    python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000  --img_size 64  --eps "$eps"  --run_nr "$run" --take_inputimage_off
                            else
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000  --img_size 64   --run_nr "$run" --take_inputimage_off
                            fi
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
                                    if  [ "$att" == std ]; then
                                        python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 1000 --eps "$eps"  --run_nr "$run"
                                    else
                                        python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 1000   --run_nr "$run"
                                    fi
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
                        if [ "$net" == imagenet32 ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000 --nr "$nr" --run_nr "$run"   --take_inputimage_off
                        fi

                        if [ "$net" == imagenet64 ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000 --nr "$nr" --run_nr "$run" --img_size 64  --take_inputimage_off
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
                                if [ "$net" == imagenet32 ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 1000 --nr "$nr" --run_nr  "$run" 
                                fi
                                if [ "$net" == imagenet64 ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 1000 --nr "$nr" --run_nr  "$run" 
                                fi
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
# detectadversarials
extractcharacteristicslayer
detectadversarialslayer



# python attacks.py --net imagenet32 --att std --batch_size 500 --num_classes 1000 --eps 4./255.
# python attacks.py --net imagenet32 --att std --batch_size 500 --num_classes 1000 --eps 2./255.
# python attacks.py --net imagenet32 --att std --batch_size 500 --num_classes 1000 --eps 1./255.
# python attacks.py --net imagenet32 --att std --batch_size 500 --num_classes 1000 --eps 0.5/255.



# python attacks.py --net imagenet64 --att std --batch_size 500 --num_classes 1000 --img_size 64 --eps 4./255.
# python attacks.py --net imagenet64 --att std --batch_size 500 --num_classes 1000 --img_size 64 --eps 2./255.
# python attacks.py --net imagenet64 --att std --batch_size 500 --num_classes 1000 --img_size 64 --eps 1./255.
# python attacks.py --net imagenet64 --att std --batch_size 250 --num_classes 1000 --img_size 64 --eps 0.5/255.

# python -u extract_characteristics.py --net imagenet64 --num_classes 1000  --detector InputMFS --img_size 64 --attack fgsm
# python -u extract_characteristics.py --net imagenet64 --num_classes 1000  --detector InputMFS --img_size 64 --attack bim
# python -u extract_characteristics.py --net imagenet64 --num_classes 1000  --detector InputMFS --img_size 64 --attack std
# python -u extract_characteristics.py --net imagenet64 --num_classes 1000  --detector InputMFS --img_size 64 --attack pgd
# python -u extract_characteristics.py --net imagenet64 --num_classes 1000  --detector InputMFS --img_size 64 --attack df
# python -u extract_characteristics.py --net imagenet64 --num_classes 1000  --detector InputMFS --img_size 64 --attack cw


# python -u extract_characteristics.py --net imagenet64 --num_classes 1000  --detector LayerMFS --img_size 64 --attack fgsm
# python -u extract_characteristics.py --net imagenet64 --num_classes 1000  --detector LayerMFS --img_size 64 --attack bim
# python -u extract_characteristics.py --net imagenet64 --num_classes 1000  --detector LayerMFS --img_size 64 --attack std
# python -u extract_characteristics.py --net imagenet64 --num_classes 1000  --detector LayerMFS --img_size 64 --attack pgd
# python -u extract_characteristics.py --net imagenet64 --num_classes 1000  --detector LayerMFS --img_size 64 --attack df
# python -u extract_characteristics.py --net imagenet64 --num_classes 1000  --detector LayerMFS --img_size 64 --attack cw



# for nr in {1,2,3,4}; do
#     echo "Generate Clean Data:  run: $nr" 
#     python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data', net=['imagenet32'], dest='./log_evaluation/imagenet3264', run_nr=$nr)"
#     genereratecleandata
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/clean_data', net=['imagenet32'], dest='./log_evaluation/imagenet3264', run_nr=$nr)"
# done


# for nr in {1,2,3,4}; do
#     log_msg "Attacks: run: $nr" 
#     python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data',       net=['imagenet32'], dest='./log_evaluation/imagenet3264', run_nr=$nr)"
#     attacks
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/attacks',  net=['imagenet32'], dest='./log_evaluation/imagenet3264', run_nr=$nr)"
# done

# python -c "import evaluate_detection; evaluate_detection.clean_root_folders( root='./data/clean_data', net=['imagenet32', 'imagenet64']  )"
# python -c "import evaluate_detection; evaluate_detection.clean_root_folders( root='./data/attacks',    net=[['imagenet32', 'imagenet64'] )"

# extractcharacteristics

# genereratecleandata



############################################
# celebahq64
# for nr in 1; do
#     # python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=$nr)"
#     # extractcharacteristics
#     # python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/extracted_characteristics',    net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=$nr)"

#     # python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data', net=['imagenet64'], dest='./log_evaluation/imagenet3264', run_nr=$nr)"
#     extractcharacteristics
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/extracted_characteristics', net=['imagenet64'], dest='./log_evaluation/imagenet3264', run_nr=$nr)"
# done

# for nr in 1; do
#     detectadversarials
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/detection', net=['imagenet64'], dest='./log_evaluation/imagenet3264', run_nr=$nr)"
# done 

############################################
# celebahq32
# for nr in 1; do
#     detectadversarials
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/detection', net=['imagenet32'], dest='./log_evaluation/imagenet3264', run_nr=$nr)"
# done 


# for nr in 1; do
#     # python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=$nr)"
#     # extractcharacteristics
#     # python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/extracted_characteristics',    net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=$nr)"

#     python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data', net=['imagenet64'], dest='./log_evaluation/imagenet3264', run_nr=$nr)"
#     extractcharacteristics
    
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/extracted_characteristics', net=['imagenet64'], dest='./log_evaluation/imagenet3264', run_nr=$nr)"
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/detection', net=['imagenet64'], dest='./log_evaluation/imagenet3264', run_nr=$nr)"

# done

# for nr in 1; do
#     detectadversarials
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/detection', net=['imagenet64'], dest='./log_evaluation/imagenet3264', run_nr=$nr)"
# done 





# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/clean_data', net=['imagenet32', 'imagenet64'], dest='./log_evaluation/imagenet3264', run_nr=2)"
# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/attacks',    net=['imagenet32', 'imagenet64'], dest='./log_evaluation/imagenet3264', run_nr=2)"
# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/extracted_characteristics', net=['imagenet32'], dest='./log_evaluation/imagenet3264', run_nr=1)"
# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/detection', net=['imagenet32'], dest='./log_evaluation/imagenet3264', run_nr=1)"


# detectadversarials


# python -u extract_characteristics.py --net imagenet32  --detector LID --num_classes 1000 --attack fgsm
# python -u extract_characteristics.py --net imagenet32  --detector LID --num_classes 1000 --attack bim
# python -u extract_characteristics.py --net imagenet32  --detector LID --num_classes 1000 --attack std
# python -u extract_characteristics.py --net imagenet32  --detector LID --num_classes 1000 --attack pgd
# python -u extract_characteristics.py --net imagenet32  --detector LID --num_classes 1000 --attack df
# python -u extract_characteristics.py --net imagenet32  --detector LID --num_classes 1000 --attack cw


# python -u extract_characteristics.py --net imagenet32  --detector Mahalanobis --num_classes 1000 --attack fgsm
# python -u extract_characteristics.py --net imagenet32  --detector Mahalanobis --num_classes 1000 --attack bim
# python -u extract_characteristics.py --net imagenet32  --detector Mahalanobis --num_classes 1000 --attack std
# python -u extract_characteristics.py --net imagenet32  --detector Mahalanobis --num_classes 1000 --attack pgd
# python -u extract_characteristics.py --net imagenet32  --detector Mahalanobis --num_classes 1000 --attack df
# python -u extract_characteristics.py --net imagenet32  --detector Mahalanobis --num_classes 1000 --attack cw



# #-----------------------------------------------------------------------------------------------------------------------------------
log_msg "finished"
exit 0

# : <<'END'
#   just a comment!
# END

# TODO List
# [] Run one time
#    [x] Generate Clean Data
#    [x] Generate Attacks
#    [] Generate Extract Charactersitics
#    [] Opimtize Params Charactersitics
#       [] Input MFS PFS
#       [] Layer MFS PFS
#       [] LID
#       [] Mahannobis
#       [] Statistical Test
#    [] Generate LR RF
# [] Save in file structure
# [] Create CSV





python -u detect_adversarials.py --net imagenet32 --num_classes 1000  --wanted_samples 1500 --clf LR --detector InputMFS --attack df

