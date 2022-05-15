#!/bin/bash

function log_msg {
  echo  "`date` $@"
}


DATASETS="cif10"

RUNS="1"

ATTACKS="std"
# ATTACKS="df"

DETECTORS="InputMFSAnalysis"

CLF="RF"

IMAGENET32CLASSES="25 50 100 250 1000"
# NRSAMPLES="300 500 1000 1200 1500 2000" # only at detectadversarialslayer

WANTEDSAMPLES="2000"
ALLSAMPLES="3000"
# NRSAMPLES="2000" # detect
NRSAMPLES="1500" # detect

DATASETSLAYERNR="cif10"
ATTACKSLAYERNR="df"

# LAYERNR="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
LAYERNR="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14"
DETECTORSLAYERNR="LayerMFS LayerPFS"
PCA_FEATURES="0"



# python -u extract_characteristics.py --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis"  --run_nr "1"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off  --fr 0 --to 8
# python -u detect_adversarials.py --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis" --wanted_samples "2000" --clf "RF" --num_classes 10  --run_nr "1" 



# python -u extract_characteristics.py --net "cif10" --attack"$ATTACKS" --detector "InputMFSAnalysis"  --run_nr "1"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off  --fr 0 --to 16
# python -u detect_adversarials.py --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis" --wanted_samples "2000" --clf "RF" --num_classes 10  --run_nr "1" 



# python -u extract_characteristics.py --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis"  --run_nr "1"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off  --fr 0 --to 24
# python -u detect_adversarials.py --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis" --wanted_samples "2000" --clf "RF" --num_classes 10  --run_nr "1" 


# python -u extract_characteristics.py --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis"  --run_nr "1"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off  --fr 0 --to 32
# python -u detect_adversarials.py --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis" --wanted_samples "2000" --clf "RF" --num_classes 10  --run_nr "1" 



# python -u extract_characteristics.py --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis"  --run_nr "1"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off  --fr 8 --to 16
# python -u detect_adversarials.py --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis" --wanted_samples "2000" --clf "RF" --num_classes 10  --run_nr "1" 



python -u extract_characteristics.py --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis"  --run_nr "1"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off  --fr 8 --to 24
python -u detect_adversarials.py     --net "cif10" --attack "$ATTACKS"     --detector "InputMFSAnalysis"  --wanted_samples "2000" --clf "RF" --num_classes 10  --run_nr "1" 


python -u extract_characteristics.py --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis"  --run_nr "1"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off  --fr 8 --to 32
python -u detect_adversarials.py     --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis" --wanted_samples "2000" --clf "RF" --num_classes 10  --run_nr "1" 


python -u extract_characteristics.py --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis"  --run_nr "1"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off  --fr 16 --to 24
python -u detect_adversarials.py     --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis" --wanted_samples "2000" --clf "RF" --num_classes 10  --run_nr "1" 

python -u extract_characteristics.py --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis"  --run_nr "1"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off  --fr 16 --to 32
python -u detect_adversarials.py --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis" --wanted_samples "2000" --clf "RF" --num_classes 10  --run_nr "1" 

python -u extract_characteristics.py --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis"  --run_nr "1"  --wanted_samples "$WANTEDSAMPLES" --take_inputimage_off  --fr 24 --to 32
python -u detect_adversarials.py --net "cif10" --attack "$ATTACKS" --detector "InputMFSAnalysis" --wanted_samples "2000" --clf "RF" --num_classes 10  --run_nr "1" 

