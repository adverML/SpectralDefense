#!/bin/bash

# python -u generate_clean_data.py --net cif10 --num_classes 10  --run_nr 10 --wanted_samples 9500 --shuffle_off
# python -u generate_clean_data.py --net cif10 --num_classes 10  --run_nr 11 --wanted_samples 9500 --shuffle_off
# python -u generate_clean_data.py --net cif10 --num_classes 10  --run_nr 12 --wanted_samples 9500 --shuffle_off
# python -u generate_clean_data.py --net cif10 --num_classes 10  --run_nr 13 --wanted_samples 9500 --shuffle_off
# python -u generate_clean_data.py --net cif10 --num_classes 10  --run_nr 14 --wanted_samples 9500 --shuffle_off
# python -u generate_clean_data.py --net cif10 --num_classes 10  --run_nr 15 --wanted_samples 9500 --shuffle_off
# python -u generate_clean_data.py --net cif10 --num_classes 10  --run_nr 16 --wanted_samples 9500 --shuffle_off

# python -u attacks.py --net cif10 --attack  fgsm  --batch_size 500  --eps 0.001 --run_nr 10  --wanted_samples 2000 --all_samples 9500
# python -u attacks.py --net cif10 --attack  fgsm  --batch_size 500  --eps 0.003 --run_nr 11  --wanted_samples 2000 --all_samples 9500
# python -u attacks.py --net cif10 --attack  fgsm  --batch_size 500  --eps 0.01  --run_nr 12  --wanted_samples 2000 --all_samples 9500
# python -u attacks.py --net cif10 --attack  fgsm  --batch_size 500  --eps 0.03  --run_nr 13  --wanted_samples 2000 --all_samples 9500
# python -u attacks.py --net cif10 --attack  fgsm  --batch_size 500  --eps 0.1   --run_nr 14  --wanted_samples 2000 --all_samples 9500
# python -u attacks.py --net cif10 --attack  fgsm  --batch_size 500  --eps 0.3   --run_nr 15  --wanted_samples 2000 --all_samples 9500
# python -u attacks.py --net cif10 --attack  fgsm  --batch_size 500  --eps 1     --run_nr 16  --wanted_samples 2000 --all_samples 9500

python -u extract_characteristics.py --net cif10 --attack fgsm --detector InputMFS   --run_nr 10  --wanted_samples 2000 --take_inputimage_off
python -u extract_characteristics.py --net cif10 --attack fgsm --detector InputMFS   --run_nr 11  --wanted_samples 2000 --take_inputimage_off
python -u extract_characteristics.py --net cif10 --attack fgsm --detector InputMFS   --run_nr 12  --wanted_samples 2000 --take_inputimage_off
python -u extract_characteristics.py --net cif10 --attack fgsm --detector InputMFS   --run_nr 13  --wanted_samples 2000 --take_inputimage_off
python -u extract_characteristics.py --net cif10 --attack fgsm --detector InputMFS   --run_nr 14  --wanted_samples 2000 --take_inputimage_off
python -u extract_characteristics.py --net cif10 --attack fgsm --detector InputMFS   --run_nr 15  --wanted_samples 2000 --take_inputimage_off
python -u extract_characteristics.py --net cif10 --attack fgsm --detector InputMFS   --run_nr 16  --wanted_samples 2000 --take_inputimage_off

python -u detect_adversarials.py --net cif10 --attack fgsm --detector InputMFS --wanted_samples 2000 --clf LR  --num_classes 10  --run_nr 10  --pca_features 0
python -u detect_adversarials.py --net cif10 --attack fgsm --detector InputMFS --wanted_samples 2000 --clf LR  --num_classes 10  --run_nr 11  --pca_features 0
python -u detect_adversarials.py --net cif10 --attack fgsm --detector InputMFS --wanted_samples 2000 --clf LR  --num_classes 10  --run_nr 12  --pca_features 0
python -u detect_adversarials.py --net cif10 --attack fgsm --detector InputMFS --wanted_samples 2000 --clf LR  --num_classes 10  --run_nr 13  --pca_features 0
python -u detect_adversarials.py --net cif10 --attack fgsm --detector InputMFS --wanted_samples 2000 --clf LR  --num_classes 10  --run_nr 14  --pca_features 0
python -u detect_adversarials.py --net cif10 --attack fgsm --detector InputMFS --wanted_samples 2000 --clf LR  --num_classes 10  --run_nr 15  --pca_features 0
python -u detect_adversarials.py --net cif10 --attack fgsm --detector InputMFS --wanted_samples 2000 --clf LR  --num_classes 10  --run_nr 16  --pca_features 0

