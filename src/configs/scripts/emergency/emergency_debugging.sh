python -u extract_characteristics.py --net cif10 --attack fgsm --detector LID  --eps "8./255." --run_nr 1  --wanted_samples 2000 --take_inputimage_off --k_lid 10

python -u detect_adversarials.py --net cif10 --attack fgsm --detector LID --wanted_samples 2000 --clf RF --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"

python -u extract_characteristics.py --net cif10 --attack fgsm --detector LID  --eps "8./255." --run_nr 1  --wanted_samples 2000 --take_inputimage_off 
python -u extract_characteristics.py --net cif10 --attack bim  --detector LID  --eps "8./255." --run_nr 1  --wanted_samples 2000 --take_inputimage_off 
python -u extract_characteristics.py --net cif10 --attack pgd  --detector LID  --eps "8./255." --run_nr 1  --wanted_samples 2000 --take_inputimage_off 
python -u extract_characteristics.py --net cif10 --attack cw   --detector LID  --eps "8./255." --run_nr 1  --wanted_samples 2000 --take_inputimage_off 


python -u detect_adversarials.py --net cif10 --attack fgsm --detector LID --wanted_samples 2000 --clf LR --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"
python -u detect_adversarials.py --net cif10 --attack bim  --detector LID --wanted_samples 2000 --clf LR --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"
python -u detect_adversarials.py --net cif10 --attack pgd  --detector LID --wanted_samples 2000 --clf LR --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"
python -u detect_adversarials.py --net cif10 --attack cw   --detector LID --wanted_samples 2000 --clf LR --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"

python -u detect_adversarials.py --net cif10 --attack fgsm --detector LI --wanted_samples 2000 --clf RF --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"
python -u detect_adversarials.py --net cif10 --attack bim  --detector LI --wanted_samples 2000 --clf RF --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"
python -u detect_adversarials.py --net cif10 --attack pgd  --detector LI --wanted_samples 2000 --clf RF --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"
python -u detect_adversarials.py --net cif10 --attack cw   --detector LID --wanted_samples 2000 --clf RF --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"



python -u extract_characteristics.py --net cif10 --attack fgsm --detector LIDNOISE  --eps "8./255." --run_nr 1  --wanted_samples 2000 --take_inputimage_off 
python -u extract_characteristics.py --net cif10 --attack bim --detector LIDNOISE  --eps "8./255." --run_nr 1  --wanted_samples 2000 --take_inputimage_off 
python -u extract_characteristics.py --net cif10 --attack pgd --detector LIDNOISE  --eps "8./255." --run_nr 1  --wanted_samples 2000 --take_inputimage_off 
python -u extract_characteristics.py --net cif10 --attack cw --detector LIDNOISE  --eps "8./255." --run_nr 1  --wanted_samples 2000 --take_inputimage_off 


python -u detect_adversarials.py --net cif10 --attack fgsm --detector LIDNOISE --wanted_samples 2000 --clf LR --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"
python -u detect_adversarials.py --net cif10 --attack bim  --detector LIDNOISE --wanted_samples 2000 --clf LR --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"
python -u detect_adversarials.py --net cif10 --attack pgd  --detector LIDNOISE --wanted_samples 2000 --clf LR --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"
python -u detect_adversarials.py --net cif10 --attack cw   --detector LIDNOISE --wanted_samples 2000 --clf LR --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"


python -u detect_adversarials.py --net cif10 --attack fgsm --detector LIDNOISE --wanted_samples 2000 --clf RF --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"
python -u detect_adversarials.py --net cif10 --attack bim  --detector LIDNOISE --wanted_samples 2000 --clf RF --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"
python -u detect_adversarials.py --net cif10 --attack pgd  --detector LIDNOISE --wanted_samples 2000 --clf RF --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"
python -u detect_adversarials.py --net cif10 --attack cw   --detector LIDNOISE --wanted_samples 2000 --clf RF --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"







python -u detect_adversarials.py --net cif10 --attack pgd  --detector LIDNOISE --wanted_samples 2000 --clf RF --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0" --tuning gridsearch



python -u extract_characteristics.py --net cif10 --attack pgd --detector LIDNOISE  --eps "8./255." --run_nr 15  --wanted_samples 8000 --take_inputimage_off 


python -u detect_adversarials.py --net cif10 --attack pgd  --detector LIDNOISE --wanted_samples 8000 --clf RF --eps "8./255."  --num_classes 10  --run_nr 15  --pca_features "0"  --tuning gridsearch

python -u detect_adversarials.py --net cif10 --attack pgd  --detector LIDNOISE --wanted_samples 8000 --clf LR --eps "8./255."  --num_classes 10  --run_nr 15  --pca_features "0" 





python -u extract_characteristics.py --net cif10 --attack pgd --detector FFTmultiLIDMFS  --eps "8./255." --run_nr 1  --wanted_samples 2000 --take_inputimage_off 
python -u extract_characteristics.py --net cif10 --attack pgd --detector FFTmultiLIDPFS  --eps "8./255." --run_nr 1  --wanted_samples 2000 --take_inputimage_off 


python -u detect_adversarials.py --net cif10 --attack pgd  --detector FFTmultiLIDMFS --wanted_samples 2000 --clf RF --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"
python -u detect_adversarials.py --net cif10 --attack pgd  --detector FFTmultiLIDPFS --wanted_samples 2000 --clf RF --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"

python -u detect_adversarials.py --net cif10 --attack pgd  --detector FFTmultiLIDMFS --wanted_samples 2000 --clf LR --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"
python -u detect_adversarials.py --net cif10 --attack pgd  --detector FFTmultiLIDPFS --wanted_samples 2000 --clf LR --eps "8./255."  --num_classes 10  --run_nr 1  --pca_features "0"