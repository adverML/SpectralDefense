
# execute table1.sh first

# layer settings from [Yang et al., 2021]  

python  extract_characteristics.py --load_json LIDLessFeatures/cif10_fgsm_2000.json
python  extract_characteristics.py --load_json LIDLessFeatures/cif10_bim_2000.json
python  extract_characteristics.py --load_json LIDLessFeatures/cif10_pgd_2000.json
python  extract_characteristics.py --load_json LIDLessFeatures/cif10_cw_2000.json


python  extract_characteristics.py --load_json multiLIDLessFeatures/cif10_fgsm_2000.json
python  extract_characteristics.py --load_json multiLIDLessFeatures/cif10_bim_2000.json
python  extract_characteristics.py --load_json multiLIDLessFeatures/cif10_pgd_2000.json
python  extract_characteristics.py --load_json multiLIDLessFeatures/cif10_cw_2000.json



python detect_adversarials.py --load_json LIDLessFeatures/LR_cif10_fgsm_2000.json
python detect_adversarials.py --load_json LIDLessFeatures/LR_cif10_bim_2000.json
python detect_adversarials.py --load_json LIDLessFeatures/LR_cif10_pgd_2000.json
python detect_adversarials.py --load_json LIDLessFeatures/LR_cif10_cw_2000.json



python detect_adversarials.py --load_json multiLIDLessFeatures/LR_cif10_fgsm_2000.json
python detect_adversarials.py --load_json multiLIDLessFeatures/LR_cif10_bim_2000.json
python detect_adversarials.py --load_json multiLIDLessFeatures/LR_cif10_pgd_2000.json
python detect_adversarials.py --load_json multiLIDLessFeatures/LR_cif10_cw_2000.json


python detect_adversarials.py --load_json multiLIDLessFeatures/RF_cif10_fgsm_2000.json
python detect_adversarials.py --load_json multiLIDLessFeatures/RF_cif10_bim_2000.json
python detect_adversarials.py --load_json multiLIDLessFeatures/RF_cif10_pgd_2000.json
python detect_adversarials.py --load_json multiLIDLessFeatures/RF_cif10_cw_2000.json