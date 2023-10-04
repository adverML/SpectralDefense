#python generate_clean_data_trte.py --load_json trte_cif10.json

python extract_characteristics_trte.py --load_json multiLID/cif10_pgd_trte.json

python  detect_adversarials_trte.py --load_json multiLID/RF_cif10_pgd_trte.json
python  detect_adversarials_trte.py --load_json multiLID/LR_cif10_pgd_trte.json
