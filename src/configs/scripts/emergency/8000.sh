python  extract_characteristics.py --load_json FFTmultiLIDMFS/cif10_pgd_8000.json
python  extract_characteristics.py --load_json FFTmultiLIDPFS/cif10_pgd_8000.json

python  detect_adversarials.py --load_json FFTmultiLIDMFS/RF_cif10_pgd_8000.json
python  detect_adversarials.py --load_json FFTmultiLIDMFS/LR_cif10_pgd_8000.json

python  detect_adversarials.py --load_json FFTmultiLIDPFS/RF_cif10_pgd_8000.json
python  detect_adversarials.py --load_json FFTmultiLIDPFS/LR_cif10_pgd_8000.json