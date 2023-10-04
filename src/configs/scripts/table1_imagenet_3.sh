run_nr=3

######### LID
python  extract_characteristics.py --load_json LID/IMAGENET/wrn502_fgsm_2000.json  --run_nr $run_nr
python  extract_characteristics.py --load_json LID/IMAGENET/wrn502_bim_2000.json  --run_nr $run_nr
python  extract_characteristics.py --load_json LID/IMAGENET/wrn502_pgd_2000.json  --run_nr $run_nr
python  extract_characteristics.py --load_json LID/IMAGENET/wrn502_aa_2000.json  --run_nr $run_nr
python  extract_characteristics.py --load_json LID/IMAGENET/wrn502_df_2000.json  --run_nr $run_nr
python  extract_characteristics.py --load_json LID/IMAGENET/wrn502_cw_2000.json  --run_nr $run_nr


######### multiLID 
python  extract_characteristics.py --load_json multiLID/IMAGENET/wrn502_fgsm_2000.json  --run_nr $run_nr
python  extract_characteristics.py --load_json multiLID/IMAGENET/wrn502_bim_2000.json  --run_nr $run_nr
python  extract_characteristics.py --load_json multiLID/IMAGENET/wrn502_pgd_2000.json  --run_nr $run_nr
python  extract_characteristics.py --load_json multiLID/IMAGENET/wrn502_aa_2000.json  --run_nr $run_nr
python  extract_characteristics.py --load_json multiLID/IMAGENET/wrn502_df_2000.json  --run_nr $run_nr
python  extract_characteristics.py --load_json multiLID/IMAGENET/wrn502_cw_2000.json  --run_nr $run_nr
