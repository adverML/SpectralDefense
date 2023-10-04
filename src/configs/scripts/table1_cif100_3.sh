run_nr=3

######### LID
python  extract_characteristics.py --load_json LID/CIFAR100/wrn2810_fgsm_2000.json   --run_nr $run_nr
python  extract_characteristics.py --load_json LID/CIFAR100/wrn2810_bim_2000.json   --run_nr $run_nr
python  extract_characteristics.py --load_json LID/CIFAR100/wrn2810_pgd_2000.json   --run_nr $run_nr
python  extract_characteristics.py --load_json LID/CIFAR100/wrn2810_aa_2000.json   --run_nr $run_nr
python  extract_characteristics.py --load_json LID/CIFAR100/wrn2810_df_2000.json   --run_nr $run_nr
python  extract_characteristics.py --load_json LID/CIFAR100/wrn2810_cw_2000.json   --run_nr $run_nr

python  extract_characteristics.py --load_json LID/CIFAR100/vgg16_fgsm_2000.json  --run_nr $run_nr
python  extract_characteristics.py --load_json LID/CIFAR100/vgg16_bim_2000.json  --run_nr $run_nr
python  extract_characteristics.py --load_json LID/CIFAR100/vgg16_pgd_2000.json  --run_nr $run_nr
python  extract_characteristics.py --load_json LID/CIFAR100/vgg16_aa_2000.json  --run_nr $run_nr
python  extract_characteristics.py --load_json LID/CIFAR100/vgg16_df_2000.json  --run_nr $run_nr
python  extract_characteristics.py --load_json LID/CIFAR100/vgg16_cw_2000.json  --run_nr $run_nr


######### multiLID 
python  extract_characteristics.py --load_json multiLID/CIFAR100/wrn2810_fgsm_2000.json   --run_nr $run_nr
python  extract_characteristics.py --load_json multiLID/CIFAR100/wrn2810_bim_2000.json   --run_nr $run_nr
python  extract_characteristics.py --load_json multiLID/CIFAR100/wrn2810_pgd_2000.json   --run_nr $run_nr
python  extract_characteristics.py --load_json multiLID/CIFAR100/wrn2810_aa_2000.json   --run_nr $run_nr
python  extract_characteristics.py --load_json multiLID/CIFAR100/wrn2810_df_2000.json   --run_nr $run_nr
python  extract_characteristics.py --load_json multiLID/CIFAR100/wrn2810_cw_2000.json   --run_nr $run_nr

python  extract_characteristics.py --load_json multiLID/CIFAR100/vgg16_fgsm_2000.json   --run_nr $run_nr
python  extract_characteristics.py --load_json multiLID/CIFAR100/vgg16_bim_2000.json   --run_nr $run_nr
python  extract_characteristics.py --load_json multiLID/CIFAR100/vgg16_pgd_2000.json   --run_nr $run_nr
python  extract_characteristics.py --load_json multiLID/CIFAR100/vgg16_aa_2000.json   --run_nr $run_nr
python  extract_characteristics.py --load_json multiLID/CIFAR100/vgg16_df_2000.json   --run_nr $run_nr
python  extract_characteristics.py --load_json multiLID/CIFAR100/vgg16_cw_2000.json   --run_nr $run_nr
