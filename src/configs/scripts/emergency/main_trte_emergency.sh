net="cif10"
run="50"
att="pgd"
detector="LIDNOISE"
eps="8./255."

clf="LR RF"

NRWANTEDSAMPLES="0"

WANTEDSAMPLES_TR="44000"
WANTEDSAMPLES_TE="8600"



#python -u generate_clean_data_trte.py --net "$net"  --run_nr "$run" --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"  --wanted_samples_te "$WANTEDSAMPLES_TE" 
#python -u attacks_trte.py --net "$net" --attack "$att"  --batch_size 500 --eps "$eps"  --run_nr "$run"  --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE"


python -u extract_characteristics_trte.py --net "$net" --attack "$att" --detector "$det"  --eps "$eps" --run_nr "$run" --wanted_samples "$NRWANTEDSAMPLES" --wanted_samples_tr "$WANTEDSAMPLES_TR"   --wanted_samples_te "$WANTEDSAMPLES_TE" --take_inputimage_off