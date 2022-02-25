#!/bin/bash

# define python command including script and all parameters to execute


# NOTE: the "-u" flag is essential as we want the output of the python script directly and not the moment the buffer it full
#-----------------------------------------------------------------------------------------------------------------------------------

python generate_clean_data_imagenet.py --net imagenet32 --num_classes 25
python generate_clean_data_imagenet.py --net imagenet32 --num_classes 50
python generate_clean_data_imagenet.py --net imagenet32 --num_classes 75
python generate_clean_data_imagenet.py --net imagenet32 --num_classes 250

#-----------------------------------------------------------------------------------------------------------------------------------
echo "finished experiment"

exit 0
#-----------------------------------------------------------------------------------------------------------------------------------