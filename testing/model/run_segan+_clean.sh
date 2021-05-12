#!/bin/bash

CKPT_PATH="ckpt_segan+"

# please specify the path to your G model checkpoint
# as in weights_G-EOE_<iter>.ckpt
G_PRETRAINED_CKPT="weights.ckpt"

# please specify the path to your folder containing
# noisy test files, each wav in there will be processed
TEST_FILES_PATH="dataset/clean/"

# please specify the output folder where cleaned files
# will be saved
SAVE_PATH="dataset/clean/"

# python -u clean.py --g_pretrained_ckpt $CKPT_PATH/$G_PRETRAINED_CKPT \
#	--test_files $TEST_FILES_PATH --cfg_file $CKPT_PATH/train.opts \
#	--synthesis_path $SAVE_PATH --soundfile


python -u clean.py \
    --g_pretrained_ckpt ./ckpt_segan+/weights.ckpt \
	--test_files ./dataset/noise \
	--cfg_file ./ckpt_segan+/train.opts \
	--synthesis_path ./dataset/clean/ \
#	 --soundfile
