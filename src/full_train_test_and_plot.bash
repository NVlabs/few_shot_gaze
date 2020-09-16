#!/bin/bash

###########################
# Necessary Configurations

# We skip DT-ED training by default, such that the pre-trained weights
# can be used as-is. Please make sure that you have followed the README.md
# instructions to acquire these weights.
#
# Note: This is different to the `--skip-training` argument to
#       `1_train_dt_ed.py` in that it skips the script completely.
#
# Set to 0 to perform DT-ED training and inference for the HDF output.
SKIP_DTED_TRAINING=0

# NOTE: please make sure to update the two paths below as necessary.
MPIIGAZE_FILE="../preprocess/outputs/MPIIGaze.h5
GAZECAPTURE_FILE="../preprocess/outputs/GazeCapture.h5

# This batch size should fit a 11GB single GPU
# The original training used 8x Tesla V100 GPUs.
BATCH_SIZE=64

# Set the experiment output directory.
# NOTE: make sure to change this if you do not intend to over-write
#       previous outputs.
OUTPUT_DIR="outputs_of_full_train_test_and_plot"


if [[ $SKIP_DTED_TRAINING -eq 0 ]]
then
	############################
	# 1. Perform DT-ED training
	#
	#    The original setup used here was with:
	#    > Batch size:  1536
	#    > # of epochs: 20
	#    > # of GPUs:   8
	#    > GPU model:   Tesla V100 (32GB)
	#    > Mixed precision training with apex -O1

    TRAIN_CMD=""
	TRAIN_CMD="1_train_dt_ed.py \
		--mpiigaze-file ${MPIIGAZE_FILE} \
		--gazecapture-file ${GAZECAPTURE_FILE} \
		\
		--num-training-epochs 20 \
		--batch-size $BATCH_SIZE \
		--eval-batch-size 1024 \
		\
		--normalize-3d-codes \
		--embedding-consistency-loss-type angular \
		--backprop-gaze-to-encoder \
		\
		--num-data-loaders 16 \
		\
		--save-image-samples 20 \
		--use-tensorboard \
		--save-path ${OUTPUT_DIR} \
        "
    eval "python3 -m torch.distributed.launch --nproc_per_node=8 $TRAIN_CMD --distributed; "
    eval "python3 $TRAIN_CMD --skip-training --generate-predictions; "

		#####################################################################################
		# NOTE: when adding the lines below, make sure to use the backslash ( \ ) correctly,
		#       such that the full command is correctly constructed and registered.

		# Use (append to above) the line below if wanting to use pre-trained weights, and skip training
		# DO NOT JUST UNCOMMENT IT, IT WILL HAVE NO EFFECT DUE TO BASH PARSING
		# --skip-training \

		# Use (append to above) the line below if wanting to use mixed-precision training (as done in the paper)
		# DO NOT JUST UNCOMMENT IT, IT WILL HAVE NO EFFECT DUE TO BASH PARSING
		# --use-apex \

		# To change the number of GPUs used while training DT-ED change --nproc_per_node=8 in
		# "python3 -m torch.distributed.launch --nproc_per_node=8 $TRAIN_CMD --distributed; "

fi

###########################
# 2. Perform Meta Learning
#
#    This step processes 6 experiments at a time because a single experiment
#    does not make use of the GPU capacity sufficiently well.
#
#    Please note, that you need output HDF files from the previous step to
#    proceed to the next step. These HDF files are provided to you by default
#    in this specific example pipeline.
#
#    In this example script, we use pre-trained MAML weights that we provided on
#    20th January 2020. This can be done by providing the `--skip-training`
#    command line argument. Note that the full testing procedure is still very
#    time consuming, as 1000-step fine-tuning must occur for each participant
#    in GazeCapture and MPIIGaze.

ML_COMMON=" --disable-tqdm --output-dir ./"

python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR}  1 &
python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR}  2 &
python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR}  3 &
python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR}  4 &
python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR}  5 &
python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR}  6 &
wait

python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR}  7 &
python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR}  8 &
python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR}  9 &
python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR} 10 &
python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR} 11 &
python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR} 12 &
wait

python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR} 13 &
python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR} 14 &
python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR} 15 &
python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR} 16 &
python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR} 17 &
python3 2_meta_learning.py ${ML_COMMON} ${OUTPUT_DIR} 18 &
wait


####################################################################
# 3. Collect all of the individual meta-learning experiment results

python3 3_combine_maml_results.py ${OUTPUT_DIR}
