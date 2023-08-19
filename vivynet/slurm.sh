#!/bin/sh

# Source the file config file to run
source ./configs/default.sh

# Print config settings
echo -e "\n\n\nVARIANT:  ${VARIANT}"
echo "============================================"
echo "\$2 > USER_DIR:  ${USER_DIR}"
echo "\$3 > TASK:  ${TASK}"
echo "\$4 > CRITERION:  ${CRITERION}"
echo "\$5 > ARCH:  ${ARCH}"
echo "\$6 > SEED:  ${SEED}"
echo "\$7 > FREEZE_ENC:  ${FREEZE_ENC}"
echo "\$8 > FREEZE_DEC:  ${FREEZE_DEC}"
echo "\$9 > EVT_VOC_SIZE:  ${EVT_VOC_SIZE}"
echo "\$10 > TRK_VOC_SIZE:  ${TRK_VOC_SIZE}"
echo "\$11 > DUR_VOC_SIZE:  ${DUR_VOC_SIZE}"
echo "\$12 > INS_VOC_SIZE:  ${INS_VOC_SIZE}"
echo "\$13 > MAX_REL_POS:  ${MAX_REL_POS}"
echo "\$14 > MAX_MEA_POS:  ${MAX_MEA_POS}"
echo "\$15 > TOKENS_PER_SAMPLE:  ${TOKENS_PER_SAMPLE}"
echo "\$16 > SHORTEN:  ${SHORTEN}"
echo "\$17 > SHORTEN_DATA_SPLIT_LIST:  ${SHORTEN_DATA_SPLIT_LIST}"
echo "\$18 > SAMPLE_BREAK_MODE:  ${SAMPLE_BREAK_MODE}"
echo "\$19 > RATIO:  ${RATIO}"
echo "\$20 > SAMPLE_OVERLAP_RATE:  ${SAMPLE_OVERLAP_RATE}"
echo "\$21 > PERM_INV:  ${PERM_INV}"
echo "\$22 > OPTIMIZER:  ${OPTIMIZER}"
echo "\$23 > ADAM_BETAS:  ${ADAM_BETAS}"
echo "\$24 > ADAM_EPS:  ${ADAM_EPS}"
echo "\$25 > CLIP_NORM:  ${CLIP_NORM}"
echo "\$26 > WEIGHT_DECAY:  ${WEIGHT_DECAY}"
echo "\$27 > BATCH_SIZE:  ${BATCH_SIZE}"
echo "\$28 > LR:  ${LR}"
echo "\$29 > LR_SCHEDULER:  ${LR_SCHEDULER}"
echo "\$30 > SAVE_DIR:  ${SAVE_DIR}"
echo "\$31 > TENSORBOARD_LOGDIR:  ${TENSORBOARD_LOGDIR}"
echo "\$32 > LOG_FORMAT:  ${LOG_FORMAT}"
echo "\$33 > LOG_INTERVAL:  ${LOG_INTERVAL}"
echo

# Checkpoint Stop
read -p "Do you want to proceed? (Y/n): " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    echo "Proceeding..."
    echo
    # Place the code to be executed on confirmation here
else
    echo "Dispatch canceled"
    exit 1
fi

# Make directories
echo "Making directories..."
mkdir ./results/$VARIANT
mkdir ./results/$VARIANT/slurm
echo "Directories made"
echo


# Start the slurm run process
echo "Dispatching train run..."
sbatch \
  --job-name="VIVYNET Training - ${VARIANT}" \
  --output="./results/${VARIANT}/slurm/out_%j.txt" \
  --error="./results/${VARIANT}/slurm/err_%j.txt"  \
  ./train_transformer_slurm.sh \