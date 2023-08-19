#!/bin/bash

#SBATCH -p gpu                              ## Partition
#SBATCH -q wildfire                         ## QOS
#SBATCH -c 1                                ## Number of Cores
#SBATCH --time=1440                         ## 1 day of compute
#SBATCH --gres=gpu:V100:1                   ## 1 of V100 GPU
#SBATCH --mem 32G                           ## 32 GB of RAM
#SBATCH --mail-type=ALL                     ## notify for any job state change
#SBATCH --mail-user=blherre4@asu.edu        ## notify email

#
#   VARIABLE EXTRACT
#   region
#

# Source the file config file to run
source ./configs/default.sh

# # Extract from given argument values
# VARIANT=$1
# USER_DIR=$2
# TASK=$3
# CRITERION=$4
# ARCH=$5
# SEED=$6
# FREEZE_ENC=$7
# FREEZE_DEC=$8
# EVT_VOC_SIZE=$9
# TRK_VOC_SIZE=$10
# DUR_VOC_SIZE=$11
# INS_VOC_SIZE=$12
# MAX_REL_POS=$13
# MAX_MEA_POS=$14
# TOKENS_PER_SAMPLE=$15
# SHORTEN=$16
# SHORTEN_DATA_SPLIT_LIST=$17
# SAMPLE_BREAK_MODE=$18
# RATIO=$19
# SAMPLE_OVERLAP_RATE=$20
# PERM_INV=$21
# OPTIMIZER=$22
# ADAM_BETAS=$23
# ADAM_EPS=$24
# CLIP_NORM=$25
# WEIGHT_DECAY=$26
# BATCH_SIZE=$27
# LR=$28
# LR_SCHEDULER=$29
# SAVE_DIR=$30
# TENSORBOARD_LOGDIR=$31
# LOG_FORMAT=$32
# LOG_INTERVAL=$33

# endregion



#
#   PRE RUN INITIALIZATION
#   region
#

echo "Purging modules"
module purge

echo "Loading Python 3 from Anaconda Module"
module load anaconda/py3

echo "Loading VIVYNET Conda Environment"
source activate vivyenv #-pytorch-1.13.1

echo "Showing GPU Details"
nvidia-smi -L
nvidia-smi

#   endregion



#
#   MODEL TRAIN RUN
#   region
#

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

# Run model train
fairseq-train ../data/final \
  --user-dir $USER_DIR \
  --task $TASK \
  --criterion $CRITERION \
  --arch $ARCH \
  --optimizer $OPTIMIZER \
  --adam-betas "$ADAM_BETAS" \
  --adam-eps $ADAM_EPS \
  --clip-norm $CLIP_NORM \
  --weight-decay $WEIGHT_DECAY \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --shorten_method $SHORTEN \
  --shorten_data_split_list "$SHORTEN_DATA_SPLIT_LIST" \
  --tokens_per_sample $TOKENS_PER_SAMPLE \
  --seed $SEED \
  --sample_break_mode $SAMPLE_BREAK_MODE \
  --ratio $RATIO \
  --sample_overlap_rate $SAMPLE_OVERLAP_RATE \
  --perm_inv $PERM_INV \
  --evt_voc_size $EVT_VOC_SIZE \
  --trk_voc_size $TRK_VOC_SIZE \
  --dur_voc_size $DUR_VOC_SIZE \
  --ins_voc_size $INS_VOC_SIZE \
  --max_rel_pos $MAX_REL_POS \
  --max_mea_pos $MAX_MEA_POS \
  --freeze_enc $FREEZE_ENC \
  --freeze_dec $FREEZE_DEC \
  --lr-scheduler $LR_SCHEDULER \
  --save-dir $SAVE_DIR \
  --tensorboard-logdir $TENSORBOARD_LOGDIR \
  --no-epoch-checkpoints \
  --log-format $LOG_FORMAT \
  --log-interval $LOG_INTERVAL

# endregion


# Finished
echo
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Finished"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"