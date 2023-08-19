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
echo "USER_DIR:  ${USER_DIR}"
echo "TASK:  ${TASK}"
echo "CRITERION:  ${CRITERION}"
echo "ARCH:  ${ARCH}"
echo "SEED:  ${SEED}"
echo "PT_ENC:  ${PT_ENC}"
echo "FREEZE_ENC:  ${FREEZE_ENC}"
echo "PT_DEC:  ${PT_DEC}"
echo "FREEZE_DEC:  ${FREEZE_DEC}"
echo "EVT_VOC_SIZE:  ${EVT_VOC_SIZE}"
echo "TRK_VOC_SIZE:  ${TRK_VOC_SIZE}"
echo "DUR_VOC_SIZE:  ${DUR_VOC_SIZE}"
echo "INS_VOC_SIZE:  ${INS_VOC_SIZE}"
echo "MAX_REL_POS:  ${MAX_REL_POS}"
echo "MAX_MEA_POS:  ${MAX_MEA_POS}"
echo "TOKENS_PER_SAMPLE:  ${TOKENS_PER_SAMPLE}"
echo "SHORTEN:  ${SHORTEN}"
echo "SHORTEN_DATA_SPLIT_LIST:  ${SHORTEN_DATA_SPLIT_LIST}"
echo "SAMPLE_BREAK_MODE:  ${SAMPLE_BREAK_MODE}"
echo "RATIO:  ${RATIO}"
echo "SAMPLE_OVERLAP_RATE:  ${SAMPLE_OVERLAP_RATE}"
echo "PERM_INV:  ${PERM_INV}"
echo "OPTIMIZER:  ${OPTIMIZER}"
echo "ADAM_BETAS:  ${ADAM_BETAS}"
echo "ADAM_EPS:  ${ADAM_EPS}"
echo "CLIP_NORM:  ${CLIP_NORM}"
echo "WEIGHT_DECAY:  ${WEIGHT_DECAY}"
echo "BATCH_SIZE:  ${BATCH_SIZE}"
echo "LR:  ${LR}"
echo "LR_SCHEDULER:  ${LR_SCHEDULER}"
echo "SAVE_DIR:  ${SAVE_DIR}"
echo "TENSORBOARD_LOGDIR:  ${TENSORBOARD_LOGDIR}"
echo "LOG_FORMAT:  ${LOG_FORMAT}"
echo "LOG_INTERVAL:  ${LOG_INTERVAL}"
echo

# Log into wandb
wandb login $WANDB_API_KEY

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
  --pt_enc $PT_ENC \
  --freeze_enc $FREEZE_ENC \
  --pt_dec $PT_DEC \
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