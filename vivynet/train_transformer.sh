#!/bin/bash

#SBATCH -p gpu                              ## Partition
#SBATCH -q wildfire                         ## QOS
#SBATCH -c 1                                ## Number of Cores
#SBATCH --time=1440                         ## 1 day of compute
#SBATCH --mem 48G                           ## 48 GB of RAM
#SBATCH --mail-type=ALL                     ## notify for any job state change

#
#   VARIABLE EXTRACT
#   region
#

# Source personal information and config file to run
if [[ $0 == *"slurm"* ]]; then
  echo "Using Slurm..."
  echo
  source ./configs/$1.sh
else
  echo "Not using Slurm..."
  echo
  source ./configs/test.sh       # <<< CHANGE CONFIG NAME IF NOT USING SLURM
fi
source ./personal.sh


# endregion



#
#   PRE RUN INITIALIZATION
#   region
#

if [[ $0 == *"slurm"* ]]; then
  echo "Using Slurm for prepping..."
  echo

  echo "Purging modules"
  module purge

  echo "Loading Python 3 from Anaconda Module"
  module load anaconda/py3

  echo "Loading VIVYNET Conda Environment"
  source activate $CONDA_ENVIRONMENT

  echo "Showing GPU Details"
  nvidia-smi -L
  nvidia-smi
fi

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
echo "DEC_EMBED_DIM:  ${DEC_EMBED_DIM}"
echo "DEC_NUM_ATTN_HEADS:  ${DEC_NUM_ATTN_HEADS}"
echo "DEC_NUM_LAYERS:  ${DEC_NUM_LAYERS}"
echo "DEC_DROPOUT:  ${DEC_DROPOUT}"
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
echo "OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "SAVE_DIR:  ${SAVE_DIR}"
echo "TENSORBOARD_LOGDIR:  ${TENSORBOARD_LOGDIR}"
echo "LOG_FORMAT:  ${LOG_FORMAT}"
echo "LOG_INTERVAL:  ${LOG_INTERVAL}"
echo

# Setup wandb
wandb login $WANDB_API_KEY
export WANDB_NAME="$RUN_NAME - $VARIANT"
export WANDB_DIR=$OUTPUT_DIR

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
  --dec_embed_dim $DEC_EMBED_DIM \
  --dec_num_attention_heads $DEC_NUM_ATTN_HEADS \
  --dec_num_layers $DEC_NUM_LAYERS \
  --dec_dropout $DEC_DROPOUT \
  --lr-scheduler $LR_SCHEDULER \
  --output_dir $OUTPUT_DIR \
  --save-dir $SAVE_DIR \
  --tensorboard-logdir $TENSORBOARD_LOGDIR \
  --no-epoch-checkpoints \
  --log-format $LOG_FORMAT \
  --log-interval $LOG_INTERVAL \
  --wandb-project $WANDB_PROJECT

# endregion


# Finished
echo
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Finished"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"