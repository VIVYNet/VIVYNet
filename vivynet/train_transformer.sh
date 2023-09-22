#!/bin/bash

#SBATCH -p gpu                              ## Partition
#SBATCH -q wildfire                         ## QOS
#SBATCH -c 1                                ## Number of Cores
#SBATCH --time=4320                         ## 3 day of compute
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
  source ./configs/test3.sh       # <<< CHANGE CONFIG NAME IF NOT USING SLURM
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
else
  echo "Making directories..."
  mkdir $VIVY_OUTPUT_DIR
  echo "Directories made"
  echo
fi

#   endregion



#
#   MODEL TRAIN RUN
#   region
#

# Print config settings
echo -e "\n\n\nVARIANT:  ${VIVY_VARIANT}"
echo "============================================"
for var in $(compgen -v VIVY_); do
    stripped_var=${var#VIVY_}
    echo "$stripped_var: ${!var}"
done
echo

# Setup wandb
wandb login $WANDB_API_KEY
export WANDB_NAME="$RUN_NAME - $VIVY_VARIANT"
export WANDB_DIR=$VIVY_OUTPUT_DIR

# Run model train
fairseq-train ../data/final \
  --user-dir $VIVY_USER_DIR \
  --task $VIVY_TASK \
  --criterion $VIVY_CRITERION \
  --arch $VIVY_ARCH \
  --seed $VIVY_SEED \
  --enc $VIVY_ENC \
  --freeze_enc $VIVY_FREEZE_ENC \
  --pt_enc $VIVY_PT_ENC \
  --dec $VIVY_DEC \
  --freeze_dec $VIVY_FREEZE_DEC \
  --pt_dec $VIVY_PT_DEC \
  --evt_voc_size $VIVY_EVT_VOC_SIZE \
  --trk_voc_size $VIVY_TRK_VOC_SIZE \
  --dur_voc_size $VIVY_DUR_VOC_SIZE \
  --ins_voc_size $VIVY_INS_VOC_SIZE \
  --max_rel_pos $VIVY_MAX_REL_POS \
  --max_mea_pos $VIVY_MAX_MEA_POS \
  --dec_embed_dim $VIVY_DEC_EMBED_DIM \
  --dec_num_attention_heads $VIVY_DEC_NUM_ATTN_HEADS \
  --dec_num_layers $VIVY_DEC_NUM_LAYERS \
  --dec_dropout $VIVY_DEC_DROPOUT \
  --latent $VIVY_LATENT \
  --latent_num_layers ${VIVY_LATENT_NUM_LAYERS:=0} \
  --latent_num_attention_heads ${VIVY_LATENT_NUM_ATTN_HEADS:=0} \
  --latent_embed_dim ${VIVY_LATENT_EMBED_DIM:=0} \
  --latent_dropout ${VIVY_LATENT_DROPOUT:=0} \
  --latent_input_dim ${VIVY_LATENT_INPUT_DIM:=0} \
  --latent_hidden_dim ${VIVY_LATENT_HIDDEN_DIM:=0} \
  --latent_output_dim ${VIVY_LATENT_OUTPUT_DIM:=0} \
  --latent_hidden_layers ${VIVY_LATENT_HIDDEN_LAYERS:=0} \
  --latent_dropout_rate ${VIVY_LATENT_DROPOUT_RATE:=0} \
  --tokens_per_sample $VIVY_TOKENS_PER_SAMPLE \
  --shorten_method $VIVY_SHORTEN \
  --shorten_data_split_list "$VIVY_SHORTEN_DATA_SPLIT_LIST" \
  --sample_break_mode $VIVY_SAMPLE_BREAK_MODE \
  --ratio $VIVY_RATIO \
  --sample_overlap_rate $VIVY_SAMPLE_OVERLAP_RATE \
  --perm_inv $VIVY_PERM_INV \
  --optimizer $VIVY_OPTIMIZER \
  --adam-betas "$VIVY_ADAM_BETAS" \
  --adam-eps $VIVY_ADAM_EPS \
  --clip-norm $VIVY_CLIP_NORM \
  --weight-decay $VIVY_WEIGHT_DECAY \
  --batch-size $VIVY_BATCH_SIZE \
  --lr $VIVY_LR \
  --lr-scheduler $VIVY_LR_SCHEDULER \
  --save-dir $VIVY_SAVE_DIR \
  --tensorboard-logdir $VIVY_TENSORBOARD_LOGDIR \
  --no-epoch-checkpoints \
  --log-format $VIVY_LOG_FORMAT \
  --log-interval $VIVY_LOG_INTERVAL \
  --wandb-project $WANDB_PROJECT

# endregion


# Finished
echo
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Finished"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"