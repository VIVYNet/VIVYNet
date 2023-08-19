# Source the wandb api key file and config file to run
source ./configs/test.sh
source ./personal.sh

# Make directories
mkdir ./results/$VARIANT

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
  --lr-scheduler $LR_SCHEDULER \
  --output_dir $OUTPUT_DIR \
  --save-dir $SAVE_DIR \
  --tensorboard-logdir $TENSORBOARD_LOGDIR \
  --no-epoch-checkpoints \
  --log-format $LOG_FORMAT \
  --log-interval $LOG_INTERVAL \
  --wandb-project $WANDB_PROJECT