VARIANT="test"

mkdir "./results/$VARIANT"
mkdir "./results/$VARIANT/slurm"

fairseq-train ../data/final \
  --user-dir ./ \
  --task text2music_ve \
  --criterion nll_loss_ve \
  --arch vivy_train_ve \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --adam-eps 1e-6 \
  --clip-norm 0.0 \
  --weight-decay 0.01 \
  --batch-size 1 \
  --lr 0.00001 \
  --max-tokens 8192 \
  --shorten_method none \
  --shorten_data_split_list '' \
  --tokens_per_sample 4096 \
  --seed 1998 \
  --sample_break_mode complete_doc \
  --ratio 4 \
  --sample_overlap_rate 4 \
  --perm_inv 2 \
  --evt_voc_size 1125 \
  --trk_voc_size 44 \
  --dur_voc_size 36 \
  --ins_voc_size 133 \
  --max_rel_pos  198 \
  --max_mea_pos  5360 \
  --freeze_enc 1 \
  --freeze_dec 1 \
  --lr-scheduler polynomial_decay \
  --save-dir ./results/$VARIANT/ckpt \
  --tensorboard-logdir ./results/$VARIANT/logs \
  --no-epoch-checkpoints
