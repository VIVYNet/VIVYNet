fairseq-train ../data/final \
  --user-dir ./ \
  --task text2music \
  --criterion nll_loss \
  --arch vivy_train \
  --optimizer adam \
  --batch-size 1 \
  --lr 0.001 \
  --max-tokens 1024 \
  --shorten_method none \
  --shorten_data_split_list '' \
  --tokens_per_sample 1024 \
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
  # --evt_voc_size 436 \
  # --trk_voc_size 44 \
  # --dur_voc_size 36 \
  # --ins_voc_size 84 \
  # --max_rel_pos  134 \
  # --max_mea_pos  2810 \
