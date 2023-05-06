fairseq-train data/processed/bin \
  --user-dir ~/VIVY/VIVYNet/testing/vivy \
  --task bert_cola_train \
  --arch bert_train \
  --optimizer adam \
  --lr 0.001 \
  --max-tokens 1000 