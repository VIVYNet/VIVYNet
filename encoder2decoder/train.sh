fairseq-train /home/blherre4/VIVY/VIVYNet/data/final \
  --user-dir /home/blherre4/VIVY/VIVYNet/encoder2decoder \
  --task text2music \
  --criterion nll_loss \
  --arch vivy_train \
  --optimizer adam \
  --lr 0.001 \
  --max-tokens 1000 