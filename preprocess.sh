TEXT=/home/blherre4/VIVY/VIVYNet/data/tokens
fairseq-preprocess --source-lang x --target-lang y \
    --trainpref $TEXT/data \
    --destdir data/tokens/final \
    --workers 20