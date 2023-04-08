TEXT=/home/blherre4/VIVY/VIVYNet/data/tokens
fairseq-preprocess --source-lang x --only-source \
    --trainpref $TEXT/data \
    --destdir data/tokens/final \
    --workers 20