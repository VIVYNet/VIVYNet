TEXT=../data/tokens
fairseq-preprocess --source-lang x --only-source \
    --trainpref $TEXT/data \
    --destdir $TEXT/final \
    --workers 20