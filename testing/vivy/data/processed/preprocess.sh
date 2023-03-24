fairseq-preprocess \
    --source-lang feat \
    --target-lang labl \
    --trainpref train \
    --validpref valid \
    --destdir bin/ \
    --dataset-impl raw \
    --workers 20