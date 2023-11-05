# Variant name definition
VIVY_VARIANT=$(basename "${BASH_SOURCE[0]}" .sh)

# Fairseq model specification
VIVY_USER_DIR=./
VIVY_TASK=text2music
VIVY_CRITERION=nll_loss
VIVY_ARCH=vivy_transformer
VIVY_SEED=1998

# Encoder model specifications
VIVY_ENC=BERT_english
VIVY_FREEZE_ENC=0
VIVY_PT_ENC=1

# Decoder model specifications
VIVY_DEC=SymphonyNet_Vanilla
VIVY_FREEZE_DEC=0
VIVY_PT_DEC=1
VIVY_EVT_VOC_SIZE=1125
VIVY_TRK_VOC_SIZE=44
VIVY_DUR_VOC_SIZE=36
VIVY_INS_VOC_SIZE=133
VIVY_MAX_REL_POS=198
VIVY_MAX_MEA_POS=5360
VIVY_DEC_EMBED_DIM=512
VIVY_DEC_NUM_ATTN_HEADS=16
VIVY_DEC_NUM_LAYERS=12
VIVY_DEC_DROPOUT=0.1

# Latent model behavior specifications
VIVY_LATENT=transformer_encoder
VIVY_LATENT_NUM_LAYERS=14
VIVY_LATENT_NUM_ATTN_HEADS=32
VIVY_LATENT_EMBED_DIM=128
VIVY_LATENT_DROPOUT=0.3
VIVY_LATENT_INPUT_DIM=768
VIVY_LATENT_OUTPUT_DIM=512
VIVY_LATENT_SELF_ATTENTION_TYPE=linear

# Data processing options
VIVY_TOKENS_PER_SAMPLE=4096
VIVY_SHORTEN=none
VIVY_SHORTEN_DATA_SPLIT_LIST=""
VIVY_SAMPLE_BREAK_MODE=complete_doc
VIVY_RATIO=4
VIVY_SAMPLE_OVERLAP_RATE=4
VIVY_PERM_INV=2

# Optimizer options
VIVY_OPTIMIZER=adam
VIVY_ADAM_BETAS="(0.9, 0.98)"
VIVY_ADAM_EPS=1e-6
VIVY_CLIP_NORM=0.0
VIVY_WEIGHT_DECAY=0.01

# Training options
VIVY_BATCH_SIZE=2
VIVY_LR=0.001
VIVY_LR_SCHEDULER=polynomial_decay

# Logging and checkpoing saving
VIVY_OUTPUT_DIR=/scratch/blherre4/vivynet/results/$VIVY_VARIANT
VIVY_SAVE_DIR=/scratch/blherre4/vivynet/results/$VIVY_VARIANT/ckpt
VIVY_TENSORBOARD_LOGDIR=/scratch/blherre4/vivynet/results/$VIVY_VARIANT/logs
VIVY_LOG_FORMAT=simple
VIVY_LOG_INTERVAL=10