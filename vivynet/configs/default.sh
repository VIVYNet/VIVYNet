# WANDB API KEY
WANDB_API_KEY="7773ca50f1f4adc95e743e19c8cf1e7b864b8dba"

# Variant name definition
VARIANT="default"

# Fairseq model specification
USER_DIR=./
TASK=text2music
CRITERION=nll_loss
ARCH=vivy_transformer
SEED=1998

# Model behavior specifications
PT_ENC=0
FREEZE_ENC=1
PT_DEC=0
FREEZE_DEC=0

# Embedding layer information
EVT_VOC_SIZE=1125
TRK_VOC_SIZE=44
DUR_VOC_SIZE=36
INS_VOC_SIZE=133
MAX_REL_POS=198
MAX_MEA_POS=5360

# Data processing options
TOKENS_PER_SAMPLE=4096
SHORTEN=none
SHORTEN_DATA_SPLIT_LIST=""
SAMPLE_BREAK_MODE=complete_doc
RATIO=4
SAMPLE_OVERLAP_RATE=4
PERM_INV=3

# Optimizer options
OPTIMIZER=adam
ADAM_BETAS="(0.9, 0.98)"
ADAM_EPS=1e-6
CLIP_NORM=0.0
WEIGHT_DECAY=0.01

# Training options
BATCH_SIZE=1
LR=0.00001
LR_SCHEDULER=polynomial_decay

# Logging and checkpoing saving
SAVE_DIR=./results/$VARIANT/ckpt
TENSORBOARD_LOGDIR=./results/$VARIANT/logs
LOG_FORMAT=simple
LOG_INTERVAL=10