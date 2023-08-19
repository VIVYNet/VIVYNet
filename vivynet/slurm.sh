#!/bin/sh

# Initialize the config variable to an empty value
CONFIG_FILE=""

# Iterate through the command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift # Remove --config from args
            shift # Remove its value from args
            ;;
        *)
            # Other flags or arguments can be processed here
            shift # Remove generic arg
            ;;
    esac
done

# If --config wasn't set, prompt the user for the config file name
if [[ -z "$CONFIG_FILE" ]]; then
    read -p "Please type the name of the config file (from ./configs/) you want to use: " CONFIG_FILE
fi

# Check if the config file exists and is a regular file
if [[ ! -f "./configs/$CONFIG_FILE.sh" ]]; then
    echo "Error: Config file '$CONFIG_FILE' does not exist"
    exit 1
fi

# Source the wandb api key file and config file to run
echo
echo "Using config file: $CONFIG_FILE"
source ./configs/$CONFIG_FILE.sh
source ./personal.sh

# Print config settings
echo -e "\n\n\nVARIANT:  ${VARIANT}"
echo "============================================"
echo "USER_DIR:  ${USER_DIR}"
echo "TASK:  ${TASK}"
echo "CRITERION:  ${CRITERION}"
echo "ARCH:  ${ARCH}"
echo "SEED:  ${SEED}"
echo "PT_ENC:  ${PT_ENC}"
echo "FREEZE_ENC:  ${FREEZE_ENC}"
echo "PT_DEC:  ${PT_DEC}"
echo "FREEZE_DEC:  ${FREEZE_DEC}"
echo "EVT_VOC_SIZE:  ${EVT_VOC_SIZE}"
echo "TRK_VOC_SIZE:  ${TRK_VOC_SIZE}"
echo "DUR_VOC_SIZE:  ${DUR_VOC_SIZE}"
echo "INS_VOC_SIZE:  ${INS_VOC_SIZE}"
echo "MAX_REL_POS:  ${MAX_REL_POS}"
echo "MAX_MEA_POS:  ${MAX_MEA_POS}"
echo "TOKENS_PER_SAMPLE:  ${TOKENS_PER_SAMPLE}"
echo "SHORTEN:  ${SHORTEN}"
echo "SHORTEN_DATA_SPLIT_LIST:  ${SHORTEN_DATA_SPLIT_LIST}"
echo "SAMPLE_BREAK_MODE:  ${SAMPLE_BREAK_MODE}"
echo "RATIO:  ${RATIO}"
echo "SAMPLE_OVERLAP_RATE:  ${SAMPLE_OVERLAP_RATE}"
echo "PERM_INV:  ${PERM_INV}"
echo "OPTIMIZER:  ${OPTIMIZER}"
echo "ADAM_BETAS:  ${ADAM_BETAS}"
echo "ADAM_EPS:  ${ADAM_EPS}"
echo "CLIP_NORM:  ${CLIP_NORM}"
echo "WEIGHT_DECAY:  ${WEIGHT_DECAY}"
echo "BATCH_SIZE:  ${BATCH_SIZE}"
echo "LR:  ${LR}"
echo "LR_SCHEDULER:  ${LR_SCHEDULER}"
echo "OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "SAVE_DIR:  ${SAVE_DIR}"
echo "TENSORBOARD_LOGDIR:  ${TENSORBOARD_LOGDIR}"
echo "LOG_FORMAT:  ${LOG_FORMAT}"
echo "LOG_INTERVAL:  ${LOG_INTERVAL}"
echo

# Checkpoint Stop
read -p "Do you want to proceed? (Y/n): " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    echo "Proceeding..."
    echo
    # Place the code to be executed on confirmation here
else
    echo "Dispatch canceled"
    exit 1
fi

# Make directories
echo "Making directories..."
mkdir $OUTPUT_DIR
mkdir $OUTPUT_DIR/slurm
echo "Directories made"
echo


# Start the slurm run process
echo "Dispatching train run..."
sbatch \
  --job-name="VIVYNET Training - ${VARIANT}" \
  --output="$OUTPUT_DIR/slurm/out_%j.txt" \
  --error="$OUTPUT_DIR/slurm/err_%j.txt"  \
  --gres=gpu:A100:1 \
  --mail-user=$EMAIL \
  ./train_transformer_slurm.sh \
  $VARIANT