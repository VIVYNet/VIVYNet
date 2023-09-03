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
echo -e "\n\n\nVARIANT:  ${VIVY_VARIANT}"
echo "============================================"
for var in $(compgen -v VIVY_); do
    stripped_var=${var#VIVY_}
    echo "$stripped_var: ${!var}"
done
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
mkdir $VIVY_OUTPUT_DIR
mkdir $VIVY_OUTPUT_DIR/slurm
echo "Directories made"
echo


# Start the slurm run process
echo "Dispatching train run..."
sbatch \
  --job-name="VIVYNET Training - ${VIVY_VARIANT}" \
  --output="$VIVY_OUTPUT_DIR/slurm/out_%j.txt" \
  --error="$VIVY_OUTPUT_DIR/slurm/err_%j.txt"  \
  --gres=gpu:A100:1 \
  --mail-user=$EMAIL \
  ./train_transformer.sh \
  $VIVY_VARIANT