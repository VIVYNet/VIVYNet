#!/bin/sh
#SBATCH -p gpu                                      ## Partition
#SBATCH -q wildfire                                 ## QOS
#SBATCH -c 1                                        ## Number of Cores
#SBATCH --time=4320                                 ## 3 days of compute
#SBATCH --gres=gpu:RTX6000:1                        ## 1 of GTX1080 GPU
#SBATCH --mem 64G                                   ## 64 GB of RAM
#SBATCH --output=slurm/out_%j.txt                   ## job /dev/stdout record
#SBATCH --error=slurm/err_%j.txt                    ## job /dev/stderr record
#SBATCH --export=NONE                               ## keep environment clean
#SBATCH --mail-type=ALL                             ## notify for any job state change
#SBATCH --mail-user=blherre4@asu.edu                ## notify email (%u expands -> username)
#SBATCH --job-name="VIVYNet Training"               ## optional job name

echo "Purging modules"
module purge
echo "Loading python 3 from anaconda module"
module load anaconda/py3
echo "Loading VIVYNET conda environment"
source activate vivynet
echo "Showing GPU details"
nvidia-smi -L
nvidia-smi
echo "Running training python script"
bash train.sh
echo "Finished"