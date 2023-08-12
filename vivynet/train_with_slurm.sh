#!/bin/sh
#SBATCH -p gpu                                                                                        ## Partition
#SBATCH -q wildfire                                                                                   ## QOS
#SBATCH -c 1                                                                                          ## Number of Cores
#SBATCH --time=1440                                                                                   ## 1 day of compute
#SBATCH --gres=gpu:V100:1                                                                             ## 1 of V100 GPU
#SBATCH --mem 32G                                                                                     ## 64 GB of RAM
#SBATCH --output=results/run10/slurm/out_%j.txt                                                       ## job /dev/stdout record
#SBATCH --error=results/run10/slurm/err_%j.txt                                                        ## job /dev/stderr record
#SBATCH --export=NONE                                                                                 ## keep environment clean
#SBATCH --mail-type=ALL                                                                               ## notify for any job state change
#SBATCH --mail-user=blherre4@asu.edu                                                                  ## notify email (%u expands -> username)
#SBATCH --job-name="VIVYNet Training - run6"                                                          ## optional job name

echo "Purging modules"
module purge
echo "Loading python 3 from anaconda module"
module load anaconda/py3
echo "Loading VIVYNET conda environment"
source activate vivyenv #-pytorch-1.13.1
echo "Showing GPU details"
nvidia-smi -L
nvidia-smi
echo "Running training python script"
bash train_transformer.sh
echo "Finished"