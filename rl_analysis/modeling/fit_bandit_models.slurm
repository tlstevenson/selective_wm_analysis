#!/bin/bash -l
#SBATCH --job-name=bandit_fit
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH -p high
#SBATCH --account=thanksgrp
#SBATCH -o /group/thanksgrp/python/probabilistic_bandit/log/batch_%A.txt
#SBATCH --mail-type=ALL

LOGDIR=/group/thanksgrp/python/probabilistic_bandit/log/$SLURM_ARRAY_JOB_ID
mkdir -p $LOGDIR

# Run Python script

if test -z "$CONFIGPATH"
then
   echo "Please specify a config file path with the variable name CONFIGPATH"
else
   module load conda
   conda activate /group/thanksgrp/envs/neuropy

   # save absolute path to python executable
   PYTHON=$(which python)

   srun -n 1 -o $LOGDIR/fit_%a.txt $PYTHON -u /group/thanksgrp/python/code/selective_wm_analysis/rl_analysis/modeling/fit_bandit_models.py --config_path "$CONFIGPATH"
fi

