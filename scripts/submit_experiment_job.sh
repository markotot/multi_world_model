#!/bin/bash
#$ -l h_vmem=11G
#$ -l h_rt=1:0:0
#$ -pe smp 8
#$ -l gpu=1
#$ -l rocky
#$ -cwd
#$ -j y
#$ -o job_results

APPTAINERENV_NSLOTS=${NSLOTS}
apptainer run --nv --env-file myenvs --env "JOB_TYPE=$JOB_TYPE,RUN_NAME=$RUN_NAME,NUM_SEEDS=$NUM_SEEDS,ENV_NAME=$ENV_NAME,WANDB_API_KEY=$WANDB_API_KEY" containers/multi_world_model.sif
