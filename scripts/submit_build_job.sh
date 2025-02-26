#!/bin/bash
#$ -pe smp 8
#$ -l h_vmem=4G
#$ -l h_rt=1:0:0
#$ -l rocky
#$ -cwd
#$ -j y
#$ -o job_results

# Replace the following line with a program or command
APPTAINERENV_NSLOTS=${NSLOTS}
apptainer build --force containers/multi_world_model.sif $PROJECT_NAME/apptainer/multi_world_model.def;
