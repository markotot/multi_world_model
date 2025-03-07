#!/bin/bash

APOC_USERNAME=$1
APOC_PASSPHRASE=$2
APOC_PASSWORD=$3
APOC_PRIVATE_KEY=$4
PROJECT_NAME=$5

rsync -avz --partial -e "ssh -i $APOC_PRIVATE_KEY" \
      $APOC_USERNAME@login.hpc.qmul.ac.uk:$PROJECT_NAME/checkpoints .
