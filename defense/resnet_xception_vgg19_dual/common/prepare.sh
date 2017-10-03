#!/bin/bash

set -exu

TASK=$1
MODEL_NAMES=$2
echo "models: ${MODEL_NAMES}"
echo "task: ${TASK}"

python pretrained_weight_downloader.py \
  --home-dir ~ \
  --current-dir $(pwd) \
  --model-names ${MODEL_NAMES}

python metadata_creator.py \
  --task ${TASK}
