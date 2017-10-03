#!/bin/bash

INPUT_DIR=$1
OUTPUT_DIR=$2
MAX_EPSILON=$3

cd common
pip install Keras-2.0.8-py2.py3-none-any.whl --no-index --find-links=$(pwd)
pip install h5py-2.7.1-cp27-cp27mu-manylinux1_x86_64.whl --no-index --find-links=$(pwd)
cd ../

python attack.py \
  --input-dir="${INPUT_DIR}" \
  --output-dir="${OUTPUT_DIR}" \
  --max-epsilon="${MAX_EPSILON}"
