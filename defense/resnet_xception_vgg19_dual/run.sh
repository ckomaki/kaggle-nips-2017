#!/bin/bash

INPUT_DIR=$1
OUTPUT_FILE=$2

cd common
pip install Keras-2.0.8-py2.py3-none-any.whl --no-index --find-links=$(pwd)
pip install h5py-2.7.1-cp27-cp27mu-manylinux1_x86_64.whl --no-index --find-links=$(pwd)
cd ../

python defense.py \
  --input-dir="${INPUT_DIR}" \
  --output-file="${OUTPUT_FILE}" \
