#!/bin/bash
#
# run_attack.sh is a script which executes the attack
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_attack.sh INPUT_DIR OUTPUT_DIR MAX_EPSILON
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_DIR - directory where adversarial images should be written
#   MAX_EPSILON - maximum allowed L_{\infty} norm of adversarial perturbation
#

INPUT_DIR=$1
OUTPUT_DIR=$2
MAX_EPSILON=$3

# For how many iterations run this attack
NUM_ITERATIONS=20

NPY_DIR=$(mktemp -d)

attack0()
{
    python attack_iter_target_class.py \
      --input_dir="${INPUT_DIR}" \
      --output_dir="${OUTPUT_DIR}" \
      --max_epsilon="${MAX_EPSILON}" \
      --num_iter="${NUM_ITERATIONS}" \
      --step_n="${STEP_N}" \
      --step="${1}" \
      --npy_dir=${NPY_DIR} \
      --checkpoint_path=ens_adv_inception_resnet_v2.ckpt
}

attack1()
{
    python attack_iter_target_class_inception.py \
      --input_dir="${INPUT_DIR}" \
      --output_dir="${OUTPUT_DIR}" \
      --max_epsilon="${MAX_EPSILON}" \
      --num_iter="${NUM_ITERATIONS}" \
      --step_n="${STEP_N}" \
      --step="${1}" \
      --npy_dir=${NPY_DIR} \
      --checkpoint_path=inception_v3.ckpt
}


STEP_N=12
attack0 0
attack1 1
attack0 2
attack1 3
attack0 4
attack1 5
attack0 6
attack1 7
attack0 8
attack1 9
attack0 10
attack1 11


