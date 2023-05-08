#!/bin/bash

#cd ../..

# custom config
DATA="./data/"
TRAINER=MaPLe

DATASET=$1
SEED=$2
CUSTOM_NAME=$3
WEIGHTSPATH='weights/maple/ori'

CFG=0-7tpt_vit_b16_c2_ep5_batch4_2ctx_cross_datasets
SHOTS=16
LOADEP=2

MODEL_DIR=${WEIGHTSPATH}/seed${SEED}

DIR=output/evaluation/${TRAINER}/DistrEntr_tpt_vit_b16_c2_ep5_batch4_2ctx_cross_datasets_${SHOTS}shots/${DATASET}/seed${SEED}_${CUSTOM_NAME}
if [ -d "$DIR" ]; then
    echo "Results are already available in ${DIR}. Skipping..."
else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"
    # Evaluate on evaluation datasets
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --tpt \
    DATASET.NUM_SHOTS ${SHOTS} \

fi