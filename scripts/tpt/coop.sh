#!/bin/bash

#cd ../..

# custom config
DATA="./data/"
TRAINER=CoOp
CFG=tpt_vit_b16_ep50

DATASET=$1
SEED=$2
WEIGHTSPATH='weights/coop/vit_b16_ep50_16shots/nctx4_cscFalse_ctpend'

SHOTS=16
NCTX=4
CSC=False
CTP=end

for SEED in 1
do
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED} \
    --model-dir ${WEIGHTSPATH}/seed${SEED} \
    --load-epoch 50 \
    --tpt \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP}
done