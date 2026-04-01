#!/bin/bash
# NPC self-supervised learning example
python3 main.py --config config/self_supervised/npc_example.yml \
               --njobs 32 \
               --dev_njobs 4 \
               --task self-learning\
               --ckpdir ckpt/ \
               --logdir log/ \
               --seed 0

# Using NPC representation in phone classification
# (change all phn to spk for speaker classification)
python3 main.py --config config/downstream/phn_clf_example.yml \
               --njobs 24 \
               --dev_njobs 8 \
               --task phn-clf\
               --ckpdir ckpt/ \
               --logdir log/ \
               --seed 0
