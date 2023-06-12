#!/bin/bash

export WANDB_API_KEY=d34f864932245bbdf3a9396a1ebde883ad2068f3

python -u train_ff_spice.py \
    --num-layers 6 \
    --latent-size 128 \
    --mlp-hidden-size 512 \
    --node-attn --use-bn --pred-pos-residual \
    --random-rotation \
    --checkpoint_dir ./checkpoints/ff_spice_small \
