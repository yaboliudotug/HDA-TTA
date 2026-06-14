#! /usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATASET=cifar10

# ===================================
# Configuration - modify these paths
# ===================================
DATADIR=/disk/liuyabo/data/tta_classification
LEVEL=5
LR=0.01
BS_SSL=256
BS_ALIGN=256

if [ "$#" -lt 2 ]; then
    CORRUPT=snow
    NSAMPLE=100000
else
    CORRUPT=$1
    NSAMPLE=$2
fi

# ===================================
# Run TTAC on all 15 corruptions
# ===================================
for CORRUPT in 'gaussian_noise' 'shot_noise' 'impulse_noise' 'defocus_blur' 'glass_blur' 'motion_blur' 'zoom_blur' 'frost' 'fog' 'brightness' 'contrast' 'elastic_transform' 'pixelate' 'jpeg_compression' 'snow'
do
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --dataroot ${DATADIR} \
        --dataset ${DATASET} \
        --resume ../../ttac_v1/cifar/results/${DATASET}_joint_resnet50 \
        --outf results/${DATASET}_${CORRUPT}_ttac \
        --corruption ${CORRUPT} \
        --level ${LEVEL} \
        --workers 4 \
        --batch_size ${BS_SSL} \
        --batch_size_align ${BS_ALIGN} \
        --lr ${LR} \
        --num_sample ${NSAMPLE} \
        --iters 4 \
        --align_ext \
        --filter_with_cluster_within_category \
        --num_samples_per_cluster 20 \
        --use_feature_matching_loss \
        --random_around_feature \
        --do_addtional_cluster_within_category \
        --weight_cluster_kl_loss 0.0001
done