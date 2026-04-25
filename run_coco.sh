#!/usr/bin/env bash

EXPNAME=$1
SAVEDIR=checkpoints/coco/${EXPNAME}
IMAGENET_PRETRAIN=/data/nvme0n1/grade1/students/huangfuyi/code/DeFRCN-main/data/pretrain_weights/ImageNetPretrained/MSRA/R-101.pkl                         # <-- change it to you path
IMAGENET_PRETRAIN_TORCH=/data/nvme0n1/grade1/students/huangfuyi/code/DeFRCN-main/data/pretrain_weights/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth


# ------------------------------- Base Pre-train ---------------------------------- #
python3 main.py --num-gpus 2 --config-file configs/coco/defrcn_det_r101_base.yaml     \
    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                         \
           OUTPUT_DIR ${SAVEDIR}/defrcn_det_r101_base


# ----------------------------- Model Preparation --------------------------------- #
python3 tools/model_surgery.py --dataset coco --method randinit                        \
    --src-path ${SAVEDIR}/defrcn_det_r101_base/model_final.pth                         \
    --save-dir ${SAVEDIR}/defrcn_det_r101_base
BASE_WEIGHT=${SAVEDIR}/defrcn_det_r101_base/model_reset_surgery.pth


# ------------------------------ Novel Fine-tuning ------------------------------- #
# --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)
for seed in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 5 10 30
    do
        python3 tools/create_config.py --dataset coco14 --config_root configs/coco     \
            --shot ${shot} --seed ${seed} --setting 'gfsod'
        CONFIG_PATH=configs/coco/defrcn_gfsod_r101_novel_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVEDIR}/defrcn_gfsod_r101_novel/tfa-like/${shot}shot_seed${seed}
        python3 main.py --num-gpus 2 --config-file ${CONFIG_PATH}                      \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}               \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth
    done
done
python3 tools/extract_results.py --res-dir ${SAVEDIR}/defrcn_gfsod_r101_novel/tfa-like --shot-list 5 10 30  # surmarize all results


