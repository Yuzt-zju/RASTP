#!/bin/bash

# 配置
datasets=("sports" "toys")
gpus=("2" "2" "2")
cpu_ranges=("31-40" "31-40" "31-40")
seeds=(1 999)  # 添加 seed 列表

experiment="tiger_train_flat"
num_hierarchies=4
mode=rkmeans

# 顺序遍历每个数据集
for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    gpu="${gpus[$i]}"
    cpu_range="${cpu_ranges[$i]}"

    # 对每个 seed 运行一次
    for seed in "${seeds[@]}"; do
        echo "Running dataset: $dataset on GPU $gpu, CPU cores $cpu_range, seed=$seed"

        CUDA_VISIBLE_DEVICES="$gpu" \
        taskset -c "$cpu_range" \
        python -m src.train \
            experiment="$experiment" \
            data_dir="data/amazon_data/$dataset" \
            semantic_id_path="logs/inference/runs/$dataset/sid/$mode/pickle/merged_predictions_tensor.pt" \
            num_hierarchies="$num_hierarchies" \
            seed="$seed"
    done
done

echo "All datasets and seeds completed."