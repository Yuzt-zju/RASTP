CUDA_VISIBLE_DEVICES=0 python -m src.inference experiment=tiger_inference_flat 
    data_dir=data/amazon_data/beauty 
    semantic_id_path=logs/inference/runs/beauty/sid/rkmeans/pickle/merged_predictions_tensor.pt
    ckpt_path=logs/train/runs/beauty/gr/rkmeans_vatp_seed_2025/checkpoints/checkpoint_epoch_000_step_002900.ckpt
    seed=2025
    num_hierarchies=4 