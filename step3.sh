CUDA_VISIBLE_DEVICES=0 python -m src.inference experiment=rqvae_inference_flat 
    data_dir=data/amazon_data/beauty
    embedding_path=logs/inference/runs/2025-09-08/13-43-23/pickle/merged_predictions_tensor.pt 
    embedding_dim=2048  
    num_hierarchies=3   
    codebook_width=256 
    ckpt_path=logs/train/runs/2025-09-08/13-52-37/checkpoints/checkpoint_000_000030.ckpt 

CUDA_VISIBLE_DEVICES=0 python -m src.inference experiment=rqvae_inference_flat 
    data_dir=data/amazon_data/sports
    embedding_path=logs/inference/runs/sports/item_embedding/pickle/merged_predictions_tensor.pt
    embedding_dim=2048  
    num_hierarchies=3   
    codebook_width=256 
    ckpt_path=logs/train/runs/sports/sid/rqvae/checkpoints/checkpoint_000_003000.ckpt

CUDA_VISIBLE_DEVICES=0 python -m src.inference experiment=rqvae_inference_flat 
    data_dir=data/amazon_data/toys
    embedding_path=logs/inference/runs/toys/item_embedding/pickle/merged_predictions_tensor.pt
    embedding_dim=2048  
    num_hierarchies=3   
    codebook_width=256 
    ckpt_path=logs/train/runs/toys/sid/rqvae/checkpoints/checkpoint_000_003000.ckpt

rqvae_train_flat