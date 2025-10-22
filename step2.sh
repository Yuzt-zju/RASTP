CUDA_VISIBLE_DEVICES=1 python -m src.train experiment=rqvae_train_flat data_dir=data/amazon_data/beauty embedding_path=logs/inference/runs/beauty/item_embedding/pickle/merged_predictions_tensor.pt embedding_dim=2048 num_hierarchies=3 codebook_width=256  
CUDA_VISIBLE_DEVICES=2 python -m src.train experiment=rqvae_train_flat data_dir=data/amazon_data/sports embedding_path=logs/inference/runs/sports/item_embedding/pickle/merged_predictions_tensor.pt embedding_dim=2048 num_hierarchies=3 codebook_width=256  
CUDA_VISIBLE_DEVICES=1 python -m src.train experiment=rqvae_train_flat dataset=toys embedding_path=logs/inference/runs/toys/item_embedding/pickle/merged_predictions_tensor.pt embedding_dim=2048 num_hierarchies=3 codebook_width=256  

rkmeans_train_flat
rqvae_train_flat