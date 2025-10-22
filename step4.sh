CUDA_VISIBLE_DEVICES=0 taskset -c 21-30 python -m src.train experiment=tiger_train_flat 
    data_dir=data/amazon_data/sports
    semantic_id_path=logs/inference/runs/sports/sid/rkmeans/pickle/merged_predictions_tensor.pt 
    num_hierarchies=4 
    seed=1
    # Please note that we add 1 for num_hierarchies because in the previous step we appended one additional digit to de-duplicate the semantic IDs we generate.

CUDA_VISIBLE_DEVICES=3 taskset -c 31-40 python -m src.train experiment=tiger_train_flat 
    data_dir=data/amazon_data/sports
    semantic_id_path=logs/inference/runs/sports/sid/rqvae/pickle/merged_predictions_tensor.pt
    num_hierarchies=4 


CUDA_VISIBLE_DEVICES=0 taskset -c 21-30 python -m src.train experiment=tiger_train_flat 
    data_dir=data/amazon_data/toys
    semantic_id_path=logs/inference/runs/toys/sid/rqvae/pickle/merged_predictions_tensor.pt
    num_hierarchies=4 