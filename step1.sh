CUDA_VISIBLE_DEVICES=0 python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/amazon_data/beauty
CUDA_VISIBLE_DEVICES=0 python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/amazon_data/toys
CUDA_VISIBLE_DEVICES=0 python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/amazon_data/sports