# RASTP - Representation-Aware Semantic Token Pruning

[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/config-hydra-89b8cd)](https://hydra.cc/)
[![Lightning](https://img.shields.io/badge/pytorch-lightning-792ee5)](https://lightning.ai/)

**RASTP** (Representation-Aware Semantic Token Pruning) is a **training-efficient strategy** for generative recommendation systems that use Semantic Identifiers (SIDs). It dynamically prunes less informative SID tokens during training to significantly reduce computational overhead while maintaining or even slightly improving recommendation performance.

Built upon the [GRID](https://github.com/snap-research/GRID) framework, RASTP introduces a lightweight, plug-and-play module that evaluates token importance based on **semantic saliency** (representation magnitude) and **attention centrality** (cumulative attention scores).

## 🚀 Overview

RASTP addresses the key bottleneck in SID-based generative recommendation: **long input sequences caused by multi-token SIDs**, which dramatically increase training time and memory consumption.

### Core Mechanism

- **Dynamic Token Selection**: During training, RASTP computes an importance score for each SID token:
  ```
  Importance = (Cumulative Attention Score) × (L1 Norm of Token Representation)
  ```
- **Pruning Strategy**: Retains only the top-*k* most informative tokens (e.g., top 2/3) based on importance scores.
- **Plug-and-Play**: Integrated into the Transformer encoder, requiring minimal code changes and incurring negligible overhead.

### Key Benefits

- ✅ **26.7% faster training** on real-world Amazon datasets
- ✅ **Maintains or slightly improves** recommendation performance (Recall@K, NDCG@K)
- ✅ Compatible with any Transformer-based generative recommender (e.g., TIGER)
- ✅ Reduces GPU memory pressure during training

## 📦 Installation

```bash
git clone https://github.com/Yuzt-zju/RASTP.git
cd RASTP

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start
### 1. Data Preparation

Follow the [GRID](https://github.com/snap-research/GRID).

### 2. Replace key file

To avoid interference from other implementations, our strategy is to directly modify the relevant file in the library. Therefore, after setting up the environment, replace the corresponding library file with the `modeling_t5.py` provided in our repository. For details, see the `class T5Stack` in this file—specifically, line 1324, where you can specify after which Transformer layer pruning should be applied (-1 means no rastp). 

### 2. Generate Semantic IDs (SIDs)

Follow the standard pipeline to obtain SIDs:

```bash
# Step 1: Generate embeddings
python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/amazon_data/beauty # avaiable data includes 'beauty', 'sports', and 'toys'

# Step 2: Train SID model (e.g., Residual K-means)
python -m src.train experiment=rkmeans_train_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=<output_path_from_step_2>/merged_predictions_tensor.pt \ # this can be found in the log dirs in step2
    embedding_dim=2048 \ # the model dimension of the LLMs you use in step 2. 2048 for flan-t5-xl as used in this example.
    num_hierarchies=3 \  # we train 3 codebooks
    codebook_width=256 \ # each codebook has 256 rows of centroids  

# Step 3: Generate SIDs
python -m src.inference experiment=rkmeans_inference_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=<output_path_from_step_2>/merged_predictions_tensor.pt \ 
    embedding_dim=2048 \ 
    num_hierarchies=3 \  
    codebook_width=256 \ 
    ckpt_path=<the_checkpoint_you_just_get_above> # this can be found in the log dir for training SIDs
```

### 3. Train with RASTP

Enable RASTP by specifying the pruning configuration in your training command:

```bash
# Train generative recommender with RASTP enabled
python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/beauty \ 
    semantic_id_path=<output_path_from_step_3>/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4 
```

## 📊 Results

On Amazon datasets (Beauty, Sports, Toys), RASTP achieves:

| Metric | Without RASTP | With RASTP | Δ |
|--------|---------------|------------|----|
| Training Time | 100% | **73.3%** | **↓26.7%** |
| Recall@10 (Beauty) | 0.0645 | **0.0656** | ↑ |
| NDCG@10 (Toys) | 0.0287 | **0.0289** | ↑ |
## 🤝 Acknowledgments

- Built on [GRID](https://github.com/snap-research/GRID) by Snap Research
- Powered by [PyTorch](https://pytorch.org/), [PyTorch Lightning](https://lightning.ai/), and [Hydra](https://hydra.cc/)
- Inspired by industrial needs for daily retraining of large-scale recommendation systems