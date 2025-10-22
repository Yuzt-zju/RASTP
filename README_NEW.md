# RASTP - Recommendation with Advanced Semantic Token Processing

[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/config-hydra-89b8cd)](https://hydra.cc/)
[![Lightning](https://img.shields.io/badge/pytorch-lightning-792ee5)](https://lightning.ai/)

**RASTP** (Recommendation with Advanced Semantic Token Processing) is an enhanced framework built on top of the GRID (Generative Recommendation with Semantic IDs) architecture. This project extends the original GRID implementation with advanced T5 modeling capabilities and improved semantic ID processing for recommendation systems.

## 🚀 Overview

RASTP builds upon the GRID framework with three core enhancements:

- **Enhanced T5 Modeling**: Custom T5 model implementation with advanced attention mechanisms and token processing
- **Improved Semantic ID Learning**: Enhanced residual quantization techniques for better semantic representation
- **Advanced Generative Recommendations**: Extended transformer architectures for more accurate recommendation generation

## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM (recommended for large datasets)

### Environment Setup

```bash
# Clone the repository
git clone <your-repository-url>
cd RASTP

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### 1. Data Preparation

Prepare your dataset following the GRID format structure:

```
data/
├── amazon_data/
│   ├── beauty/
│   │   ├── train/       # training sequences of user history 
│   │   ├── validation/  # validation sequences of user history 
│   │   ├── test/        # testing sequences of user history 
│   │   └── items/       # text descriptions of all items
│   ├── sports/
│   └── toys/
```

**Data Format Requirements:**
- Training/validation/test files should contain user interaction sequences
- Items file should contain text descriptions for each item
- Follow the same format as the original GRID Amazon dataset

**Pre-processed Amazon Data:**
You can download the pre-processed Amazon data from the [GRID Google Drive link](https://drive.google.com/file/d/1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz/view?usp=sharing).

### 2. Environment Preparation

Ensure your environment meets the following requirements:

```bash
# Check Python version
python --version  # Should be 3.10+

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify GPU memory
nvidia-smi
```

**System Requirements:**
- GPU with at least 8GB VRAM (recommended: 16GB+)
- 16GB+ system RAM
- CUDA 11.8+ or 12.0+

### 3. Replace modeling_t5.py File

**Important:** This project includes a custom `modeling_t5.py` file that extends the standard T5 implementation. This file must be properly integrated into your HuggingFace transformers installation.

**Option A: Manual Integration (Recommended)**
```bash
# Find your transformers installation path
python -c "import transformers; print(transformers.__file__)"

# Backup original file
cp <transformers_path>/models/t5/modeling_t5.py <transformers_path>/models/t5/modeling_t5.py.backup

# Replace with custom implementation
cp modeling_t5.py <transformers_path>/models/t5/modeling_t5.py
```

**Option B: Environment Variable Method**
```bash
# Set environment variable to use custom T5 implementation
export TRANSFORMERS_CUSTOM_T5_PATH=/path/to/your/RASTP/modeling_t5.py
```

**Verification:**
```bash
# Test the custom T5 implementation
python -c "from transformers import T5ForConditionalGeneration; print('Custom T5 loaded successfully')"
```

### 4. Step-by-Step Commands

#### Step 1: Generate Item Embeddings with LLMs

Generate embeddings from Large Language Models for all items:

```bash
# Run step1.sh for all datasets
bash step1.sh

# Or run individually:
CUDA_VISIBLE_DEVICES=0 python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/amazon_data/beauty
CUDA_VISIBLE_DEVICES=0 python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/amazon_data/toys
CUDA_VISIBLE_DEVICES=0 python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/amazon_data/sports
```

**Output:** Embeddings will be saved in `logs/inference/runs/{dataset}/item_embedding/pickle/merged_predictions_tensor.pt`

#### Step 2: Train Semantic ID Models

Learn semantic ID centroids using Residual Quantization techniques:

```bash
# Run step2.sh for all datasets
bash step2.sh

# Or run individually (example for beauty dataset):
CUDA_VISIBLE_DEVICES=1 python -m src.train experiment=rqvae_train_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=logs/inference/runs/beauty/item_embedding/pickle/merged_predictions_tensor.pt \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256
```

**Available Models:**
- `rkmeans_train_flat`: Residual K-means
- `rqvae_train_flat`: Residual Quantization VAE

**Output:** Checkpoints will be saved in `logs/train/runs/{dataset}/sid/{model}/checkpoints/`

#### Step 3: Generate Semantic IDs

Convert embeddings to semantic IDs using trained models:

```bash
# Run step3.sh for all datasets
bash step3.sh

# Or run individually (example):
CUDA_VISIBLE_DEVICES=0 python -m src.inference experiment=rqvae_inference_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=logs/inference/runs/beauty/item_embedding/pickle/merged_predictions_tensor.pt \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256 \
    ckpt_path=logs/train/runs/beauty/sid/rqvae/checkpoints/checkpoint_000_003000.ckpt
```

**Output:** Semantic IDs will be saved in `logs/inference/runs/{dataset}/sid/{model}/pickle/merged_predictions_tensor.pt`

#### Step 4: Train Generative Recommendation Models

Train the recommendation model using learned semantic IDs:

```bash
# Run step4.sh for all datasets
bash step4.sh

# Or run individually (example):
CUDA_VISIBLE_DEVICES=0 taskset -c 21-30 python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/sports \
    semantic_id_path=logs/inference/runs/sports/sid/rkmeans/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4 \
    seed=1
```

**Note:** `num_hierarchies=4` (add 1 to the previous step's value due to deduplication)

#### Step 5: Generate Recommendations

Run inference to generate recommendations:

```bash
# Run step5.sh for all datasets
bash step5.sh

# Or run individually (example):
CUDA_VISIBLE_DEVICES=0 python -m src.inference experiment=tiger_inference_flat \
    data_dir=data/amazon_data/beauty \
    semantic_id_path=logs/inference/runs/beauty/sid/rkmeans/pickle/merged_predictions_tensor.pt \
    ckpt_path=logs/train/runs/beauty/gr/rkmeans_vatp_seed_2025/checkpoints/checkpoint_epoch_000_step_002900.ckpt \
    seed=2025 \
    num_hierarchies=4
```

## 🛠️ Advanced Usage

### Batch Training Scripts

Use the provided batch training scripts for efficient multi-dataset training:

```bash
# Train with multiple seeds for robustness
bash train_gr.sh   # Uses seeds: 999, 1024, 2025
bash train_gr2.sh  # Uses seeds: 1, 999
```

### Custom Configuration

Modify experiment configurations in `configs/experiment/`:

```bash
# Example: Custom embedding dimensions
python -m src.train experiment=custom_experiment embedding_dim=1024

# Example: Different codebook sizes
python -m src.train experiment=custom_experiment codebook_width=512
```

### Monitoring Training

Monitor training progress using TensorBoard:

```bash
# Start TensorBoard
tensorboard --logdir=logs/train/runs

# View in browser: http://localhost:6006
```

## 📊 Supported Models

### Semantic ID Generation:
1. **Residual K-means** (`rkmeans_train_flat`) - Fast and efficient
2. **Residual Quantization VAE** (`rqvae_train_flat`) - Better quality, slower training
3. **Residual Vector Quantization** - Balanced approach

### Generative Recommendation:
1. **TIGER** (`tiger_train_flat`) - Transformer-based generative model

## 🔧 Troubleshooting

### Common Issues:

**CUDA Out of Memory:**
```bash
# Reduce batch size or use gradient accumulation
python -m src.train experiment=tiger_train_flat trainer.accumulate_grad_batches=4
```

**Custom T5 Not Loading:**
```bash
# Verify installation
python -c "import transformers; from transformers import T5ForConditionalGeneration; print('T5 loaded')"
```

**Path Not Found Errors:**
```bash
# Check data directory structure
ls -la data/amazon_data/beauty/
```

### Performance Optimization:

**Multi-GPU Training:**
```bash
# Use multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m src.train experiment=tiger_train_flat trainer.devices=2
```

**CPU Affinity:**
```bash
# Bind to specific CPU cores (already included in scripts)
taskset -c 21-30 python -m src.train experiment=tiger_train_flat
```

## 📚 Citation

If you use RASTP in your research, please cite both the original GRID paper and this work:

```bibtex
@inproceedings{grid,
  title     = {Generative Recommendation with Semantic IDs: A Practitioner's Handbook},
  author    = {Ju, Clark Mingxuan and Collins, Liam and Neves, Leonardo and Kumar, Bhuvesh and Wang, Louis Yufeng and Zhao, Tong and Shah, Neil},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM)},
  year      = {2025}
}

@misc{rastp2025,
  title={RASTP: Recommendation with Advanced Semantic Token Processing},
  author={Your Name},
  year={2025},
  note={Enhanced implementation of GRID with advanced T5 modeling}
}
```

## 🤝 Acknowledgments

- Built on top of [GRID](https://github.com/snap-research/GRID) by Snap Research
- Enhanced with custom T5 modeling capabilities
- Powered by [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)
- Configuration management by [Hydra](https://hydra.cc/)

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section above
- Refer to the original GRID documentation for additional guidance

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
