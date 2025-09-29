# MRP-CSLR: A Multi-Representation Prompt–Guided Framework for Continuous Sign Language Recognition

This repository contains the core implementation of MRP-CSLR, a novel framework for continuous sign language recognition that combines multi-representation learning with prompt-guided attention mechanisms.

## 🏗️ Architecture Overview

### Key Components

1. **Multi-Representation Encoder**
   - Visual Feature Extractor with CNN backbone (ResNet/EfficientNet)
   - Spatial Encoder for spatial representation learning
   - Temporal Encoder with Transformer/LSTM for temporal modeling
   - Fusion mechanism to combine multi-modal representations

2. **Prompt-Guided Attention**
   - Learnable prompt embeddings for sign vocabulary
   - Cross-modal attention between visual features and prompts
   - Context-aware prompt generation
   - Self-attention for enhanced feature learning

3. **Sequence Recognition**
   - CTC-based sequence-to-sequence learning
   - Temporal smoothing and normalization
   - Multiple classification heads

## 📁 Project Structure

```
MRP-CSLR/
├── models/                          # Core model implementations
│   ├── __init__.py
│   ├── multi_representation_encoder.py  # Multi-representation encoder
│   ├── prompt_guided_attention.py      # Prompt-guided attention mechanism
│   └── mrp_cslr.py                     # Main MRP-CSLR model
├── utils/                           # Utility functions
│   ├── __init__.py
│   ├── data_processing.py          # Video processing and data utilities
│   └── training.py                 # Training utilities and metrics
├── configs/                        # Configuration files
│   ├── __init__.py
│   └── config.py                   # Model and training configurations
├── scripts/                        # Training and inference scripts
│   ├── train.py                    # Main training script
│   └── inference.py                # Inference script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ynlgcn-YLC/MRP-CSLR.git
cd MRP-CSLR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

Train with default configuration:
```bash
python scripts/train.py --config default
```

Train with debug configuration (for quick testing):
```bash
python scripts/train.py --config debug
```

Train with dataset-specific configurations:
```bash
# For PHOENIX-2014 dataset
python scripts/train.py --config phoenix

# For CSL-Daily dataset  
python scripts/train.py --config csl_daily

# For large-scale experiments
python scripts/train.py --config large
```

### Inference

Run inference on a video:
```bash
python scripts/inference.py --model_path checkpoints/best_model.pth --video_path path/to/video.mp4
```

Create and test with a demo video:
```bash
python scripts/inference.py --model_path checkpoints/best_model.pth --create_demo
```

## 🔧 Configuration

The framework supports multiple pre-defined configurations:

- `default`: Baseline configuration
- `debug`: Minimal settings for quick testing  
- `phoenix`: Optimized for PHOENIX-2014 dataset
- `csl_daily`: Optimized for CSL-Daily dataset
- `large`: Large-scale model configuration

You can also create custom configurations by modifying `configs/config.py`.

## 🏋️ Model Architecture Details

### Multi-Representation Encoder
- **Visual Backbone**: Configurable CNN (ResNet50, EfficientNet, etc.)
- **Feature Dimension**: 2048 (default), configurable
- **Spatial Encoding**: Multi-layer perceptron with layer normalization
- **Temporal Encoding**: Transformer encoder with multi-head attention

### Prompt-Guided Attention
- **Prompt Embeddings**: Learnable embeddings for sign vocabulary
- **Context Generation**: Dynamic prompt selection based on visual context
- **Cross-Modal Attention**: 8-head attention between visual and prompt features
- **Feature Enhancement**: Self-attention and feed-forward networks

### Training Strategy
- **Loss Function**: CTC loss for sequence-to-sequence learning
- **Optimization**: AdamW with different learning rates for different components
- **Learning Rate**: Warmup + cosine annealing schedule
- **Data Augmentation**: Video-specific augmentations (crop, flip, color jitter)

## 📊 Performance

The framework has been evaluated on standard sign language datasets:

- **PHOENIX-2014**: State-of-the-art performance on German sign language
- **CSL-Daily**: Competitive results on Chinese sign language
- **Custom Datasets**: Flexible architecture supports various sign languages

## 🛠️ Customization

### Adding New Datasets

1. Implement a custom dataset class inheriting from `torch.utils.data.Dataset`
2. Update the vocabulary size in configuration
3. Modify data loading and preprocessing as needed

### Model Architecture Modifications

- **Backbone**: Change `visual_backbone` in model config
- **Dimensions**: Adjust `feature_dim`, `embed_dim`, `num_heads`
- **Prompts**: Modify `num_prompts` based on vocabulary size
- **Loss Function**: Switch between CTC and cross-entropy loss

### Training Hyperparameters

All hyperparameters are configurable through the configuration system:
- Learning rates for different components
- Batch size and sequence length
- Augmentation parameters
- Validation and checkpointing intervals

## 🔬 Core Components Testing

Test individual components:

```bash
# Test multi-representation encoder
python -c "from models.multi_representation_encoder import *; print('✓ Multi-representation encoder')"

# Test prompt-guided attention
python -c "from models.prompt_guided_attention import *; print('✓ Prompt-guided attention')"

# Test complete model
python -c "from models.mrp_cslr import *; print('✓ Complete MRP-CSLR model')"

# Test data processing
python -c "from utils.data_processing import *; print('✓ Data processing utilities')"

# Test training utilities
python -c "from utils.training import *; print('✓ Training utilities')"
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision 0.10+
- OpenCV 4.5+
- Additional dependencies listed in `requirements.txt`

## 🎯 Key Features

- **Multi-Representation Learning**: Combines spatial and temporal representations
- **Prompt-Guided Framework**: Uses learnable prompts for enhanced recognition
- **Flexible Architecture**: Supports different backbones and configurations  
- **CTC-based Training**: Handles variable-length sequences without alignment
- **Comprehensive Evaluation**: Multiple metrics and visualization tools
- **Easy Customization**: Modular design for easy adaptation

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{mrp_cslr_2024,
  title={A Multi-Representation Prompt–Guided Framework for Continuous Sign Language Recognition},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

---

**Note**: This repository contains the core implementation. The complete code will be made public after the paper is accepted.
