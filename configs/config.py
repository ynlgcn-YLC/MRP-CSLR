"""
Configuration file for MRP-CSLR training
Contains default hyperparameters and model configurations
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vocab_size: int = 1000
    visual_backbone: str = 'resnet50'
    feature_dim: int = 2048
    embed_dim: int = 512
    num_heads: int = 8
    num_prompts: int = 100
    dropout: float = 0.1
    use_ctc_loss: bool = True


@dataclass
class DataConfig:
    """Data processing configuration"""
    input_size: Tuple[int, int] = (224, 224)
    max_sequence_length: int = 512
    fps: int = 25
    augment: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-3
    backbone_lr: float = 1e-4
    attention_lr: float = 1e-3
    classifier_lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    optimizer: str = 'adamw'  # 'adamw', 'adam', 'sgd'
    
    # Validation and checkpointing
    val_interval: int = 5
    checkpoint_interval: int = 10
    log_interval: int = 50
    
    # Paths
    data_dir: str = './data'
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    vocab_file: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Full experiment configuration"""
    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    
    # Experiment metadata
    experiment_name: str = 'mrp_cslr_baseline'
    description: str = 'MRP-CSLR baseline experiment'
    seed: int = 42
    
    def __post_init__(self):
        """Initialize sub-configs if not provided"""
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
    
    def to_dict(self):
        """Convert to dictionary for logging"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'experiment_name': self.experiment_name,
            'description': self.description,
            'seed': self.seed
        }


# Predefined configurations for different datasets and scenarios

def get_phoenix_config() -> ExperimentConfig:
    """Configuration for PHOENIX-2014 dataset"""
    config = ExperimentConfig()
    config.experiment_name = 'mrp_cslr_phoenix'
    config.description = 'MRP-CSLR on PHOENIX-2014 dataset'
    
    # Dataset-specific settings
    config.model.vocab_size = 1295  # PHOENIX vocabulary size
    config.data.input_size = (224, 224)
    config.data.max_sequence_length = 300
    config.training.batch_size = 4  # Adjust based on GPU memory
    
    return config


def get_csl_daily_config() -> ExperimentConfig:
    """Configuration for CSL-Daily dataset"""
    config = ExperimentConfig()
    config.experiment_name = 'mrp_cslr_csl_daily'
    config.description = 'MRP-CSLR on CSL-Daily dataset'
    
    # Dataset-specific settings
    config.model.vocab_size = 2000  # Adjust based on CSL-Daily vocabulary
    config.data.input_size = (224, 224)
    config.data.max_sequence_length = 400
    config.training.batch_size = 6
    
    return config


def get_debug_config() -> ExperimentConfig:
    """Configuration for debugging and quick testing"""
    config = ExperimentConfig()
    config.experiment_name = 'mrp_cslr_debug'
    config.description = 'Debug configuration for MRP-CSLR'
    
    # Small settings for quick testing
    config.model.vocab_size = 100
    config.model.embed_dim = 256
    config.model.num_prompts = 50
    config.data.max_sequence_length = 50
    config.training.num_epochs = 5
    config.training.batch_size = 2
    config.training.val_interval = 2
    config.training.checkpoint_interval = 3
    
    return config


def get_large_config() -> ExperimentConfig:
    """Configuration for large-scale experiments"""
    config = ExperimentConfig()
    config.experiment_name = 'mrp_cslr_large'
    config.description = 'Large-scale MRP-CSLR experiment'
    
    # Large model settings
    config.model.visual_backbone = 'efficientnet_b4'
    config.model.feature_dim = 1792
    config.model.embed_dim = 768
    config.model.num_heads = 12
    config.model.num_prompts = 200
    config.data.input_size = (288, 288)
    config.training.num_epochs = 200
    config.training.warmup_epochs = 10
    
    return config


# Configuration registry
CONFIG_REGISTRY = {
    'default': ExperimentConfig,
    'phoenix': get_phoenix_config,
    'csl_daily': get_csl_daily_config,
    'debug': get_debug_config,
    'large': get_large_config
}


def get_config(config_name: str = 'default') -> ExperimentConfig:
    """Get configuration by name"""
    if config_name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIG_REGISTRY.keys())}")
    
    config_fn = CONFIG_REGISTRY[config_name]
    return config_fn() if callable(config_fn) else config_fn


if __name__ == "__main__":
    # Test configurations
    print("Testing configurations...")
    
    for config_name in CONFIG_REGISTRY.keys():
        print(f"\n=== {config_name.upper()} CONFIG ===")
        config = get_config(config_name)
        print(f"Experiment: {config.experiment_name}")
        print(f"Vocab size: {config.model.vocab_size}")
        print(f"Embed dim: {config.model.embed_dim}")
        print(f"Batch size: {config.training.batch_size}")
        print(f"Num epochs: {config.training.num_epochs}")
    
    print("\nConfiguration test passed!")