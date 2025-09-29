#!/usr/bin/env python3
"""
Main training script for MRP-CSLR
Multi-Representation Prompt–Guided Framework for Continuous Sign Language Recognition
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_mrp_cslr_model
from utils.training import MRPCSLRTrainer
from utils.data_processing import VideoTransform, SignLanguageVocabulary, DataCollator
from configs.config import get_config


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DummySignLanguageDataset(Dataset):
    """
    Dummy dataset for demonstration purposes
    In practice, this would be replaced with actual sign language dataset
    """
    def __init__(self, num_samples=1000, max_sequence_length=100, vocab_size=1000, transform=None):
        self.num_samples = num_samples
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.transform = transform
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate dummy video frames (T, H, W, C)
        seq_len = random.randint(20, self.max_sequence_length)
        frames = np.random.randint(0, 255, (seq_len, 224, 224, 3), dtype=np.uint8)
        
        # Apply transforms if provided
        if self.transform:
            frames = self.transform(frames)  # Returns (T, C, H, W)
        else:
            # Convert to tensor format
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        
        # Generate dummy labels
        label_len = random.randint(5, seq_len // 4)
        labels = [random.randint(3, self.vocab_size - 1) for _ in range(label_len)]  # Start from 3 (skip special tokens)
        
        return {
            'video': frames,
            'labels': labels,
            'video_id': f'dummy_{idx}'
        }


def create_data_loaders(config):
    """Create training and validation data loaders"""
    # Create transforms
    train_transform = VideoTransform(
        input_size=config.data.input_size,
        mean=config.data.mean,
        std=config.data.std,
        augment=True
    )
    
    val_transform = VideoTransform(
        input_size=config.data.input_size,
        mean=config.data.mean,
        std=config.data.std,
        augment=False
    )
    
    # Create datasets (dummy for demonstration)
    train_dataset = DummySignLanguageDataset(
        num_samples=800,
        max_sequence_length=config.data.max_sequence_length,
        vocab_size=config.model.vocab_size,
        transform=train_transform
    )
    
    val_dataset = DummySignLanguageDataset(
        num_samples=200,
        max_sequence_length=config.data.max_sequence_length,
        vocab_size=config.model.vocab_size,
        transform=val_transform
    )
    
    # Create data collator
    collator = DataCollator(pad_token_id=1)  # 1 is pad token
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train MRP-CSLR model')
    parser.add_argument('--config', type=str, default='default', 
                       help='Configuration name (default, phoenix, csl_daily, debug, large)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU ID to use (default: use all available)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with minimal settings')
    
    args = parser.parse_args()
    
    # Get configuration
    if args.debug:
        config = get_config('debug')
    else:
        config = get_config(args.config)
    
    print(f"=== MRP-CSLR Training ===")
    print(f"Experiment: {config.experiment_name}")
    print(f"Description: {config.description}")
    print(f"Configuration: {args.config}")
    
    # Set random seeds
    set_random_seeds(config.seed)
    
    # Set GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Create directories
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = build_mrp_cslr_model(
        vocab_size=config.model.vocab_size,
        visual_backbone=config.model.visual_backbone,
        feature_dim=config.model.feature_dim,
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        num_prompts=config.model.num_prompts,
        dropout=config.model.dropout,
        use_ctc_loss=config.model.use_ctc_loss
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer_config = config.training.to_dict()
    trainer = MRPCSLRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()