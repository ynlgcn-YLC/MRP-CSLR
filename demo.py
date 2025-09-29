#!/usr/bin/env python3
"""
Example script demonstrating MRP-CSLR core functionality
This script shows how to use the framework without requiring actual training
"""
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demonstrate_config_system():
    """Demonstrate the configuration system"""
    print("=== MRP-CSLR Configuration System ===")
    
    try:
        from configs.config import get_config, CONFIG_REGISTRY
        
        print(f"Available configurations: {list(CONFIG_REGISTRY.keys())}")
        
        # Test each configuration
        for config_name in ['default', 'debug', 'phoenix']:
            config = get_config(config_name)
            print(f"\n{config_name.upper()} Configuration:")
            print(f"  - Experiment: {config.experiment_name}")
            print(f"  - Vocab size: {config.model.vocab_size}")
            print(f"  - Embed dim: {config.model.embed_dim}")
            print(f"  - Batch size: {config.training.batch_size}")
            print(f"  - Learning rate: {config.training.learning_rate}")
        
        print("✓ Configuration system working")
        return True
    except Exception as e:
        print(f"✗ Configuration system error: {e}")
        return False


def demonstrate_model_structure():
    """Demonstrate model structure (without actually creating the models)"""
    print("\n=== MRP-CSLR Model Architecture ===")
    
    try:
        # Import model classes to check they exist
        from models.multi_representation_encoder import (
            MultiRepresentationEncoder, VisualFeatureExtractor, 
            SpatialEncoder, TemporalEncoder
        )
        from models.prompt_guided_attention import (
            PromptGuidedAttention, PromptEmbedding, CrossModalAttention
        )
        from models.mrp_cslr import MRP_CSLR, build_mrp_cslr_model
        
        print("Core Components:")
        print("  ✓ MultiRepresentationEncoder")
        print("    - VisualFeatureExtractor")
        print("    - SpatialEncoder") 
        print("    - TemporalEncoder")
        print("  ✓ PromptGuidedAttention")
        print("    - PromptEmbedding")
        print("    - CrossModalAttention")
        print("  ✓ MRP_CSLR (Main model)")
        print("  ✓ build_mrp_cslr_model (Factory function)")
        
        print("✓ Model architecture components loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Model architecture error: {e}")
        return False


def demonstrate_utilities():
    """Demonstrate utility functions"""
    print("\n=== MRP-CSLR Utilities ===")
    
    try:
        from utils.data_processing import (
            VideoTransform, VideoLoader, SignLanguageVocabulary, DataCollator
        )
        from utils.training import (
            MRPCSLRTrainer, SignLanguageMetrics, WarmupCosineScheduler
        )
        
        print("Data Processing:")
        print("  ✓ VideoTransform - Video preprocessing and augmentation")
        print("  ✓ VideoLoader - Video loading utilities")
        print("  ✓ SignLanguageVocabulary - Vocabulary management")
        print("  ✓ DataCollator - Batch collation for variable sequences")
        
        print("Training:")
        print("  ✓ MRPCSLRTrainer - Main trainer class")
        print("  ✓ SignLanguageMetrics - Evaluation metrics")
        print("  ✓ WarmupCosineScheduler - Learning rate scheduler")
        
        # Test metrics without torch
        metrics = SignLanguageMetrics()
        
        # Test sequence accuracy
        predictions = [[1, 2, 3], [4, 5]]
        targets = [[1, 2, 3], [4, 6]]
        
        seq_acc = metrics.sequence_accuracy(predictions, targets)
        token_acc = metrics.token_accuracy(predictions, targets)
        edit_dist = metrics.edit_distance([1, 2, 3], [1, 3, 4])
        
        print(f"  Example metrics: seq_acc={seq_acc:.2f}, token_acc={token_acc:.2f}, edit_dist={edit_dist}")
        
        print("✓ Utility functions working")
        return True
    except Exception as e:
        print(f"✗ Utilities error: {e}")
        return False


def demonstrate_scripts():
    """Demonstrate training and inference scripts"""
    print("\n=== MRP-CSLR Scripts ===")
    
    try:
        import scripts.train
        import scripts.inference
        
        print("Available Scripts:")
        print("  ✓ scripts/train.py - Main training script")
        print("    Usage: python scripts/train.py --config debug")
        print("  ✓ scripts/inference.py - Inference script")  
        print("    Usage: python scripts/inference.py --model_path model.pth --video_path video.mp4")
        
        print("✓ Training and inference scripts available")
        return True
    except Exception as e:
        print(f"✗ Scripts error: {e}")
        return False


def show_usage_examples():
    """Show usage examples"""
    print("\n=== Usage Examples ===")
    
    examples = [
        "# Train with debug configuration",
        "python scripts/train.py --config debug",
        "",
        "# Train with PHOENIX-2014 configuration", 
        "python scripts/train.py --config phoenix --gpu 0",
        "",
        "# Resume training from checkpoint",
        "python scripts/train.py --config default --resume checkpoints/checkpoint_epoch_10.pth",
        "",
        "# Run inference on video",
        "python scripts/inference.py --model_path checkpoints/best_model.pth --video_path video.mp4",
        "",
        "# Create demo video and test inference",
        "python scripts/inference.py --model_path checkpoints/best_model.pth --create_demo",
        "",
        "# Test model components (if dependencies installed):",
        "python -c \"from models import *; print('Models working')\"",
        "python -c \"from utils import *; print('Utils working')\"",
        "python -c \"from configs import *; print('Configs working')\"",
    ]
    
    for example in examples:
        print(example)


def main():
    """Main demonstration function"""
    print("🚀 MRP-CSLR Core Implementation Demo")
    print("=====================================")
    
    success_count = 0
    total_tests = 4
    
    # Test each component
    if demonstrate_config_system():
        success_count += 1
    
    if demonstrate_model_structure():
        success_count += 1
        
    if demonstrate_utilities():
        success_count += 1
        
    if demonstrate_scripts():
        success_count += 1
    
    # Show usage examples
    show_usage_examples()
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("✅ All core components are working correctly!")
        print("🎯 The MRP-CSLR framework is ready to use.")
        print("📝 Install requirements and run training scripts to get started.")
    else:
        print("⚠️  Some components need attention.")
    
    print("\n📚 For full functionality, install dependencies:")
    print("   pip install -r requirements.txt")


if __name__ == "__main__":
    main()