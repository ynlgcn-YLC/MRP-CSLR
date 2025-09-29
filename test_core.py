#!/usr/bin/env python3
"""
Core functionality test without heavy dependencies
Tests only the configuration system and basic imports
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_system():
    """Test configuration system"""
    print("Testing configuration system...")
    try:
        from configs.config import get_config, CONFIG_REGISTRY
        
        print(f"Available configurations: {list(CONFIG_REGISTRY.keys())}")
        
        # Test basic config loading
        config = get_config('debug')
        print(f"Debug config loaded: {config.experiment_name}")
        print(f"  - Vocab size: {config.model.vocab_size}")
        print(f"  - Embed dim: {config.model.embed_dim}")
        
        config = get_config('default')  
        print(f"Default config loaded: {config.experiment_name}")
        
        print("✅ Configuration system working")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_imports():
    """Test that all modules can be imported (syntax check)"""
    print("Testing module imports...")
    
    modules_to_test = [
        'configs',
        'configs.config'
    ]
    
    success = 0
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
            success += 1
        except Exception as e:
            print(f"❌ {module}: {e}")
    
    print(f"Import test: {success}/{len(modules_to_test)} successful")
    return success == len(modules_to_test)

def test_file_structure():
    """Test that all expected files exist"""
    print("Testing file structure...")
    
    required_files = [
        'requirements.txt',
        'README.md',
        'models/__init__.py',
        'models/multi_representation_encoder.py',
        'models/prompt_guided_attention.py', 
        'models/mrp_cslr.py',
        'utils/__init__.py',
        'utils/data_processing.py',
        'utils/training.py',
        'configs/__init__.py',
        'configs/config.py',
        'scripts/train.py',
        'scripts/inference.py'
    ]
    
    missing = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing:
        print(f"❌ Missing files: {missing}")
        return False
    else:
        print("✅ All required files present")
        return True

def main():
    """Run all tests"""
    print("🧪 MRP-CSLR Core Tests")
    print("======================")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Basic Imports", test_imports),
        ("Configuration System", test_config_system),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        success = test_func()
        results.append(success)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*30}")
    print(f"SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core tests passed!")
        print("📦 MRP-CSLR framework is properly structured")
        print("⚡ Ready for dependency installation and training")
    else:
        print("⚠️  Some tests failed - check the errors above")
    
    print("\n📋 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run training: python scripts/train.py --config debug")
    print("3. Run inference: python scripts/inference.py --model_path model.pth --create_demo")

if __name__ == "__main__":
    main()