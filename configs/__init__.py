"""
Configuration package for MRP-CSLR
Contains model and training configurations
"""

from .config import (
    ModelConfig,
    DataConfig, 
    TrainingConfig,
    ExperimentConfig,
    get_config,
    get_phoenix_config,
    get_csl_daily_config,
    get_debug_config,
    get_large_config
)

__all__ = [
    'ModelConfig',
    'DataConfig',
    'TrainingConfig', 
    'ExperimentConfig',
    'get_config',
    'get_phoenix_config',
    'get_csl_daily_config',
    'get_debug_config',
    'get_large_config'
]