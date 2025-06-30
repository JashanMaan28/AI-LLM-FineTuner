"""
AI LLM FineTuner - Dataset generation utilities for fine-tuning LLMs.
"""

__version__ = "1.0.0"
__author__ = "AI LLM FineTuner"

from .generator import DatasetGenerator, generate_training_dataset
from .config import Config

__all__ = ["DatasetGenerator", "generate_training_dataset", "Config"]
