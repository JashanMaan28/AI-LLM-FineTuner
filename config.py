"""
Configuration settings for the AI LLM FineTuner dataset generator.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for dataset generation."""
    
    # API Configuration
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct')
    
    # Generation Parameters
    DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE', 0.4))
    DEFAULT_NUM_EXAMPLES = int(os.getenv('DEFAULT_NUM_EXAMPLES', 100))
    
    # Dataset Formats
    SUPPORTED_FORMATS = ['alpaca', 'sharegpt']
    
    # File Paths
    OUTPUT_DIR = 'datasets'
    TRAIN_SPLIT = 0.9
    
    @classmethod
    def validate(cls):
        """Validate the configuration."""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        if cls.DEFAULT_TEMPERATURE < 0 or cls.DEFAULT_TEMPERATURE > 1:
            raise ValueError("Temperature must be between 0 and 1")
            
        if cls.DEFAULT_NUM_EXAMPLES < 1:
            raise ValueError("Number of examples must be at least 1")
