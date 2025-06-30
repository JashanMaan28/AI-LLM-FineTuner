"""
Dataset generator for fine-tuning LLMs using Groq API.
Supports both Alpaca and ShareGPT formats.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from groq import Groq
from tqdm import tqdm
import logging

from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generate training datasets for LLM fine-tuning using Groq API."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the dataset generator.
        
        Args:
            api_key: Groq API key (uses config default if None)
            model: Model to use for generation (uses config default if None)
        """
        Config.validate()
        
        self.api_key = api_key or Config.GROQ_API_KEY
        self.model = model or Config.DEFAULT_MODEL
        self.client = Groq(api_key=self.api_key)
        
        # Create output directory
        Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)
        
    def generate_example(self, prompt: str, prev_examples: List[str], temperature: float = 0.5) -> str:
        """
        Generate a single training example using Groq API.
        
        Args:
            prompt: High-level description of the model to train
            prev_examples: Previously generated examples for context
            temperature: Generation temperature (0-1)
            
        Returns:
            Generated example string
        """
        messages = [
            {
                "role": "system",
                "content": f"""You are generating data which will be used to train a machine learning model.

You will be given a high-level description of the model we want to train, and from that, you will generate data samples, each with a prompt/response pair.

You will do so in this format:
```
prompt
-----------
$prompt_goes_here
-----------

response
-----------
$response_goes_here
-----------
```

Only one prompt/response pair should be generated per turn.

For each turn, make the example slightly more complex than the last, while ensuring diversity.

Make sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.

Here is the type of model we want to train:
`{prompt}`"""
            }
        ]
        
        # Add previous examples as context (limit to 10 most recent)
        if prev_examples:
            recent_examples = prev_examples[-10:] if len(prev_examples) > 10 else prev_examples
            for example in recent_examples:
                messages.append({
                    "role": "assistant",
                    "content": example
                })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=1354,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating example: {str(e)}")
            raise
    
    def generate_system_message(self, prompt: str, temperature: float = 0.3) -> str:
        """
        Generate a system message for the fine-tuned model.
        
        Args:
            prompt: High-level description of the model
            temperature: Generation temperature
            
        Returns:
            Generated system message
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You will be given a high-level description of the model we are training, and from that, you will generate a simple system prompt for that model to use. Remember, you are not generating the system message for data generation -- you are generating the system message to use for inference. A good format to follow is `Given $INPUT_DATA, you will $WHAT_THE_MODEL_SHOULD_DO.`.

Make it as concise as possible. Include nothing but the system prompt in your response.

For example, never write: `"$SYSTEM_PROMPT_HERE"`.

It should be like: `$SYSTEM_PROMPT_HERE`."""
                    },
                    {
                        "role": "user",
                        "content": prompt.strip(),
                    }
                ],
                temperature=temperature,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating system message: {str(e)}")
            raise
    
    def parse_examples(self, raw_examples: List[str]) -> List[Dict[str, str]]:
        """
        Parse raw generated examples into structured format.
        
        Args:
            raw_examples: List of raw example strings
            
        Returns:
            List of parsed examples with 'prompt' and 'response' keys
        """
        parsed_examples = []
        
        for example in raw_examples:
            try:
                split_example = example.split('-----------')
                if len(split_example) >= 4:
                    prompt = split_example[1].strip()
                    response = split_example[3].strip()
                    if prompt and response:
                        parsed_examples.append({
                            'prompt': prompt,
                            'response': response
                        })
            except Exception as e:
                logger.warning(f"Failed to parse example: {str(e)}")
                continue
                
        return parsed_examples
    
    def generate_dataset(self, 
                        prompt: str, 
                        num_examples: int = 100, 
                        temperature: float = 0.4,
                        format_type: str = 'alpaca') -> Dict[str, Any]:
        """
        Generate a complete dataset for fine-tuning.
        
        Args:
            prompt: Description of the model to train
            num_examples: Number of examples to generate
            temperature: Generation temperature
            format_type: Dataset format ('alpaca' or 'sharegpt')
            
        Returns:
            Dictionary containing the generated dataset and metadata
        """
        if format_type not in Config.SUPPORTED_FORMATS:
            raise ValueError(f"Format must be one of {Config.SUPPORTED_FORMATS}")
        
        logger.info(f"Generating {num_examples} examples with temperature {temperature}")
        logger.info(f"Model description: {prompt}")
        
        # Generate system message
        logger.info("Generating system message...")
        system_message = self.generate_system_message(prompt, temperature=0.3)
        logger.info(f"System message: {system_message}")
        
        # Generate examples
        raw_examples = []
        for i in tqdm(range(num_examples), desc="Generating examples"):
            try:
                example = self.generate_example(prompt, raw_examples, temperature)
                raw_examples.append(example)
            except Exception as e:
                logger.error(f"Failed to generate example {i}: {str(e)}")
                continue
        
        # Parse examples
        parsed_examples = self.parse_examples(raw_examples)
        logger.info(f"Successfully parsed {len(parsed_examples)} examples")
        
        if not parsed_examples:
            raise ValueError("No valid examples were generated")
        
        # Format dataset
        if format_type == 'alpaca':
            dataset = self._format_alpaca(parsed_examples, system_message)
        else:  # sharegpt
            dataset = self._format_sharegpt(parsed_examples, system_message)
        
        return {
            'dataset': dataset,
            'system_message': system_message,
            'metadata': {
                'num_examples': len(dataset),
                'format': format_type,
                'model_used': self.model,
                'temperature': temperature,
                'prompt': prompt
            }
        }
    
    def _format_alpaca(self, examples: List[Dict[str, str]], system_message: str) -> List[Dict[str, str]]:
        """Format examples in Alpaca format."""
        alpaca_data = []
        
        for example in examples:
            alpaca_data.append({
                "instruction": example['prompt'],
                "input": "",
                "output": example['response'],
                "system": system_message
            })
        
        return alpaca_data
    
    def _format_sharegpt(self, examples: List[Dict[str, str]], system_message: str) -> List[Dict[str, Any]]:
        """Format examples in ShareGPT format."""
        sharegpt_data = []
        
        for example in examples:
            sharegpt_data.append({
                "conversations": [
                    {
                        "from": "system",
                        "value": system_message
                    },
                    {
                        "from": "human",
                        "value": example['prompt']
                    },
                    {
                        "from": "gpt",
                        "value": example['response']
                    }
                ]
            })
        
        return sharegpt_data
    
    def save_dataset(self, 
                    dataset_info: Dict[str, Any], 
                    filename_prefix: str = "dataset",
                    train_split: float = 0.9) -> Dict[str, str]:
        """
        Save the generated dataset to files.
        
        Args:
            dataset_info: Dataset information from generate_dataset
            filename_prefix: Prefix for output files
            train_split: Fraction of data for training (rest for validation)
            
        Returns:
            Dictionary with paths to saved files
        """
        dataset = dataset_info['dataset']
        format_type = dataset_info['metadata']['format']
        
        # Shuffle and split dataset
        shuffled_data = dataset.copy()
        random.shuffle(shuffled_data)
        
        split_idx = int(len(shuffled_data) * train_split)
        train_data = shuffled_data[:split_idx]
        val_data = shuffled_data[split_idx:]
        
        # Create output paths
        train_path = Path(Config.OUTPUT_DIR) / f"{filename_prefix}_{format_type}_train.jsonl"
        val_path = Path(Config.OUTPUT_DIR) / f"{filename_prefix}_{format_type}_val.jsonl"
        metadata_path = Path(Config.OUTPUT_DIR) / f"{filename_prefix}_{format_type}_metadata.json"
        
        # Save datasets
        self._save_jsonl(train_data, train_path)
        self._save_jsonl(val_data, val_path)
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info['metadata'], f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved:")
        logger.info(f"  Training: {train_path} ({len(train_data)} examples)")
        logger.info(f"  Validation: {val_path} ({len(val_data)} examples)")
        logger.info(f"  Metadata: {metadata_path}")
        
        return {
            'train': str(train_path),
            'validation': str(val_path),
            'metadata': str(metadata_path)
        }
    
    def _save_jsonl(self, data: List[Dict], filepath: Path):
        """Save data in JSONL format."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def generate_training_dataset(prompt: str,
                            num_examples: int = 100,
                            temperature: float = 0.4,
                            format_type: str = 'alpaca',
                            filename_prefix: str = 'dataset',
                            api_key: Optional[str] = None,
                            model: Optional[str] = None) -> Dict[str, str]:
    """
    High-level function to generate and save a training dataset.
    
    Args:
        prompt: Description of the model you want to train
        num_examples: Number of examples to generate
        temperature: Generation temperature (0-1, lower=more focused)
        format_type: 'alpaca' or 'sharegpt'
        filename_prefix: Prefix for output files
        api_key: Groq API key (optional, uses config if None)
        model: Model to use (optional, uses config if None)
        
    Returns:
        Dictionary with paths to generated files
        
    Example:
        >>> files = generate_training_dataset(
        ...     prompt="A model that explains complex scientific concepts in simple terms",
        ...     num_examples=50,
        ...     temperature=0.5,
        ...     format_type='alpaca'
        ... )
        >>> print(f"Training data saved to: {files['train']}")
    """
    generator = DatasetGenerator(api_key=api_key, model=model)
    
    # Generate dataset
    dataset_info = generator.generate_dataset(
        prompt=prompt,
        num_examples=num_examples,
        temperature=temperature,
        format_type=format_type
    )
    
    # Save dataset
    file_paths = generator.save_dataset(dataset_info, filename_prefix)
    
    return file_paths


if __name__ == "__main__":
    # Example usage
    example_prompt = """
    A helpful AI assistant that can answer questions about machine learning concepts, 
    explain algorithms, and provide coding examples in Python. The assistant should be 
    able to break down complex topics into understandable explanations suitable for 
    both beginners and intermediate learners.
    """
    
    try:
        files = generate_training_dataset(
            prompt=example_prompt,
            num_examples=20,  # Small number for testing
            temperature=0.4,
            format_type='alpaca',
            filename_prefix='ml_assistant'
        )
        
        print("Dataset generation completed successfully!")
        print(f"Files created: {files}")
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {str(e)}")
