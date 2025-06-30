"""
Main entry point for the AI LLM FineTuner dataset generator.
"""

import argparse
import sys
from pathlib import Path

from generator import generate_training_dataset
from config import Config


def main():
    """Main function to run the dataset generator."""
    parser = argparse.ArgumentParser(
        description="Generate training datasets for LLM fine-tuning using Groq API"
    )
    
    parser.add_argument(
        "prompt",
        help="Description of the model you want to train"
    )
    
    parser.add_argument(
        "-n", "--num-examples",
        type=int,
        default=Config.DEFAULT_NUM_EXAMPLES,
        help=f"Number of examples to generate (default: {Config.DEFAULT_NUM_EXAMPLES})"
    )
    
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=Config.DEFAULT_TEMPERATURE,
        help=f"Generation temperature 0-1 (default: {Config.DEFAULT_TEMPERATURE})"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=Config.SUPPORTED_FORMATS,
        default="alpaca",
        help="Dataset format (default: alpaca)"
    )
    
    parser.add_argument(
        "-o", "--output-prefix",
        default="dataset",
        help="Output filename prefix (default: dataset)"
    )
    
    parser.add_argument(
        "-m", "--model",
        default=Config.DEFAULT_MODEL,
        help=f"Model to use for generation (default: {Config.DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--api-key",
        help="Groq API key (uses environment variable if not provided)"
    )
    
    args = parser.parse_args()
    
    # Validate temperature
    if not 0 <= args.temperature <= 1:
        print("Error: Temperature must be between 0 and 1")
        sys.exit(1)
    
    # Validate number of examples
    if args.num_examples < 1:
        print("Error: Number of examples must be at least 1")
        sys.exit(1)
    
    try:
        print(f"Generating {args.num_examples} examples...")
        print(f"Model: {args.model}")
        print(f"Format: {args.format}")
        print(f"Temperature: {args.temperature}")
        print(f"Prompt: {args.prompt}")
        print("-" * 50)
        
        files = generate_training_dataset(
            prompt=args.prompt,
            num_examples=args.num_examples,
            temperature=args.temperature,
            format_type=args.format,
            filename_prefix=args.output_prefix,
            api_key=args.api_key,
            model=args.model
        )
        
        print("\n" + "=" * 50)
        print("Dataset generation completed successfully!")
        print("=" * 50)
        print(f"Training data: {files['train']}")
        print(f"Validation data: {files['validation']}")
        print(f"Metadata: {files['metadata']}")
        
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
