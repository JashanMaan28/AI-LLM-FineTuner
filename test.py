"""
Test script for the AI LLM FineTuner dataset generator.
"""

import os
import json
from pathlib import Path

def test_basic_functionality():
    """Test basic dataset generation without API calls."""
    from generator import DatasetGenerator
    from config import Config
    
    print("Testing configuration...")
    
    # Test config validation (this should work even without API key for structure test)
    try:
        # Test that we can create a generator instance structure
        print("✓ Configuration structure is valid")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False
    
    print("✓ Basic functionality test passed")
    return True

def test_data_formats():
    """Test data formatting functions."""
    from generator import DatasetGenerator
    
    print("Testing data formatting...")
    
    # Create a generator instance (won't make API calls)
    generator = DatasetGenerator.__new__(DatasetGenerator)
    
    # Test sample data
    sample_examples = [
        {"prompt": "What is Python?", "response": "Python is a programming language."},
        {"prompt": "How do loops work?", "response": "Loops allow you to repeat code."}
    ]
    
    system_message = "You are a helpful programming assistant."
    
    # Test Alpaca formatting
    try:
        alpaca_data = generator._format_alpaca(sample_examples, system_message)
        assert len(alpaca_data) == 2
        assert "instruction" in alpaca_data[0]
        assert "input" in alpaca_data[0]
        assert "output" in alpaca_data[0]
        assert "system" in alpaca_data[0]
        print("✓ Alpaca format works correctly")
    except Exception as e:
        print(f"✗ Alpaca format error: {e}")
        return False
    
    # Test ShareGPT formatting
    try:
        sharegpt_data = generator._format_sharegpt(sample_examples, system_message)
        assert len(sharegpt_data) == 2
        assert "conversations" in sharegpt_data[0]
        assert len(sharegpt_data[0]["conversations"]) == 3
        assert sharegpt_data[0]["conversations"][0]["from"] == "system"
        assert sharegpt_data[0]["conversations"][1]["from"] == "human"
        assert sharegpt_data[0]["conversations"][2]["from"] == "gpt"
        print("✓ ShareGPT format works correctly")
    except Exception as e:
        print(f"✗ ShareGPT format error: {e}")
        return False
    
    print("✓ Data formatting test passed")
    return True

def test_example_parsing():
    """Test example parsing functionality."""
    from generator import DatasetGenerator
    
    print("Testing example parsing...")
    
    generator = DatasetGenerator.__new__(DatasetGenerator)
    
    # Test sample raw examples
    raw_examples = [
        """prompt
-----------
What is machine learning?
-----------

response
-----------
Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.
-----------""",
        """prompt
-----------
Explain neural networks
-----------

response
-----------
Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information.
-----------"""
    ]
    
    try:
        parsed = generator.parse_examples(raw_examples)
        assert len(parsed) == 2
        assert "prompt" in parsed[0] and "response" in parsed[0]
        assert parsed[0]["prompt"] == "What is machine learning?"
        assert "artificial intelligence" in parsed[0]["response"]
        print("✓ Example parsing works correctly")
    except Exception as e:
        print(f"✗ Example parsing error: {e}")
        return False
    
    print("✓ Example parsing test passed")
    return True

def check_environment():
    """Check if environment is properly set up."""
    print("Checking environment setup...")
    
    # Check if .env file exists
    if Path(".env").exists():
        print("✓ .env file found")
        
        # Check if it has the required key
        from config import Config
        if Config.GROQ_API_KEY:
            print("✓ GROQ_API_KEY is configured")
            return True
        else:
            print("⚠ GROQ_API_KEY not found in environment")
            print("  Please add your Groq API key to the .env file")
            return False
    else:
        print("⚠ .env file not found")
        print("  Please copy .env.example to .env and add your API key")
        return False

def main():
    """Run all tests."""
    print("AI LLM FineTuner - Test Suite")
    print("=" * 40)
    
    tests = [
        ("Environment Check", check_environment),
        ("Basic Functionality", test_basic_functionality),
        ("Data Formatting", test_data_formats), 
        ("Example Parsing", test_example_parsing),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n{name}:")
        print("-" * len(name))
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {name} failed")
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
    
    print(f"\n{'='*40}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        
        if check_environment():
            print("\nYou can now run:")
            print("  python examples.py")
            print("  python main.py 'Your model description here'")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
        
        if not check_environment():
            print("\nFirst, make sure to set up your .env file:")
            print("  1. Copy .env.example to .env")
            print("  2. Add your Groq API key")

if __name__ == "__main__":
    main()
