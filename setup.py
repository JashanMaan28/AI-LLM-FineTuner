"""
Setup script for AI LLM FineTuner.
"""

import os
import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def setup_environment():
    """Set up environment file."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✓ .env file already exists")
        return True
    
    if not env_example.exists():
        print("✗ .env.example file not found")
        return False
    
    # Copy the example file
    try:
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✓ Created .env file from template")
        
        # Ask user for API key
        api_key = input("\nEnter your Groq API key (or press Enter to skip): ").strip()
        if api_key:
            # Update the .env file with the API key
            with open(env_file, 'r') as f:
                content = f.read()
            
            content = content.replace("your_groq_api_key_here", api_key)
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print("✓ API key added to .env file")
        else:
            print("⚠ Skipped API key setup. You'll need to edit .env manually.")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create .env file: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    try:
        Path("datasets").mkdir(exist_ok=True)
        print("✓ Created datasets directory")
        return True
    except Exception as e:
        print(f"✗ Failed to create directories: {e}")
        return False

def run_tests():
    """Run basic tests."""
    print("Running tests...")
    try:
        import test
        # Don't actually run the tests here, just check if imports work
        print("✓ Test module loads correctly")
        return True
    except Exception as e:
        print(f"✗ Test import failed: {e}")
        return False

def main():
    """Main setup function."""
    print("AI LLM FineTuner Setup")
    print("=" * 30)
    
    steps = [
        ("Installing Dependencies", install_dependencies),
        ("Setting up Environment", setup_environment),
        ("Creating Directories", create_directories),
        ("Checking Tests", run_tests),
    ]
    
    success_count = 0
    
    for name, func in steps:
        print(f"\n{name}...")
        if func():
            success_count += 1
        else:
            print(f"Setup step '{name}' failed.")
    
    print(f"\n{'='*30}")
    print(f"Setup completed: {success_count}/{len(steps)} steps successful")
    
    if success_count == len(steps):
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Make sure your Groq API key is in the .env file")
        print("2. Run tests: python test.py")
        print("3. Try examples: python examples.py")
        print("4. Generate your first dataset: python main.py 'Your model description'")
    else:
        print(f"\n⚠ Setup completed with {len(steps) - success_count} issues.")
        print("Please resolve the errors above before proceeding.")

if __name__ == "__main__":
    main()
