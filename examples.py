"""
Example usage of the AI LLM FineTuner dataset generator.
"""

from generator import generate_training_dataset

# Example 1: Generate a dataset for a code explanation model
def example_code_explainer():
    """Generate dataset for a model that explains code."""
    prompt = """
    A helpful programming assistant that can explain code snippets, debug issues, 
    and provide coding examples in multiple programming languages. The assistant 
    should be able to break down complex algorithms, explain programming concepts, 
    and help with code optimization. Focus on Python, JavaScript, and SQL.
    """
    
    files = generate_training_dataset(
        prompt=prompt,
        num_examples=50,
        temperature=0.3,  # Lower temperature for more focused, technical responses
        format_type='alpaca',
        filename_prefix='code_explainer'
    )
    
    print(f"Code explainer dataset created: {files}")


# Example 2: Generate a dataset for a creative writing assistant
def example_creative_writer():
    """Generate dataset for a creative writing model."""
    prompt = """
    A creative writing assistant that helps users with storytelling, character 
    development, plot creation, and writing techniques. The assistant should be 
    able to generate story ideas, provide writing prompts, give feedback on 
    creative pieces, and explain literary devices. Focus on fiction writing, 
    poetry, and creative non-fiction.
    """
    
    files = generate_training_dataset(
        prompt=prompt,
        num_examples=75,
        temperature=0.7,  # Higher temperature for more creative responses
        format_type='sharegpt',
        filename_prefix='creative_writer'
    )
    
    print(f"Creative writer dataset created: {files}")


# Example 3: Generate a dataset for a math tutor
def example_math_tutor():
    """Generate dataset for a math tutoring model."""
    prompt = """
    A patient and knowledgeable math tutor that helps students understand 
    mathematical concepts from basic arithmetic to advanced calculus. The tutor 
    should be able to solve problems step-by-step, explain mathematical reasoning, 
    provide practice problems, and adapt explanations to different learning styles. 
    Cover topics including algebra, geometry, trigonometry, statistics, and calculus.
    """
    
    files = generate_training_dataset(
        prompt=prompt,
        num_examples=100,
        temperature=0.2,  # Very low temperature for precise mathematical explanations
        format_type='alpaca',
        filename_prefix='math_tutor'
    )
    
    print(f"Math tutor dataset created: {files}")


# Example 4: Generate a dataset for a general knowledge assistant
def example_general_knowledge():
    """Generate dataset for a general knowledge model."""
    prompt = """
    A knowledgeable assistant that can answer questions across various domains 
    including science, history, geography, culture, current events, and general 
    trivia. The assistant should provide accurate, informative responses while 
    being engaging and easy to understand. Include explanations of complex topics 
    in simple terms and provide interesting facts and context.
    """
    
    files = generate_training_dataset(
        prompt=prompt,
        num_examples=80,
        temperature=0.5,  # Moderate temperature for balanced responses
        format_type='sharegpt',
        filename_prefix='general_knowledge'
    )
    
    print(f"General knowledge dataset created: {files}")


if __name__ == "__main__":
    print("AI LLM FineTuner - Example Dataset Generation")
    print("=" * 50)
    
    examples = [
        ("Code Explainer", example_code_explainer),
        ("Creative Writer", example_creative_writer),
        ("Math Tutor", example_math_tutor),
        ("General Knowledge", example_general_knowledge)
    ]
    
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    try:
        choice = input("\nEnter the number of the example to run (or 'all' for all): ").strip()
        
        if choice.lower() == 'all':
            for name, func in examples:
                print(f"\nGenerating {name} dataset...")
                func()
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                name, func = examples[idx]
                print(f"\nGenerating {name} dataset...")
                func()
            else:
                print("Invalid choice!")
                
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nMake sure you have set up your .env file with GROQ_API_KEY!")
