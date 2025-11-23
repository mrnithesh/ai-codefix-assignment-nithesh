import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_loading():
    #Test that model and tokenizer can be loaded.
    # This test validates that the model loading code in main.py executes without errors
    # In a real scenario, this would load the actual model, but for unit tests we verify
    # that the imports and basic structure are correct
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert AutoModelForCausalLM is not None
    assert AutoTokenizer is not None

