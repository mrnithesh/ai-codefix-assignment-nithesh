from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from models import Input, Output, TokenUsage
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import difflib
import json
import re
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory for models
CACHE_DIR = "D:/ai_models_cache" 

os.makedirs(CACHE_DIR, exist_ok=True)

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")

app = FastAPI()

try:
    logger.info(f"Loading model: {MODEL_NAME}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    if cuda_available:
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        cache_dir=CACHE_DIR,
        device_map="auto"
    )
    
    # Log device information after loading
    try:
        device_info = {}
        if hasattr(model, 'hf_device_map'):
            device_info = model.hf_device_map
            logger.info(f"Model device map: {device_info}")
        else:
            # Try to get device from model parameters
            try:
                device = next(model.parameters()).device
                logger.info(f"Model device: {device}")
                if device.type == 'cuda':
                    logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
                    logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
            except StopIteration:
                logger.warning("Could not determine model device")
    except Exception as e:
        logger.warning(f"Could not get device info: {str(e)}")
    
    logger.info(f"Model loaded successfully")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    logger.info(f"Tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "AI Code Fix API is running",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "fix_code": "/local_fix (POST)"
        }
    }

@app.post("/local_fix", response_model=Output)
async def fix_code(input: Input):
    logger.info(f"Received request to fix code: {input}")
    try:
        system_prompt = """You are an expert security code remediation assistant specializing in fixing software vulnerabilities.

Your task is to analyze vulnerable code and provide secure fixes. You must respond ONLY with valid JSON in the following format:
{
    "fixed_code": "<the complete fixed code snippet>",
    "explanation": "<a clear explanation of the vulnerability and how the fix addresses it>"
}

Important:
- Return ONLY the JSON object, no markdown code blocks, no additional text
- The fixed_code must be complete and syntactically correct
- The explanation should reference the CWE and explain the security issue"""
        
        user_message = f"""Language: {input.language}
CWE: {input.cwe}

Vulnerable Code:
```{input.language}
{input.code}
```

Please provide the fixed code and explanation in JSON format."""
        
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        
        try:
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logger.info(f"Inputs moved to device: {device}")
            if device.type == 'cuda':
                logger.info(f"GPU memory before generation - Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB, Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        except (StopIteration, AttributeError):
            pass
        
        start_time = time.time()
        logger.info(f"Generating response...")
        # Generate response
        outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
        
        # Log GPU memory after generation
        try:
            device = next(model.parameters()).device
            if device.type == 'cuda':
                logger.info(f"GPU memory after generation - Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB, Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        except (StopIteration, AttributeError):
            pass
        logger.info(f"Response generated successfully")
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        logger.info(f"Latency: {latency_ms} ms")
        input_length = inputs["input_ids"].shape[1]
        output_length = outputs.shape[1]
        output_tokens = output_length - input_length
        logger.info(f"Input tokens: {input_length}, Output tokens: {output_tokens}")
        generated_tokens = outputs[0][input_length:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        logger.info(f"Raw model response: {response_text}")
        print(f"Raw model response: {response_text}")
        
        # Parse JSON from response
        json_text = response_text.strip()
        # Remove markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            # Try to extract JSON object directly
            json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
        # Parse JSON
        try:
            parsed_response = json.loads(json_text)
            fixed_code = parsed_response.get("fixed_code", "")
            explanation = parsed_response.get("explanation", "")
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse JSON from model response: {str(e)}. Raw response: {response_text[:200]}"
            )
        logger.info(f"JSON parsed successfully")
        # Generate diff using difflib
        original_lines = input.code.splitlines(keepends=True)
        fixed_lines = fixed_code.splitlines(keepends=True)
        diff = ''.join(difflib.unified_diff(
            original_lines,
            fixed_lines,
            fromfile='original',
            tofile='fixed',
            lineterm=''
        ))
        logger.info(f"Diff generated successfully")
        # Create TokenUsage model instance
        token_usage = TokenUsage(
            input_tokens=input_length,
            output_tokens=output_tokens
        )
        return Output(
            fixed_code=fixed_code,
            diff=diff,
            explanation=explanation,
            model_used=MODEL_NAME or "unknown",
            token_usage=token_usage,
            latency_ms=latency_ms
        )
        
    except HTTPException as e:
        logger.error(f"HTTPException: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Exception: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
