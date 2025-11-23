from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from models import Input, Output, TokenUsage
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import difflib
import json
import re

# Cache directory for models
CACHE_DIR = "D:/ai_models_cache" 

os.makedirs(CACHE_DIR, exist_ok=True)

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")

app = FastAPI()

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        cache_dir=CACHE_DIR,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
except Exception as e:
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
        except (StopIteration, AttributeError):
            pass
        
        start_time = time.time()
        # Generate response
        outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
        
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        
        input_length = inputs["input_ids"].shape[1]
        output_length = outputs.shape[1]
        output_tokens = output_length - input_length
        
        generated_tokens = outputs[0][input_length:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
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
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
