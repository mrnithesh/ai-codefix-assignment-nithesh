from vllm import LLM, SamplingParams
from fastapi import FastAPI
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")

app = FastAPI()

llm = LLM(model=MODEL_NAME)

@app.post("/local_fix")
async def fix_code(code: str):
    
    return "Work in progress"
