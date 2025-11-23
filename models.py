from pydantic import BaseModel

class Input(BaseModel):
    language: str
    cwe: str
    code: str

class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    
class Output(BaseModel):
    fixed_code: str
    diff: str
    explanation: str
    model_used: str
    token_usage: TokenUsage
    latency_ms: int 