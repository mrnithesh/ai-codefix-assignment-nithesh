from pydantic import BaseModel

class Input(BaseModel):
    language: str
    cwe: str
    code: str

class Output(BaseModel):
    fixed_code: str
    diff: str
    explanation: str
    model_used: str
    token_usage: dict[str, int]
    latency_ms: int 