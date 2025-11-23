import sys
import os
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from models import Output

client = TestClient(app)

@patch("main.model")
@patch("main.tokenizer")
def test_api_response_schema(mock_tokenizer, mock_model):
    #Test API response schema validation for /local_fix endpoint.
    # Mock tokenizer
    mock_tokenizer.apply_chat_template.return_value = "mocked_prompt"
    mock_tokenizer.return_value = {"input_ids": MagicMock(shape=(1, 10))}
    mock_tokenizer.decode.return_value = """[FIXED_CODE_START]
def secure_code():
    pass
[FIXED_CODE_END]
[EXPLANATION_START]
This is a secure fix.
[EXPLANATION_END]"""
    
    # Mock model
    mock_outputs = MagicMock()
    mock_outputs.shape = (1, 20)
    mock_model.generate.return_value = mock_outputs
    mock_model.parameters.return_value = iter([MagicMock(device="cpu")])
    
    # Test request
    payload = {
        "language": "python",
        "cwe": "CWE-89",
        "code": "import sqlite3\nconn = sqlite3.connect('db')\ncursor = conn.cursor()\ncursor.execute('SELECT * FROM users WHERE name = ' + username)"
    }
    
    response = client.post("/local_fix", json=payload)
    
    # Validate response schema
    assert response.status_code == 200
    data = response.json()
    
    # Validate all required fields exist
    assert "fixed_code" in data
    assert "diff" in data
    assert "explanation" in data
    assert "model_used" in data
    assert "token_usage" in data
    assert "latency_ms" in data
    
    # Validate token_usage structure
    assert "input_tokens" in data["token_usage"]
    assert "output_tokens" in data["token_usage"]
    
    # Validate response matches Output model schema
    output = Output(**data)
    assert output.fixed_code is not None
    assert output.explanation is not None
    assert isinstance(output.latency_ms, int)
    assert isinstance(output.token_usage.input_tokens, int)
    assert isinstance(output.token_usage.output_tokens, int)
