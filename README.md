# AI Code Remediation Microservice

This project is an AI-powered microservice designed to analyze vulnerable code snippets and provide secure fixes using a local Large Language Model (LLM). It features a RAG (Retrieval-Augmented Generation) system to contextually enhance the model's output with security guidelines.

## Features

*   **Local Inference**: Uses `Qwen2.5-Coder-1.5B-Instruct` (configurable) running locally via Hugging Face Transformers.
*   **RAG Integration**: Retrieves relevant CWE (Common Weakness Enumeration) guidelines using `SentenceTransformers` and `FAISS` to improve fix accuracy.
*   **FastAPI Backend**: Exposes a RESTful endpoint `/local_fix` for code remediation.
*   **Observability**: Logs latency, token usage, and RAG retrieval matches.

## Setup Instructions

### Prerequisites

*   Python 3.10+
*   Git
*   (Optional) NVIDIA GPU with CUDA support for faster inference.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/mrnithesh/ai-codefix-assignment-nithesh.git
    cd ai-codefix-assignment-nithesh
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**:
    Create a `.env` file in the root directory:
    ```
    MODEL_NAME=Qwen/Qwen2.5-Coder-1.5B-Instruct
    ```

## Usage

### Running the Server

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### API Endpoint: `/local_fix`

**Method**: `POST`

**Request Body**:
```json
{
  "language": "python",
  "cwe": "CWE-89",
  "code": "import sqlite3\nconn = sqlite3.connect('db')\ncursor = conn.cursor()\ncursor.execute('SELECT * FROM users WHERE name = ' + username)"
}
```

**Response**:
```json
{
  "fixed_code": "...",
  "diff": "...",
  "explanation": "...",
  "model_used": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
  "token_usage": {
    "input_tokens": 120,
    "output_tokens": 85
  },
  "latency_ms": 1500
}
```

### Running Tests

**Integration Script**:
Run the provided test script to send sample vulnerabilities to the running server:
```bash
python test_local.py
```

**Unit Tests**:
Run the test suite using `pytest`:
```bash
pytest
```

## Docker Usage

You can containerize the application for consistent deployment.

1.  **Build the Docker image**:
    ```bash
    docker build -t ai-codefix .
    ```

2.  **Run the container**:
    ```bash
    docker run -p 8000:8000 ai-codefix
    ```

## Design & Observations

### RAG Implementation
The system uses a "Mini-RAG" approach:
*   **Indexing**: Security recipes (text files in `recipes/`) are embedded using `sentence-transformers/all-MiniLM-L6-v2` and indexed with `FAISS`.
*   **Retrieval**: Incoming requests are converted to a query combining the CWE ID, language, and a snippet of the code. This hybrid query ensures the retriever finds guidelines relevant to both the *type* of vulnerability (e.g., SQL Injection) and the *context* (e.g., Python SQL syntax).
*   **Context Injection**: The retrieved recipe is injected into the system prompt, guiding the LLM to generate fixes aligned with best practices.

### Model Performance
*   **Unquantized Model**: The current implementation uses the default unquantized `Qwen2.5-Coder-1.5B-Instruct` model loaded via `transformers`. This decision was made to ensure maximum compatibility across different operating systems (Windows, Linux, macOS) and hardware configurations (CPU & GPU) without relying on complex, platform-specific dependencies.
*   **Latency**: On consumer-grade CPUs, inference may be slower due to the lack of quantization. Using a GPU significantly reduces this latency.
*   **Future Optimization**: Performance on CPU-only environments could be drastically improved by switching to a quantized model format (e.g., GGUF via `llama.cpp` or 4-bit loading via `bitsandbytes`). However, `bitsandbytes` support on Windows CPU is limited and experimental, hence the choice for the stable unquantized approach for this assignment.

### Assumptions and Limitations

#### ⚠️ **Assumption on Model Choice and Performance**

To adhere strictly to the **Mandatory Requirements** for **CPU-only inference** and the listed inference libraries, I was required to use the standard **Hugging Face Transformers** library.

1.  **Inference Library Constraint:** The recommended high-performance GPU libraries like **vLLM** lack official, stable native Windows support, and CPU-optimized quantization formats (e.g., GGUF via `llama.cpp`) were not listed as a primary choice.

2.  **Quantization Constraint:** Standard quantization techniques used with Hugging Face (like `bitsandbytes`, AWQ) are primarily designed for and dependent on **GPU hardware (CUDA kernels)** and **do not support efficient pure CPU inference**.

Therefore, I made the necessary assumption to deploy a **non-quantized (e.g., FP32 or BF16) model** using the standard `Hugging Face transformers` pipeline for guaranteed CPU compatibility.

**Performance Impact:** Running a non-quantized model of this size on a CPU-only environment results in significantly **reduced token-per-second throughput** and **higher memory consumption** compared to a GPU-accelerated or a `llama.cpp`/GGUF optimized setup. The reported `latency_ms` and observations in this README reflect this expected performance penalty.

#### Other Limitations
*   The current RAG corpus is limited to a few example recipes.
*   The 1.5B model may occasionally hallucinate on very complex or obscure libraries.

## Author
Nithesh
