# LLM-Based Circuit Analyzer - Requirements and Setup

## Requirements.txt

# Core ML/AI Libraries
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
sentence-transformers>=2.2.0
ultralytics>=8.0.0
opencv-python>=4.7.0
Pillow>=9.5.0

# Circuit Simulation
PySpice>=1.5.0
spice-parser>=0.1.0
netlistx>=0.1.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0

# Vector Databases and Search
faiss-cpu>=1.7.4  # or faiss-gpu for GPU support
chromadb>=0.4.0
elasticsearch>=8.8.0
rank-bm25>=0.2.2

# Document Processing
PyPDF2>=3.0.0
python-docx>=0.8.11
beautifulsoup4>=4.12.0
markdown>=3.4.0
spacy>=3.6.0

# API and Async
aiohttp>=3.8.0
asyncio-mqtt>=0.13.0
openai>=0.27.0  # Optional: for OpenAI API
anthropic>=0.3.0  # Optional: for Anthropic API

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
click>=8.1.0
tqdm>=4.65.0

# Development and Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.0.0
flake8>=6.0.0

## Setup Instructions

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv circuit_analyzer_env

# Activate environment
# On Windows:
circuit_analyzer_env\Scripts\activate
# On macOS/Linux:
source circuit_analyzer_env/bin/activate

# Install requirements
pip install -r requirements.txt

# Install spaCy language model
python -m spacy download en_core_web_sm
```

### 2. Hardware Requirements

**Minimum Requirements:**
- RAM: 8GB
- Storage: 10GB free space
- GPU: Optional but recommended (NVIDIA GPU with CUDA support)

**Recommended Requirements:**
- RAM: 16GB+
- Storage: 20GB+ free space  
- GPU: NVIDIA GPU with 8GB+ VRAM
- CPU: Multi-core processor (8+ cores recommended)

### 3. External Dependencies

#### NgSpice Installation (Required for Circuit Simulation)

**Windows:**
```bash
# Download and install NgSpice from: http://ngspice.sourceforge.net/download.html
# Add NgSpice bin directory to PATH environment variable
```

**macOS:**
```bash
brew install ngspice
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ngspice ngspice-doc
```

#### Elasticsearch (Optional - for advanced search)

**Using Docker:**
```bash
docker run -d --name elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  elasticsearch:8.8.0
```

**Manual Installation:**
Follow instructions at: https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html

### 4. Model Downloads

The system will automatically download required models on first run:

- YOLO models (for component detection)
- Transformer models (for text processing)
- Sentence transformer models (for embeddings)

**Manual Model Download (Optional):**
```python
from ultralytics import YOLO
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Download models manually
yolo_model = YOLO('yolov8n.pt')
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
classifier = pipeline('text-classification', model='distilbert-base-uncased')
```

### 5. Configuration Setup

#### Create Configuration File

Create `config.json`:
```json
{
  "net_head": {
    "yolo_model_path": "yolov8n.pt",
    "component_threshold": 0.5,
    "max_components": 50
  },
  "refinement_head": {
    "mllm_model": "microsoft/DialoGPT-large",
    "max_tokens": 200
  },
  "decision_head": {
    "llm": {
      "type": "huggingface",
      "model_name": "microsoft/DialoGPT-large",
      "api_key": null
    },
    "max_steps": 10
  },
  "rag": {
    "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
    "domain_model": "sentence-transformers/allenai-specter",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 5,
    "knowledge_base_path": "./circuit_knowledge_base",
    "processed_docs_path": "./processed_circuit_docs",
    "elasticsearch": {
      "enabled": false,
      "host": "localhost:9200",
      "index": "circuit_knowledge"
    }
  }
}
```

#### Environment Variables

Create `.env` file:
```
# API Keys (Optional)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Paths
KNOWLEDGE_BASE_PATH=./circuit_knowledge_base
PROCESSED_DOCS_PATH=./processed_circuit_docs
MODEL_CACHE_PATH=./model_cache

# Elasticsearch (Optional)
ELASTICSEARCH_HOST=localhost:9200
ELASTICSEARCH_INDEX=circuit
