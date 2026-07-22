# ⚡ PromptLite — Prompt Token Optimizer

> Reduce LLM token usage by 30–60% while preserving prompt intention — with semantic equivalence verification.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-orange)](https://openai.com)
[![Claude](https://img.shields.io/badge/Anthropic-Claude-8A63D2)](https://anthropic.com)

---

## 🎯 The Problem

Verbose prompts waste tokens — and tokens cost money. A prompt like:

> *"Hello! I hope you're doing well. I would really appreciate it if you could help me with something important. I need you to please summarize..."*

can be reduced to:

> *"Summarize the following, including key findings, diagnosis, and treatment recommendations:"*

**Same intent. 60% fewer tokens. Lower cost. Better latency.**

---

## 🏗️ How It Works

```
Input prompt
     ↓ tiktoken
Count original tokens
     ↓ GPT-4o-mini
Extract core intention (task, context, constraints)
     ↓ Optimization pipeline
Apply 7 reduction techniques
     ↓ tiktoken
Count optimized tokens
     ↓ text-embedding-3-small
Compute semantic similarity (cosine)
     ↓ GPT-4o-mini (optional)
Test output equivalence
     ↓
Grade: A / B / C / D
```

---

## 🛠️ Optimization Techniques

| Technique | Example |
|---|---|
| Remove filler phrases | `"Please help me"` → removed |
| Imperative form | `"Can you summarize"` → `"Summarize"` |
| Remove AI acknowledgments | `"As an AI language model"` → removed |
| Consolidate instructions | 3 similar rules → 1 clear rule |
| Remove obvious context | `"It's important that you"` → removed |
| Compress examples | 3 examples → 1 best example |
| Structured format | prose requirements → bullet list |

---

## 📊 Grading System

| Grade | Meaning |
|---|---|
| **A** | Excellent — high reduction, intention fully preserved |
| **B** | Good — strong reduction, minimal fidelity loss |
| **C** | Acceptable — moderate reduction or minor intent drift |
| **D** | Problematic — LLM output changed significantly, do not use |

Grade formula: `(reduction_pct × 0.4) + (intention_score × 0.3) + (output_similarity × 0.3)`

---

## 🚀 Quickstart

```bash
git clone https://github.com/Gor0d/promptlite
cd promptlite

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY and/or ANTHROPIC_API_KEY to .env

# Terminal 1 — API
uvicorn api.main:app --reload

# Terminal 2 — Dashboard
streamlit run dashboard/app.py
```

### 🐳 Docker

```bash
cp .env.example .env   # add your keys
docker compose up      # API on :8000, dashboard on :8501
```

### 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest            # unit + API tests, no token cost (LLM calls are mocked)
ruff check .
```

---

## 🤖 Providers

PromptLite is provider-agnostic — use **OpenAI** or **Anthropic (Claude)**:

| Provider | Models | Default |
|---|---|---|
| `openai` | `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo` | `gpt-4o-mini` |
| `anthropic` | `claude-haiku-4-5`, `claude-sonnet-5`, `claude-opus-4-8` | `claude-haiku-4-5` |

Pick the provider/model per request (`"provider"`, `"model"`) or in the dashboard.
Semantic similarity uses OpenAI embeddings when `OPENAI_API_KEY` is set, with a local lexical fallback otherwise.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/optimize` | Optimize a single prompt |
| `POST` | `/batch` | Optimize multiple prompts |
| `GET` | `/benchmark` | Run benchmark on 6 real-world prompts |
| `GET` | `/models` | List providers and models |
| `GET` | `/techniques` | List optimization techniques |
| `GET` | `/health` | Health check |

### Example Request

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I would really appreciate it if you could please help me summarize the following text...",
    "test_outputs": false,
    "provider": "anthropic",
    "model": "claude-haiku-4-5"
  }'
```

### Example Response

```json
{
  "original_tokens": 87,
  "optimized_tokens": 12,
  "tokens_saved": 75,
  "reduction_pct": 86.2,
  "intention_score": 0.94,
  "output_similarity": 0.97,
  "grade": "A",
  "optimized_prompt": "Summarize the following text concisely:",
  "cost_saved_usd": 0.000011
}
```

---

## 💡 Use Cases

- **LLM Training Data**: Clean verbose prompts before fine-tuning datasets
- **Production Systems**: Reduce API costs in high-volume applications  
- **RAG Systems**: Optimize system prompts served with every query
- **Prompt Auditing**: Evaluate prompt quality across a codebase

---

## 👨‍💻 Author

**Emerson Guimarães** — AI Engineer & LLM Specialist  
🔗 [linkedin.com/in/emersongsguimaraes](https://linkedin.com/in/emersongsguimaraes)  
🐙 [github.com/Gor0d](https://github.com/Gor0d)
