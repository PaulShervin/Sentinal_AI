
# Learning Plan — ML Engineer Internship (interview starts 2026-10-18)

## Environment (checked 2026-10-12)
- Windows 11, i5-1335U, 16 GB RAM, MX550 2 GB (torch is CPU build → train on CPU)
- Python 3.13.7 (global) with torch 2.13, transformers 4.56, tiktoken, fastapi, flask, gradio, uvicorn, gunicorn
- Ollama 0.40.2 with gemma3 (3.3 GB) → local inference engine
- Docker 29.4 + Compose v5
- Constraints: vLLM not runnable (needs Linux + bigger GPU) → explain conceptually, use Ollama in practice.
  Gunicorn does not run natively on Windows → run it inside Docker.

## 6-day schedule
| Day | Date | Deep | Minimal | Daily drills |
|---|---|---|---|---|
| 1 | Mon 12 | T1: tokenization, next-token prediction, train/infer, bigram → checkpoint → generate | — | 2 LeetCode |
| 2 | Tue 13 | T1/T2: tiny Transformer (attention), training-param experiments; inference params via Ollama | KV cache, quantization, vLLM (concept) | 2 LeetCode + 1 debug drill |
| 3 | Wed 14 | T3 + T4: plain-Python agent harness, tools, validation, limits, traces, tests; prompt drafting | context summarization | 2 LeetCode + 1 debug drill |
| 4 | Thu 15 | T6 + T7: FastAPI/Uvicorn, Flask/Gunicorn, Dockerfile + Compose (you write first) | multi-stage builds | 2 LeetCode + 1 debug drill |
| 5 | Fri 16 | T8 + T5: Gradio UI on the API, architecture walkthrough; T9 debugging drills block | drift/monitoring (concept) | 2 LeetCode |
| 6 | Sat 17 | Mock interviews across all 10 topics, revise CONCEPTS_TO_REVISE.md | — | 3 LeetCode |

## Projects
- A `project_a_lm/` — tiny LM lab (tokenizer → bigram → Transformer → checkpoint → generate) + Ollama inference experiments
- B `project_b_agent/` — plain-Python agent harness
- C `project_c_app/` — model API (FastAPI + Flask) + Gradio + Docker Compose

## Progress (status = not started / in progress / can explain / verified in mock)
| # | Topic | Status |
|---|---|---|
| 1 | Training a basic LLM + fast inference | in progress |
| 2 | Training & inference parameters | not started |
| 3 | Agent harness without frameworks | not started |
| 4 | Prompt engineering | not started |
| 5 | ML system architecture | not started |
| 6 | Flask+Gunicorn / FastAPI+Uvicorn | not started |
| 7 | Docker & Compose | not started |
| 8 | Gradio | not started |
| 9 | Debugging without AI | not started |
| 10 | LeetCode arrays & dicts | not started |
