# LangChain Version Notes

## Environment
- Python: 3.12.5
- Package manager: uv
- Date resolved: 2026-02-28

## Resolved Core Stack

| Package | Version | Notes |
|---|---|---|
| `langchain` | 1.2.10 | Latest stable 1.x |
| `langchain-core` | 1.2.16 | Compatible with 1.2.10 |
| `langchain-community` | 0.4.1 | Latest |
| `langchain-text-splitters` | 1.1.1 | Latest |
| `langchain-ollama` | 1.0.1 | Latest |
| `langchain-groq` | 1.1.2 | Latest |
| `chromadb` | 1.5.2 | Latest |
| `faiss-cpu` | 1.13.2 | Latest |
| `sentence-transformers` | 5.2.3 | Pulls in torch (~2GB) |
| `torch` | 2.10.0 | Auto-pulled by sentence-transformers |
| `pydantic` | 2.12.5 | Latest v2 |
| `tiktoken` | 0.12.0 | Latest |
| `pypdf` | 6.7.4 | Latest |

## Why Version Pinning Failed

The original `requirements.txt` mixed incompatible ecosystem versions:
- `langchain==1.1.3` requires `langchain-core>=1.1.2,<2.0.0`
- `langchain-core==0.3.59` does not satisfy that range
- `langchain-community==0.3.23` requires `langchain>=0.3.24,<1.0.0`
- `langchain-ollama==0.3.3` requires `langchain-core>=0.3.60`

**Fix:** Remove all exact version pins from `requirements.txt` and let uv resolve a compatible set, then freeze to `requirements.lock`.

## Production Status

LangChain 1.x is the current stable series as of early 2026. This is what modern RAG/agent production systems run.

## Notes

- `langchain-classic==1.0.1` is a backward-compat shim for old 0.x API patterns — auto-pulled, not used directly.
- `langgraph==1.0.10` was auto-pulled as a transitive dependency.
- If Ollama is your only embedding provider, `sentence-transformers` can be removed to avoid the heavy torch dependency.
- `requirements.lock` (generated via `uv pip freeze`) is the reproducible lockfile — use this to replicate the environment exactly.

## How to Reinstall

```bash
uv pip install -r requirements.txt        # resolves latest compatible versions
uv pip freeze > requirements.lock         # freeze for reproducibility
uv pip install -r requirements.lock       # exact reproduction
```
