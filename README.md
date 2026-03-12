# LangChain Reference — Basic to Advanced

Python 3.12.5 | LangChain 1.1.3 | Ollama (local) + Groq (cloud)

## Setup

```bash
cd langchain-references

# Create virtual environment
py -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models (must have Ollama running)
ollama pull llama3.2
ollama pull nomic-embed-text   # for embeddings

# (Optional) Set Groq API key for file 15+
# Get free key at console.groq.com
set GROQ_API_KEY=gsk_your_key_here
```

## Files — Learning Path

| File | Concept | Key APIs |
|------|---------|----------|
| `01_llm_basics.py` | LLM vs ChatModel, invoke/stream/batch | `OllamaLLM`, `ChatOllama`, `HumanMessage` |
| `02_prompt_templates.py` | PromptTemplate, ChatPromptTemplate, few-shot | `ChatPromptTemplate`, `MessagesPlaceholder`, `FewShotChatMessagePromptTemplate` |
| `03_output_parsers.py` | Parse model output into Python types | `StrOutputParser`, `JsonOutputParser`, `.with_structured_output()` |
| `04_lcel_chains.py` | LCEL pipe operator, Runnables | `RunnablePassthrough`, `RunnableParallel`, `RunnableLambda`, `RunnableBranch` |
| `05_memory_chat_history.py` | Conversation memory, multi-session | `ChatMessageHistory`, `RunnableWithMessageHistory`, `trim_messages` |
| `06_document_loaders.py` | Load docs from files, CSV, JSON | `TextLoader`, `CSVLoader`, `JSONLoader`, `DirectoryLoader` |
| `07_text_splitters.py` | Chunk large documents for RAG | `RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter` |
| `08_embeddings_vectorstores.py` | Semantic search, vector DBs | `OllamaEmbeddings`, `FAISS`, `Chroma`, `as_retriever()` |
| `09_rag_pipeline.py` | Full RAG system end-to-end | Complete ingest → retrieve → augment → generate pipeline |
| `10_tools_and_agents.py` | Tool calling, ReAct agent loop | `@tool`, `bind_tools`, `create_react_agent`, `MemorySaver` |
| `11_custom_tools.py` | Production-quality custom tools | `StructuredTool`, `BaseTool`, async tools, error handling |
| `12_streaming.py` | Token streaming for real-time UX | `.stream()`, `.astream()`, `.astream_events()` |
| `13_structured_output.py` | Typed extraction from LLMs | `.with_structured_output()`, Pydantic models, classification |
| `14_callbacks_tracing.py` | Observability and monitoring | `BaseCallbackHandler`, token counting, JSON logging |
| `15_groq_integration.py` | Cloud inference, fallback chains | `ChatGroq`, `.with_fallbacks()`, speed comparison |
| `16_advanced_patterns.py` | Production RAG + agent patterns | Map-reduce, self-critique, multi-query, HyDE, async parallel |
| `17_capstone_ai_assistant.py` | Everything combined | Intent routing, RAG + agents + memory + streaming |

## Key Mental Models

### LCEL Pipe Operator
```
input → [Prompt] → [LLM] → [Parser] → output
       ────────────────────────────────────────
       prompt | llm | StrOutputParser()
```

### RAG Pipeline
```
Documents → Splitter → Embeddings → VectorStore
                                         ↓
User Query → Embed Query → Similarity Search → Top-K Docs
                                         ↓
                          [Docs + Query] → LLM → Answer
```

### Agent Loop
```
User Query
    ↓
LLM (with tools) → Tool call? → No → Final Answer
                        ↓
                   Execute Tool
                        ↓
                   Feed result back → LLM → ...
```

### Memory Pattern
```
session_store[session_id] = ChatMessageHistory
RunnableWithMessageHistory auto-loads/saves per request
MessagesPlaceholder injects history into the prompt
```

## Run Order (Recommended)

1. Start with `01` → `04` (core interfaces + LCEL)
2. Then `05` (memory) — critical for chatbots
3. Then `06` → `09` (document pipeline → RAG)
4. Then `10` → `11` (agents + tools)
5. Then `12` → `16` (production features)
6. Finish with `17` (capstone — everything together)

## Ollama Models

```bash
ollama list                    # see installed models
ollama pull llama3.2           # 2B, fast, great for demos
ollama pull llama3.1:8b        # 8B, better quality
ollama pull nomic-embed-text   # for embeddings (RAG)
ollama pull mxbai-embed-large  # higher quality embeddings
```
