# Scenario Guide — Why Each File Exists

> Read this before looking at code. Each section answers:
> **"When would I actually need this in a real project?"**

---

## 01 — LLM Basics

**Scenario:** You want to call an AI model (Ollama locally, Groq in cloud) and get a response. Nothing fancy — just talk to it.

**Think of it like:** `RestTemplate` or `HttpClient` in Spring Boot. This is just the raw HTTP call to the model. Everything else builds on top of this.

**When you need it:** Always. This is the foundation. But in real projects, you never use this in isolation — you use it inside chains (file 04).

**Key decision you'll make:** Use `OllamaLLM` (plain text in/out, old style) or `ChatOllama` (messages in/out, modern). Always use `ChatOllama`.

---

## 02 — Prompt Templates

**Scenario:** You're building a feature where users can ask questions and you need to inject their input into a pre-defined system instruction. Example: a customer support bot that always starts with "You are a support agent for CompanyX."

**Think of it like:** A Thymeleaf/Jinja2 template but for LLM prompts. You define the structure once, fill in the variables at runtime.

**When you need it:**
- Any time your prompt has dynamic parts (user name, retrieved documents, language preference)
- When you want to reuse the same prompt structure with different inputs
- When you need to inject conversation history into the prompt (MessagesPlaceholder)

**The "MessagesPlaceholder" part specifically:** Critical for chatbots. It's the slot where you inject prior conversation turns so the model remembers context.

---

## 03 — Output Parsers

**Scenario:** You call the LLM and it gives you back a wall of text. But your downstream code needs a Python dict, a list, or a typed object — not raw text.

**Think of it like:** Jackson/Gson in Java (JSON deserializer), but for LLM output. You tell it "I expect this shape", it parses the model's text into that shape.

**When you need it:**
- **StrOutputParser** — always, even just to extract `.content` from the AIMessage object
- **JsonOutputParser** — when you need a dict but don't care about type safety
- **`.with_structured_output(PydanticModel)`** — when downstream code depends on specific typed fields (most real use cases)

**Real examples where this matters:**
- Classify a support ticket → `severity: Enum`, `component: str` → use those fields to route to a team
- Extract entities from text → feed them into a database query
- Generate structured test data from a description

---

## 04 — LCEL Chains

**Scenario:** You have multiple steps: clean the input → call the LLM → parse the output → do something with the result. You need to compose these into a pipeline.

**Think of it like:** Unix pipes (`|`), or a Java Stream pipeline (`.filter().map().collect()`), or a Kafka Streams topology. Each step takes the output of the previous.

**When you need it:** Always when doing more than a single LLM call. This is how you wire everything together.

**Specific components and when:**
- `RunnableParallel` — when you need two LLM calls to run simultaneously (e.g., generate pros AND cons at the same time, cutting response time in half)
- `RunnableBranch` — when you need to route to different chains based on input (e.g., technical question → expert chain, simple question → ELI5 chain)
- `RunnableLambda` — when you need to inject custom Python logic mid-chain (e.g., call your database, format a string)

---

## 05 — Memory & Chat History

**Scenario:** You're building a chatbot and the user says "I work with Kafka" on turn 1, then asks "how should I handle errors in that?" on turn 3. Without memory, the LLM has no idea what "that" refers to.

**Think of it like:** HTTP sessions. LLMs are stateless (like REST APIs), so you store session state externally and inject it on every request — just like reading from a session store.

**When you need it:** Any multi-turn conversation: chatbots, AI assistants, interview bots, code review assistants.

**The production path:**
- Dev/testing → `ChatMessageHistory` (in-memory dict)
- Production → `RedisChatMessageHistory` or `SQLChatMessageHistory` (persistent, survives restarts)

**Critical gotcha:** You must trim history for long conversations. If a user chats for 2 hours, you'll exceed the model's context window. `trim_messages()` in this file shows how.

---

## 06 — Document Loaders

**Scenario:** You want to build a RAG system over your company's internal docs — PDFs, Confluence pages, CSVs, code files. You need to get that content into Python first.

**Think of it like:** A data ingestion adapter. Like Spring Batch's `ItemReader` or a Kafka consumer — it reads from a source and gives you a normalized object. Here the normalized object is `Document(page_content, metadata)`.

**When you need it:** The first step of any RAG pipeline. Before you can search your docs, you have to load them.

**Common real scenarios:**
- Load all PDFs from a folder of company policies → RAG Q&A
- Load Confluence/Notion pages → internal knowledge base search
- Load CSV of product catalog → answer questions about products
- Pull data from your own DB → wrap it as Document objects manually

**The metadata matters:** Every `Document` has metadata (source file, page number, author). This is what lets you show "Source: kafka-guide.pdf, page 3" in your answer.

---

## 07 — Text Splitters

**Scenario:** You loaded a 200-page PDF. You can't pass the entire thing to the LLM — it has a context window limit (typically 4K–128K tokens). You need to break it into smaller pieces.

**Think of it like:** Pagination, but smarter. Instead of splitting at arbitrary character counts, you try to split at natural boundaries (paragraph breaks, section headers) so each chunk is still coherent.

**When you need it:** Step 2 of any RAG pipeline, right after loading documents.

**The overlap concept:** If you split a document at a sentence boundary, the sentence before and after the split carries context. `chunk_overlap=100` means you repeat the last 100 chars in the next chunk so nothing is lost at the seam.

**Which splitter to use:**
- `RecursiveCharacterTextSplitter` → 90% of cases (default)
- `MarkdownHeaderTextSplitter` → Markdown docs (adds section title to metadata)
- Token-based splitting → when you need precise token count control (matching LLM's exact context window)

---

## 08 — Embeddings & Vector Stores

**Scenario:** A user asks "how does Kafka handle failures?" You have 500 documents. You need to find the 3–5 most relevant ones without reading all 500.

**Think of it like:** A specialized search index, like Elasticsearch — but instead of keyword search, it does *semantic* (meaning-based) search. "message bus" and "event streaming platform" will match Kafka even though the words don't overlap.

**How it works conceptually:**
1. Every document gets converted to a list of numbers (a vector) that represents its meaning
2. Your query also gets converted to a vector
3. The database finds the documents whose vectors are closest to your query's vector

**When you need it:** Step 3 of any RAG pipeline. You load → split → embed → store. Then at query time, you embed the question and search.

**FAISS vs Chroma:**
- FAISS → in-memory, fast, no persistence, good for scripts and prototypes
- Chroma → persists to disk, supports metadata filtering, better for ongoing apps

---

## 09 — RAG Pipeline

**Scenario:** You want to ask questions about YOUR data — internal docs, codebase, database records — and get accurate answers instead of hallucinated ones. This is the #1 use case for LangChain in production.

**Think of it like:** Google search + LLM. Search finds the relevant docs, LLM synthesizes the answer from them. Without the search part, the LLM just makes things up.

**When you need it:**
- Internal knowledge base Q&A ("what's our incident response process?")
- Customer support bot over your product documentation
- Code search assistant over your codebase
- Legal document Q&A
- Any time the LLM needs access to private or recent information

**The 4 steps (always the same):**
1. **Ingest** (done once): load → split → embed → store in vector DB
2. **Retrieve**: embed the user's question → similarity search → top-k docs
3. **Augment**: inject retrieved docs into the prompt as context
4. **Generate**: LLM answers using only that context

**The prompt wording is critical:** You must tell the model "answer ONLY from the context" or it will mix in its training data and hallucinate.

---

## 10 — Tools & Agents

**Scenario:** The user asks "what's 15% of 230,000?" or "look up order #12345 in our system." An LLM can't do math reliably and has no access to your database — it needs to *call a function* and use the result.

**Think of it like:** An LLM that can call your APIs. Instead of you deciding what API to call, the model reads the user's intent and decides which tool to invoke. Like a very smart middleware layer.

**When you need it:**
- Any time the answer requires real-time data (prices, inventory, order status)
- Any time the answer requires computation (math, date arithmetic)
- Any time you want the AI to take actions (create a ticket, send a notification)
- When the decision of WHICH action to take depends on the user's input

**The agent loop (what's happening under the hood):**
1. User asks something
2. LLM sees the question + available tools → decides to call tool X with args Y
3. Your code executes tool X
4. Result is fed back to the LLM
5. LLM either calls another tool or gives the final answer

---

## 11 — Custom Tools

**Scenario:** File 10 shows the concept with toy tools (calculator, string reverser). File 11 shows how you build real tools that hit APIs, query databases, validate inputs, and handle errors.

**Think of it like:** File 10 is `Hello World`. File 11 is a production service with validation, error handling, async support.

**When you need it:**
- When your tool needs a Pydantic schema (multiple typed fields, optional params, validation)
- When your tool calls an async external API and shouldn't block
- When you need full control: custom error handling, retries, callbacks
- When you're grouping related tools into a toolkit

**The key rule:** A tool's docstring is what the LLM reads to decide when to use it. Write it like you're explaining it to a smart but literal intern. Be explicit about inputs, outputs, and when to use it.

---

## 12 — Streaming

**Scenario:** You're building a chat UI and the LLM takes 5 seconds to generate a response. Without streaming, the user stares at a blank screen for 5 seconds. With streaming, they see words appearing instantly — like ChatGPT's typing effect.

**Think of it like:** Server-Sent Events (SSE) or WebSocket streaming. Instead of waiting for the full response, you push tokens as they're generated.

**When you need it:**
- Any chatbot or AI assistant UI (users expect instant visual feedback)
- Long-form generation (reports, code, summaries)
- FastAPI endpoints that stream to a frontend

**The async version matters:** `.astream()` is what you use in FastAPI. You return a `StreamingResponse` with an async generator that yields tokens — the frontend receives them in real time.

---

## 13 — Structured Output

**Scenario:** You describe a bug in plain English and want the LLM to extract a structured bug report with fields: `severity`, `component`, `steps_to_reproduce[]`. Or you want to classify a support message and route it based on `intent` and `urgency`.

**Think of it like:** LLM-powered form parsing. The user writes free text, you get back a typed Python object. Like OCR for documents but for any unstructured text.

**When you need it:**
- Extract structured data from unstructured text (resumes, bug reports, feedback)
- Classify inputs to route them in your application logic
- Any time downstream code needs to branch on the LLM's output (`if result.urgency: escalate()`)
- Auto-fill forms or database records from natural language descriptions

**The cleanest approach:** `.with_structured_output(YourPydanticModel)` — the model fills the fields directly. You get type safety, validation, and IDE autocompletion.

---

## 14 — Callbacks & Tracing

**Scenario:** You're running your AI feature in production. You need to know: How long does each LLM call take? How many tokens are we spending per request? Which requests are failing? Did a specific user session hit an error?

**Think of it like:** Spring AOP interceptors or servlet filters — hooks that fire before/after each operation without changing the operation's code. Or like Micrometer metrics in Spring Boot.

**When you need it:**
- Production monitoring: latency, token usage, error rates
- Cost control: token budget limits (stop if we spend too much)
- Debugging: trace exactly what happened in a complex chain
- Audit logging: record every LLM call with its inputs/outputs

**The LangSmith alternative:** If you set `LANGCHAIN_TRACING_V2=true`, LangSmith automatically captures all of this in a web UI with zero extra code. Use callbacks when you need custom behavior (budget limits, custom metrics) or can't use LangSmith.

---

## 15 — Groq Integration

**Scenario:** Your local Ollama model is too slow for a real-time user-facing feature (5–10s per response). You need 500ms responses. Groq's hardware (LPU) runs the same models 5–20x faster.

**Think of it like:** Swapping your local Tomcat dev server for a production CDN-backed deployment. Same code, radically different performance.

**When you need it:**
- Production APIs where response time matters
- Real-time features: live chat, auto-complete, instant classification
- When local GPU is unavailable or insufficient

**The killer feature here:** `.with_fallbacks([local_ollama])` — Groq first, fall back to local Ollama if Groq is down. You get cloud speed with local resilience. This is the pattern for production hybrid deployments.

**Cost:** Groq has a free tier with rate limits. For dev and moderate traffic, it's free.

---

## 16 — Advanced Patterns

**Scenario:** You've built the basic RAG chain. Now you're hitting real limitations:
- RAG results are poor because the query wording doesn't match document wording
- You need to process 100 documents and summarize them all
- Your LLM output quality is inconsistent
- Your production API is getting hammered and failing under load

This file is the "production hardening" chapter.

**Patterns and when each applies:**

| Pattern | Scenario |
|---|---|
| **Retry** | Cloud API (Groq, OpenAI) rate limits or transient errors. One line to add exponential backoff. |
| **Map-Reduce** | Summarize 50 documents. Map = summarize each in parallel. Reduce = combine into final output. |
| **Self-critique** | When output quality matters (legal, medical, customer-facing). Draft → critique → improve. Costs 3x but output is meaningfully better. |
| **Multi-query retrieval** | Your RAG returns irrelevant results. Generate 3 alternative phrasings of the query and merge the results. More diverse retrieval coverage. |
| **HyDE** | Query and document embeddings are in different semantic spaces. Generate a hypothetical answer and embed THAT — it's closer to what a real document looks like. |
| **Async parallel** | You need to make 10 LLM calls. Sequential = 10x slower. `asyncio.gather()` runs them all at once. |

---

## 17 — Capstone: Production AI Assistant

**Scenario:** This is everything combined into one realistic application. A tech assistant that:
- Classifies your question (is this a knowledge lookup? a calculation? general chat?)
- Routes to the right chain (RAG for docs, agent for tools, direct for chat)
- Remembers conversation history
- Streams responses
- Logs performance
- Falls back from Groq to Ollama if cloud is down

**When you need it:** When you're building an actual AI feature, not just experimenting. This is the blueprint you'd follow for a real production assistant.

**Think of it like:** The difference between a "Hello World" REST endpoint and a production service with auth, validation, error handling, logging, and health checks. This file is the production version.

---

## The Learning Path

```
Foundation (run these first, understand before moving on):
  01 → 02 → 03 → 04

Chatbot track (if building conversational features):
  05 → 12 → 13

RAG track (if building knowledge base Q&A):
  06 → 07 → 08 → 09 → 16 (multi-query + HyDE)

Agent track (if building AI that takes actions):
  10 → 11

Production track (when going live):
  14 → 15 → 16 (retry + async) → 17
```
