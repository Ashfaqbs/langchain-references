"""
17 - Capstone: Production AI Assistant
========================================
Everything combined — a production-ready AI assistant that demonstrates
all the major LangChain concepts in a realistic, cohesive application.

Features:
  - Dual LLM support (Ollama local + Groq cloud with fallback)
  - RAG over a technical knowledge base
  - Conversation memory (multi-turn)
  - Tool use (calculator, search, data lookup)
  - Structured output for specific query types
  - Streaming responses
  - Callbacks for logging and performance tracking
  - Async support for concurrent users

Architecture:
  User Query
       ↓
  [Intent Classifier] → structured output
       ↓
  [Router] → RAG chain / Tool agent / Direct answer
       ↓
  [Chain with Memory + Streaming]
       ↓
  Response

WHEN YOU NEED THIS:
  This is the blueprint for a real production AI assistant — not a demo or
  experiment. Read this file when ready to build an actual feature and see
  how everything from files 01–16 connects in a single cohesive application.

  The difference between a toy "Hello World" chain and this is the same as
  the difference between a one-off script and a production service with routing,
  error handling, observability, persistence, and graceful fallback.

WHAT EACH COMPONENT IS DOING AND WHY:

  classify_intent() — structured output (file 13):
    Reads the message and determines the route: 'rag', 'tools', or 'direct'.
    Structured output ensures route is always one of the valid enum values,
    not free text that requires parsing or could be misspelled.

  build_rag_chain() — RAG (file 09) + memory (file 05):
    For questions about specific technical topics. Retrieves relevant chunks
    from the vector store, injects them as context, and includes recent history
    so follow-up questions reference earlier turns correctly.

  build_agent_chain() — tools (file 10):
    For questions requiring computation or data lookup (math, comparisons).
    The LLM selects which tool to call, runs it, and incorporates the result.

  build_direct_chain() — basic chain (file 04) + memory (file 05):
    For general conversation that doesn't need docs or tools. The lightweight
    path — no vector search, no tool calls.

  build_llm() — Groq + fallback (file 15):
    Groq runs first for speed. Falls back to local Ollama automatically if
    Groq is unavailable. The rest of the application is unaware of which
    model is actually serving the request.

  AssistantLogger — callbacks (file 14):
    Tracks latency per LLM call. In production, these metrics would feed into
    Prometheus, Datadog, or a similar monitoring system.

  stream_chat() — streaming (file 12):
    Returns a generator. The caller (a FastAPI StreamingResponse or a CLI loop)
    receives tokens as they're generated and can display them immediately.

HOW TO ADAPT THIS FOR A DIFFERENT PROJECT:
  1. Replace KNOWLEDGE_DOCS with the target document corpus
  2. Replace/add tools that connect to the relevant APIs and databases
  3. Swap ChatMessageHistory for RedisChatMessageHistory in production
  4. Wrap TechAssistant.stream_chat() in a FastAPI StreamingResponse endpoint
  5. Tie session_id to the application's authentication/user system

Run:  python 17_capstone_ai_assistant.py
"""

import os
import asyncio
import time
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
LLM_MODEL       = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")

def build_llm():
    """Build LLM with Groq fallback to Ollama."""
    ollama_llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

    if GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            groq_llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.2,
                api_key=GROQ_API_KEY,
            )
            return groq_llm.with_fallbacks([ollama_llm])
        except Exception:
            pass

    return ollama_llm


# ─────────────────────────────────────────────────────────────────────────────
# Performance logger
# ─────────────────────────────────────────────────────────────────────────────
class AssistantLogger(BaseCallbackHandler):
    def __init__(self):
        self.calls: list = []
        self._start: float = 0

    def on_llm_start(self, *args, **kwargs):
        self._start = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs):
        elapsed_ms = round((time.time() - self._start) * 1000)
        self.calls.append(elapsed_ms)

    def stats(self) -> str:
        if not self.calls:
            return "No LLM calls recorded"
        avg = sum(self.calls) // len(self.calls)
        return f"LLM calls: {len(self.calls)}, avg latency: {avg}ms, total: {sum(self.calls)}ms"


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge base for RAG
# ─────────────────────────────────────────────────────────────────────────────
KNOWLEDGE_DOCS = [
    Document(page_content="""
Kafka Architecture: Kafka uses topics, partitions, and consumer groups.
Producers write to topics. Consumers in a group each read from distinct partitions.
Kafka brokers store logs. ZooKeeper (or KRaft) manages cluster metadata.
Key config: retention.ms, num.partitions, replication.factor.
Best practices: set replication.factor=3 in production, use partition keys for ordering,
monitor consumer lag, use separate consumer groups for different use cases.
""", metadata={"source": "kafka", "topic": "kafka"}),

    Document(page_content="""
Redis Best Practices: Use the right data structure for each use case.
Strings for simple values and counters. Hashes for objects (avoid JSON strings).
Sets for uniqueness and intersection. Sorted Sets for leaderboards and timelines.
Lists for queues (LPUSH/BRPOP pattern). Streams for event logs.
Always set TTL. Use SCAN not KEYS. Pipeline bulk operations.
Key format: service:entity:id (e.g., app:user:123:session).
Redis Cluster for horizontal scaling. Redis Sentinel for HA.
""", metadata={"source": "redis", "topic": "redis"}),

    Document(page_content="""
PostgreSQL Performance: Use EXPLAIN ANALYZE for query plans.
Create indexes on WHERE, JOIN, ORDER BY columns. Use partial indexes for filtered queries.
Use connection pooling (PgBouncer) — never open connections per-request.
Use TIMESTAMPTZ not TIMESTAMP. UUID for public IDs, BIGSERIAL for internal PKs.
Run VACUUM and ANALYZE regularly. Monitor with pg_stat_statements.
Use read replicas for read-heavy workloads. Partition large tables by date or range.
""", metadata={"source": "postgresql", "topic": "postgresql"}),

    Document(page_content="""
Spring Boot 3 Patterns: Use constructor injection, not @Autowired on fields.
@ConfigurationProperties with records for type-safe config.
@Valid on controller inputs for automatic validation.
@Transactional on service methods, not repositories.
Use @RestControllerAdvice for global exception handling.
Spring Data JPA for relational data. Keep N+1 queries in check with JOIN FETCH.
Use Spring Cache with Redis for method-level caching (@Cacheable, @CacheEvict).
Actuator for health checks and metrics. Spring Security for auth.
""", metadata={"source": "spring-boot", "topic": "spring-boot"}),

    Document(page_content="""
FastAPI Production Patterns: Use Pydantic models for all I/O.
Async endpoints for I/O-bound operations. Lifespan context manager for startup/shutdown.
Dependency injection via Depends() for DB sessions, auth, services.
Use BackgroundTasks for fire-and-forget. Rate limiting with slowapi.
CORS with explicit origins — never allow * in production.
Add request_id to every request for tracing. Use structlog for JSON logs.
Deploy behind nginx or Caddy. Use gunicorn + uvicorn workers in production.
""", metadata={"source": "fastapi", "topic": "fastapi"}),
]


def build_knowledge_base() -> FAISS:
    """Build and return the vector store from knowledge docs."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60)
    chunks = splitter.split_documents(KNOWLEDGE_DOCS)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Examples: '2**10', 'round(1000000/3, 2)'"""
    import math
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math, "round": round})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_tech_comparison(tech_a: str, tech_b: str) -> str:
    """Compare two technologies briefly. Returns key differences."""
    comparisons = {
        frozenset(["kafka", "redis"]): "Kafka: durable event log, high-throughput streaming. Redis: in-memory, low-latency, pub/sub + cache. Use Kafka for event sourcing, Redis for ephemeral real-time data.",
        frozenset(["postgresql", "mongodb"]): "PostgreSQL: relational, ACID, structured schemas. MongoDB: document-oriented, flexible schema, horizontal scaling. Choose based on data structure and query patterns.",
        frozenset(["docker", "kubernetes"]): "Docker: container runtime and image format. Kubernetes: container orchestration platform. Docker packages apps, Kubernetes runs and manages them at scale.",
        frozenset(["fastapi", "spring boot"]): "FastAPI: Python, async-first, great for ML/AI APIs. Spring Boot: Java, enterprise-grade, huge ecosystem. FastAPI is faster to develop; Spring Boot has more features.",
    }
    key = frozenset([tech_a.lower(), tech_b.lower()])
    return comparisons.get(key, f"No direct comparison available for {tech_a} vs {tech_b}. Try asking more specifically.")


# ─────────────────────────────────────────────────────────────────────────────
# Intent classification
# ─────────────────────────────────────────────────────────────────────────────
class Intent(BaseModel):
    """Classified intent for routing the query."""
    route: str = Field(description="One of: 'rag' (technical docs), 'tools' (calculation/comparison), 'direct' (general chat)")
    topic: str = Field(description="Main topic or technology mentioned")
    confidence: float = Field(description="Confidence 0.0-1.0")


def classify_intent(query: str, llm) -> Intent:
    """Classify the user's intent to route to the right chain."""
    classifier = llm.with_structured_output(Intent)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify this query for routing. 'rag' if about specific tech details, 'tools' if needs calculation or comparison, 'direct' for general chat."),
        ("human", "{query}"),
    ])
    return (prompt | classifier).invoke({"query": query})


# ─────────────────────────────────────────────────────────────────────────────
# Chain builders
# ─────────────────────────────────────────────────────────────────────────────
def build_rag_chain(vectorstore: FAISS, llm, history: ChatMessageHistory):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(f"[{doc.metadata.get('source', '?')}]:\n{doc.page_content}" for doc in docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior engineer. Answer using the provided context. If not in context, say so concisely.\n\nContext:\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    def invoke(question: str) -> str:
        context = format_docs(retriever.invoke(question))
        return chain.invoke({
            "context": context,
            "history": history.messages[-6:],  # keep last 3 turns
            "question": question,
        })

    return invoke


def build_agent_chain(llm, history: ChatMessageHistory):
    tools = [calculate, get_tech_comparison]
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    from langchain_core.messages import HumanMessage, ToolMessage

    def invoke(question: str) -> str:
        messages = list(history.messages[-4:]) + [HumanMessage(content=question)]

        for _ in range(5):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return response.content

            for call in response.tool_calls:
                tool_fn = tool_map.get(call["name"])
                result = tool_fn.invoke(call["args"]) if tool_fn else "Tool not found"
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

        return messages[-1].content if hasattr(messages[-1], "content") else "Unable to process."

    return invoke


def build_direct_chain(llm, history: ChatMessageHistory):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful, concise technical assistant."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    def invoke(question: str) -> str:
        return chain.invoke({
            "history": history.messages[-6:],
            "question": question,
        })

    return invoke


# ─────────────────────────────────────────────────────────────────────────────
# Main assistant class
# ─────────────────────────────────────────────────────────────────────────────
class TechAssistant:
    def __init__(self):
        print("Initializing TechAssistant...")
        self.logger   = AssistantLogger()
        self.llm      = build_llm()
        self.history  = ChatMessageHistory()
        self.vectorstore = build_knowledge_base()
        self.rag_chain    = build_rag_chain(self.vectorstore, self.llm, self.history)
        self.agent_chain  = build_agent_chain(self.llm, self.history)
        self.direct_chain = build_direct_chain(self.llm, self.history)
        print("Ready.\n")

    def chat(self, user_input: str) -> str:
        """Process a user query and return a response."""
        # Classify intent
        try:
            intent = classify_intent(user_input, self.llm)
            route  = intent.route
            topic  = intent.topic
        except Exception:
            route, topic = "direct", "general"

        print(f"[Route: {route}, Topic: {topic}]")

        # Route to appropriate chain
        if route == "rag":
            response = self.rag_chain(user_input)
        elif route == "tools":
            response = self.agent_chain(user_input)
        else:
            response = self.direct_chain(user_input)

        # Store in history
        from langchain_core.messages import HumanMessage, AIMessage
        self.history.add_message(HumanMessage(content=user_input))
        self.history.add_message(AIMessage(content=response))

        return response

    def stream_chat(self, user_input: str):
        """Stream a response for the given input (bypasses routing for simplicity)."""
        from langchain_core.messages import HumanMessage, AIMessage

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a concise technical assistant."),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ])
        chain = prompt | self.llm | StrOutputParser()

        full = ""
        for chunk in chain.stream({
            "history": self.history.messages[-6:],
            "question": user_input,
        }):
            yield chunk
            full += chunk

        self.history.add_message(HumanMessage(content=user_input))
        self.history.add_message(AIMessage(content=full))

    def stats(self) -> str:
        return (
            f"Session: {len(self.history.messages) // 2} turns | "
            f"{self.logger.stats()}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Demo conversation
# ─────────────────────────────────────────────────────────────────────────────
def run_demo():
    assistant = TechAssistant()

    conversation = [
        ("direct", "Hi! I'm building a high-throughput data pipeline. What should I use?"),
        ("rag",    "What are the best practices for Kafka consumer groups?"),
        ("rag",    "How should I configure Redis for session storage?"),
        ("tools",  "Compare Kafka and Redis for me."),
        ("tools",  "If I have 1.2 million events per day, how many events per second is that?"),
        ("rag",    "Any tips for PostgreSQL connection pooling?"),
    ]

    for expected_route, query in conversation:
        print("\n" + "=" * 70)
        print(f"User: {query}")
        print("-" * 70)

        response = assistant.chat(query)
        print(f"Assistant: {response}")

    print("\n" + "=" * 70)
    print("Session stats:")
    print(f"  {assistant.stats()}")

    # Demo streaming
    print("\n" + "=" * 70)
    print("Streaming demo:")
    print("User: Give me a one-liner on each: Kafka, Redis, PostgreSQL")
    print("Assistant: ", end="", flush=True)
    for chunk in assistant.stream_chat("Give me a one-liner on each: Kafka, Redis, PostgreSQL"):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    run_demo()
