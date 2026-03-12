"""
15 - Groq Integration
======================
Groq provides blazing-fast LLM inference (100-500 tokens/sec) using
custom LPU (Language Processing Unit) hardware. It's great for:
  - Low-latency production APIs
  - Real-time applications
  - When you need cloud inference without managing models

Models available on Groq (Feb 2026):
  - llama-3.3-70b-versatile  : best quality
  - llama-3.1-8b-instant     : fastest, good quality
  - mixtral-8x7b-32768       : long context (32k)
  - gemma2-9b-it             : Google's model

LangChain interface: langchain-groq
  from langchain_groq import ChatGroq

Setup: Get API key at console.groq.com (free tier available)
       Set GROQ_API_KEY environment variable

WHEN YOU NEED THIS:
  When local Ollama inference is too slow for a user-facing feature.
  Ollama on a typical laptop: 5–15s per response.
  Groq: 0.3–1s for the same model. The difference between a chatbot that feels
  laggy and one that feels instant.

  Think of it like swapping a local SQLite dev database for a managed cloud DB —
  same interface, same queries, dramatically different throughput.

  Real scenarios:
  - Production API with a response time SLA under 2s
  - Real-time chat where a 3s delay is noticeable
  - Auto-complete or inline suggestion features (must feel instant)
  - Demos where perceived speed matters

ChatGroq IS a drop-in replacement:
  Every file (01–14) that uses ChatOllama can switch to ChatGroq by changing
  one line. Same .invoke(), .stream(), .with_structured_output(), .bind_tools()
  — identical interface, different hardware running it.

.with_fallbacks() — THE PRODUCTION PATTERN (demo_fallback_chain):
  primary_llm.with_fallbacks([local_ollama_llm])
  Groq runs first. If Groq is down or rate-limited, the chain automatically
  falls back to local Ollama — no code change, no failed requests. Cloud speed
  in normal operation, local resilience when the cloud is unavailable.

COST:
  Groq offers a generous free tier. For development and moderate production
  traffic it costs nothing. Cost only becomes relevant at very high scale.

Run:  python 15_groq_integration.py
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()  # loads from .env file if present

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"
FAST_MODEL   = "llama-3.1-8b-instant"

def check_groq_key():
    if not GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY not set.")
        print("  Set it via: export GROQ_API_KEY=gsk_...")
        print("  Or create a .env file with: GROQ_API_KEY=gsk_...")
        print("  Get a free key at: https://console.groq.com\n")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic Groq setup — same interface as Ollama
# ─────────────────────────────────────────────────────────────────────────────
def demo_groq_basics():
    print("\n" + "=" * 60)
    print("1. Groq basics — drop-in replacement for any chat model")
    print("=" * 60)

    if not check_groq_key():
        return

    from langchain_groq import ChatGroq

    # ChatGroq has the exact same interface as ChatOllama, ChatOpenAI, etc.
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=256,
        api_key=GROQ_API_KEY,
    )

    response = llm.invoke([
        SystemMessage(content="You are a concise technical assistant."),
        HumanMessage(content="What makes Groq's LPU different from GPU for LLM inference?"),
    ])

    print(f"Model    : {GROQ_MODEL}")
    print(f"Response : {response.content}")
    if response.response_metadata:
        usage = response.response_metadata.get("token_usage", {})
        print(f"Tokens   : {usage.get('prompt_tokens', '?')} in, {usage.get('completion_tokens', '?')} out")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Groq in an LCEL chain — swap providers transparently
# ─────────────────────────────────────────────────────────────────────────────
def demo_groq_chain():
    print("\n" + "=" * 60)
    print("2. Groq in an LCEL chain — transparent provider swap")
    print("=" * 60)

    if not check_groq_key():
        return

    from langchain_groq import ChatGroq

    # This is the same chain you'd write with Ollama — just swap the model
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are an expert software architect. Be precise and practical."),
            ("human", "Design a brief architecture for: {system}"),
        ])
        | ChatGroq(model=GROQ_MODEL, temperature=0.3, api_key=GROQ_API_KEY)
        | StrOutputParser()
    )

    result = chain.invoke({"system": "a URL shortener service handling 10M requests/day"})
    print(result)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fallback chain — Groq → Ollama (if Groq fails)
# ─────────────────────────────────────────────────────────────────────────────
def demo_fallback_chain():
    print("\n" + "=" * 60)
    print("3. Fallback chain — cloud first, local as backup")
    print("=" * 60)

    from langchain_ollama import ChatOllama

    # .with_fallbacks() — if primary fails, automatically tries fallbacks
    # This is how you build resilient production chains
    if GROQ_API_KEY:
        from langchain_groq import ChatGroq
        primary_llm = ChatGroq(model=FAST_MODEL, temperature=0, api_key=GROQ_API_KEY)
    else:
        primary_llm = ChatOllama(model="llama3.2", temperature=0)

    fallback_llm = ChatOllama(model="llama3.2", temperature=0)

    # Chain: try primary, fall back to local if any exception occurs
    resilient_llm = primary_llm.with_fallbacks([fallback_llm])

    chain = (
        ChatPromptTemplate.from_messages([("human", "What is {topic} in one sentence?")])
        | resilient_llm
        | StrOutputParser()
    )

    result = chain.invoke({"topic": "eventual consistency"})
    print(f"Response (with fallback safety): {result}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Groq for structured output
# ─────────────────────────────────────────────────────────────────────────────
def demo_groq_structured():
    print("\n" + "=" * 60)
    print("4. Groq with structured output — fast typed extraction")
    print("=" * 60)

    if not check_groq_key():
        return

    from langchain_groq import ChatGroq
    from pydantic import BaseModel, Field
    from typing import List

    class TechStack(BaseModel):
        """Technology stack recommendation for a given use case."""
        backend: str = Field(description="Recommended backend technology")
        database: str = Field(description="Recommended database")
        cache: str = Field(description="Recommended caching solution")
        message_queue: str = Field(description="Recommended message queue if needed")
        containerization: str = Field(description="Container/orchestration recommendation")
        reasoning: str = Field(description="Brief reasoning for these choices")
        trade_offs: List[str] = Field(description="Key trade-offs to be aware of")

    llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)
    structured_llm = llm.with_structured_output(TechStack)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior architect. Recommend the optimal tech stack."),
        ("human", "Use case: {use_case}"),
    ])

    chain = prompt | structured_llm
    result = chain.invoke({
        "use_case": "Real-time fraud detection for a fintech app handling 1M transactions/day"
    })

    print(f"Backend    : {result.backend}")
    print(f"Database   : {result.database}")
    print(f"Cache      : {result.cache}")
    print(f"Queue      : {result.message_queue}")
    print(f"Containers : {result.containerization}")
    print(f"\nReasoning  :\n  {result.reasoning}")
    print(f"\nTrade-offs :")
    for t in result.trade_offs:
        print(f"  - {t}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Groq streaming — ultra-low latency token streaming
# ─────────────────────────────────────────────────────────────────────────────
def demo_groq_streaming():
    print("\n" + "=" * 60)
    print("5. Groq streaming — near real-time token delivery")
    print("=" * 60)

    if not check_groq_key():
        return

    import time
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=FAST_MODEL, temperature=0.5, api_key=GROQ_API_KEY)

    prompt = ChatPromptTemplate.from_messages([
        ("human", "Write a short story about a developer who discovers {discovery}.")
    ])

    chain = prompt | llm | StrOutputParser()

    print("Streaming from Groq (notice the speed):")
    print("-" * 40)

    start = time.time()
    char_count = 0

    for chunk in chain.stream({"discovery": "their code is sentient"}):
        print(chunk, end="", flush=True)
        char_count += len(chunk)

    elapsed = time.time() - start
    print(f"\n\n[{char_count} chars in {elapsed:.2f}s = {char_count/elapsed:.0f} chars/sec]")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Comparing Groq vs Ollama — side-by-side timing
# ─────────────────────────────────────────────────────────────────────────────
def demo_speed_comparison():
    print("\n" + "=" * 60)
    print("6. Speed comparison — Groq vs Ollama")
    print("=" * 60)

    import time
    from langchain_ollama import ChatOllama

    question = "Explain CAP theorem in 3 sentences."

    # Ollama (local)
    ollama_llm = ChatOllama(model="llama3.2", temperature=0)
    start = time.time()
    ollama_result = ollama_llm.invoke([HumanMessage(content=question)])
    ollama_time = time.time() - start

    print(f"Ollama (llama3.2)   : {ollama_time:.2f}s")
    print(f"  Response: {ollama_result.content[:80]}...")

    if not GROQ_API_KEY:
        print("\nGroq: Skipped (GROQ_API_KEY not set)")
        return

    from langchain_groq import ChatGroq

    groq_llm = ChatGroq(model=FAST_MODEL, temperature=0, api_key=GROQ_API_KEY)
    start = time.time()
    groq_result = groq_llm.invoke([HumanMessage(content=question)])
    groq_time = time.time() - start

    print(f"Groq ({FAST_MODEL}) : {groq_time:.2f}s")
    print(f"  Response: {groq_result.content[:80]}...")
    print(f"\nGroq speedup: {ollama_time / groq_time:.1f}x faster")


if __name__ == "__main__":
    demo_groq_basics()
    demo_groq_chain()
    demo_fallback_chain()
    demo_groq_structured()
    demo_groq_streaming()
    demo_speed_comparison()

    print("\n\nKey takeaways:")
    print("  - ChatGroq is a drop-in replacement for ChatOllama — same interface")
    print("  - Groq is 5-20x faster than local Ollama for most model sizes")
    print("  - Use .with_fallbacks([local_llm]) for resilient cloud+local hybrid")
    print("  - Free tier at console.groq.com — great for development and demos")
    print("  - For production: Groq for low-latency APIs, Ollama for private/offline")
    print("  - GROQ_API_KEY in .env file — never hardcode it in source code")
