"""
12 - Streaming
===============
Streaming lets you display tokens as they're generated instead of waiting
for the full response. This dramatically improves perceived performance
and is essential for chatbot UIs.

LangChain streaming methods:
  - .stream()    : sync generator — yields chunks one at a time
  - .astream()   : async generator — non-blocking for web apps
  - .astream_events() : fine-grained event stream (tool calls, chain steps, etc.)

Every Runnable in a chain propagates streaming automatically.

WHEN YOU NEED THIS:
  Any user-facing AI feature benefits from streaming. Without it, a user waits
  5–10 seconds staring at a blank screen. With it, words appear immediately —
  same total response time, but perceived as much faster because feedback starts
  at the first token. Same principle as chunked HTTP transfer encoding.

  Real scenarios:
  - Chat UI: stream tokens directly to the browser (ChatGPT-style typing effect)
  - Long-form generation: summarizing a 50-page report → visible as it generates
  - CLI tools: show output progressively while the model thinks
  - Code generation: display code as it's written, not all at once

.stream() vs .astream() — which to use:
  .stream()  → synchronous generator; use in scripts, CLI tools, notebooks
  .astream() → async generator; use in FastAPI/async web servers. Return a
               StreamingResponse wrapping the async generator.

FastAPI streaming pattern (from demo_async_stream):
  @app.get("/chat")
  async def chat(query: str):
      return StreamingResponse(stream_generator(query), media_type="text/plain")

  async def stream_generator(query: str):
      async for chunk in chain.astream({"question": query}):
          yield chunk

astream_events() — when tokens alone aren't enough:
  In a complex chain (RAG + tools + parsing), events mark WHEN each step runs:
  retrieval started, LLM started, tool called, parser finished. Use this to
  build UIs that show "Searching docs... Generating answer..." progress
  indicators rather than just a blank loading state.

Run:  python 12_streaming.py
"""

import asyncio
import sys
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

MODEL = "llama3.2"
llm = ChatOllama(model=MODEL, temperature=0.7)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic sync streaming — .stream()
# ─────────────────────────────────────────────────────────────────────────────
def demo_sync_stream():
    print("\n" + "=" * 60)
    print("1. Sync streaming with .stream()")
    print("=" * 60)

    from langchain_core.messages import HumanMessage

    print("Streaming response (tokens appear as generated):")
    print("-" * 40)

    # .stream() yields AIMessageChunk objects
    full_response = ""
    for chunk in llm.stream([HumanMessage(content="Write a haiku about distributed systems.")]):
        print(chunk.content, end="", flush=True)
        full_response += chunk.content

    print()  # newline after stream ends
    print(f"\n[Total chars streamed: {len(full_response)}]")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Streaming through a full chain
# ─────────────────────────────────────────────────────────────────────────────
def demo_chain_streaming():
    print("\n" + "=" * 60)
    print("2. Streaming through a full LCEL chain")
    print("=" * 60)

    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a technical writer. Be detailed but clear."),
            ("human", "Explain {topic} in about 100 words."),
        ])
        | llm
        | StrOutputParser()  # StrOutputParser also streams — outputs plain strings
    )

    print("Streaming chain output:")
    print("-" * 40)

    # .stream() on the chain yields string chunks (because StrOutputParser extracts content)
    for chunk in chain.stream({"topic": "event-driven architecture"}):
        print(chunk, end="", flush=True)

    print()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Async streaming — for FastAPI / async web servers
# ─────────────────────────────────────────────────────────────────────────────
async def demo_async_stream():
    print("\n" + "=" * 60)
    print("3. Async streaming with .astream()")
    print("=" * 60)

    chain = (
        ChatPromptTemplate.from_messages([
            ("human", "List 5 benefits of {technology}, one per line.")
        ])
        | llm
        | StrOutputParser()
    )

    print("Async streaming (non-blocking, ideal for FastAPI):")
    print("-" * 40)

    # In a FastAPI endpoint, you'd use StreamingResponse + this generator
    async def stream_generator(topic: str):
        async for chunk in chain.astream({"technology": topic}):
            yield chunk

    # How you'd write a FastAPI streaming endpoint:
    # @app.get("/stream")
    # async def stream_endpoint(topic: str):
    #     from fastapi.responses import StreamingResponse
    #     return StreamingResponse(stream_generator(topic), media_type="text/plain")

    full = ""
    async for chunk in stream_generator("Kafka"):
        print(chunk, end="", flush=True)
        full += chunk

    print()
    print(f"\n[Streamed {len(full)} chars asynchronously]")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Streaming with accumulation — build final response while streaming
# ─────────────────────────────────────────────────────────────────────────────
def demo_stream_accumulate():
    print("\n" + "=" * 60)
    print("4. Stream + accumulate full response")
    print("=" * 60)

    chain = (
        ChatPromptTemplate.from_messages([
            ("human", "What are 3 key design patterns in {domain}?")
        ])
        | llm
        | StrOutputParser()
    )

    print("Streaming while building the full response:")
    print("-" * 40)

    chunks = []
    for chunk in chain.stream({"domain": "microservices"}):
        print(chunk, end="", flush=True)
        chunks.append(chunk)

    print()

    # Reassemble — useful when you need the full text after streaming
    full_response = "".join(chunks)
    word_count = len(full_response.split())
    print(f"\n[Full response: {len(full_response)} chars, {word_count} words]")


# ─────────────────────────────────────────────────────────────────────────────
# 5. astream_events — fine-grained event streaming for complex chains
# ─────────────────────────────────────────────────────────────────────────────
async def demo_astream_events():
    print("\n" + "=" * 60)
    print("5. astream_events — observe every event in the chain")
    print("=" * 60)

    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Be concise."),
            ("human", "What is {topic}?"),
        ]).with_config({"run_name": "my_prompt"})
        | llm.with_config({"run_name": "my_llm"})
        | StrOutputParser().with_config({"run_name": "my_parser"})
    )

    print("Events emitted by the chain:")
    print("-" * 40)

    token_count = 0

    # astream_events emits structured events for every step
    async for event in chain.astream_events(
        {"topic": "vector databases"},
        version="v2",
    ):
        event_name = event["event"]
        run_name   = event.get("name", "")

        if event_name == "on_chain_start":
            print(f"  [START] {run_name}")

        elif event_name == "on_llm_stream":
            # Individual token chunks
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)
                token_count += 1

        elif event_name == "on_chain_end":
            if run_name == "my_parser":
                print()  # newline after streaming text
                print(f"  [END]   {run_name}")

    print(f"\n[Total token events: {token_count}]")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Streaming with parallel chains
# ─────────────────────────────────────────────────────────────────────────────
def demo_parallel_streaming():
    print("\n" + "=" * 60)
    print("6. Streaming from parallel chains")
    print("=" * 60)

    from langchain_core.runnables import RunnableParallel

    # Both chains run simultaneously but we collect results (not token-stream here)
    parallel = RunnableParallel(
        summary=(
            ChatPromptTemplate.from_messages([
                ("human", "Summarize {topic} in 2 sentences.")
            ])
            | llm
            | StrOutputParser()
        ),
        example=(
            ChatPromptTemplate.from_messages([
                ("human", "Give one real-world example of {topic}.")
            ])
            | llm
            | StrOutputParser()
        ),
    )

    # For parallel chains, stream returns one dict chunk per completed branch
    topic = "Redis pub/sub"
    print(f"Parallel streaming for topic: {topic}")
    print("-" * 40)

    result = parallel.invoke({"topic": topic})
    print("Summary:\n ", result["summary"])
    print("\nExample:\n ", result["example"])


if __name__ == "__main__":
    demo_sync_stream()
    demo_chain_streaming()
    demo_stream_accumulate()
    demo_parallel_streaming()

    # Async demos
    print("\nRunning async streaming demos...")
    asyncio.run(demo_async_stream())
    asyncio.run(demo_astream_events())

    print("\n\nKey takeaways:")
    print("  - .stream() yields chunks synchronously — good for scripts/CLI")
    print("  - .astream() yields chunks asynchronously — essential for FastAPI/web")
    print("  - StrOutputParser in the chain converts AIMessageChunk → str chunks")
    print("  - astream_events() gives you fine-grained observability of every step")
    print("  - For FastAPI: use StreamingResponse + async generator with .astream()")
    print("  - All Runnables support streaming — it propagates through | chains automatically")
