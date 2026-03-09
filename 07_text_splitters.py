"""
07 - Text Splitters
====================
LLMs have token limits (context windows). You can't feed a 100-page PDF
into a model. Text splitters break large documents into overlapping chunks
that fit within the model's context and preserve enough context for coherence.

Core concept:
  chunk_size    : max tokens/chars per chunk
  chunk_overlap : how much to repeat between consecutive chunks
                  (ensures context isn't lost at boundaries)

Splitters covered:
  - RecursiveCharacterTextSplitter  : best general-purpose (tries \n\n → \n → " " → "")
  - CharacterTextSplitter           : single separator
  - TokenTextSplitter               : split by actual token count (tiktoken)
  - MarkdownHeaderTextSplitter      : splits by heading hierarchy
  - PythonCodeTextSplitter          : AST-aware splits for Python code

WHEN YOU NEED THIS:
  Step 2 of any RAG pipeline, right after loading documents. A 200-page PDF
  cannot be embedded as one blob — it must be split into ~500 token chunks,
  each embedded separately. Think of it like pagination, but the goal is
  fitting within an LLM's context window rather than a screen.

  Real scenario: an 80-page technical spec, user asks about "authentication
  flow on page 34". Splitting lets retrieval surface just 2-3 relevant chunks
  instead of dumping all 80 pages into the prompt.

WHY chunk_overlap EXISTS:
  A sentence at the boundary of a chunk carries context from the previous
  paragraph. Overlap repeats the tail of chunk N at the head of chunk N+1.
  Without it, answers near split boundaries are incoherent because the model
  is missing the surrounding context — similar to reading a book with random
  pages torn out.

WHICH SPLITTER TO USE:
  - RecursiveCharacterTextSplitter     → default for 90% of cases (prose, docs)
  - MarkdownHeaderTextSplitter         → Markdown docs; adds section title to
                                         metadata so retrieval knows which section matched
  - from_tiktoken_encoder (token-based) → exact token control to match a model's
                                          context window precisely
  - from_language(Language.PYTHON)     → code-aware splitting at function/class
                                         boundaries, never mid-function

TUNING GUIDE (for RAG):
  chunk_size=500–1000 tokens, chunk_overlap=50–200 tokens is the typical range.
  Smaller chunks = more precise retrieval. Larger chunks = more context per hit.
  Start at 600/120 and tune based on retrieval quality.

Run:  python 07_text_splitters.py
"""

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sample long text
# ─────────────────────────────────────────────────────────────────────────────
LONG_TEXT = """
Apache Kafka is a distributed event streaming platform capable of handling
trillions of events a day. Originally developed at LinkedIn, Kafka was
open-sourced in 2011 and is now maintained by the Apache Software Foundation.

Kafka is designed around the following core concepts:

Topics: A topic is a category or feed name to which records are published.
Topics in Kafka are always multi-subscriber — a topic can have zero, one,
or many consumers that subscribe to the data written to it.

Partitions: Topics are split into partitions to allow for parallelism.
Each partition is an ordered, immutable sequence of records. Records are
appended to the end of the partition log. Each record in a partition is
assigned a sequential ID number called an offset.

Producers: Producers publish data to the topics of their choice. The producer
is responsible for choosing which record to assign to which partition within
the topic. This can be done in a round-robin fashion simply to balance load,
or it can be done according to some semantic partition function.

Consumers: Consumers label themselves with a consumer group name, and each
record published to a topic is delivered to one consumer instance within each
subscribing consumer group. Consumer instances can be in separate processes
or on separate machines.

Brokers: Kafka is run as a cluster on one or more servers that can span
multiple datacenters or cloud regions. The Kafka cluster stores streams of
records in categories called topics. The Kafka cluster durably persists all
published records — whether or not they have been consumed — using a
configurable retention period.

Zookeeper vs KRaft: Historically, Kafka used Apache ZooKeeper for cluster
coordination and metadata management. Starting with Kafka 2.8, a new
consensus protocol called KRaft was introduced to eliminate the ZooKeeper
dependency, simplifying operations and improving scalability.
"""

MARKDOWN_TEXT = """
# Kafka Guide

## Introduction
Apache Kafka is a distributed event streaming platform.

### What is Kafka?
Kafka handles trillions of events per day with high throughput.

## Core Concepts

### Topics
Topics are categories for messages. They are partitioned for parallelism.

### Partitions
Each partition is an ordered, immutable log of records.

### Producers and Consumers
Producers write to topics. Consumers read from topics using consumer groups.

## Operations

### Installation
Install Kafka from the Apache website or use Docker.

### Configuration
Edit server.properties to configure brokers.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. RecursiveCharacterTextSplitter — best default choice
# ─────────────────────────────────────────────────────────────────────────────
def demo_recursive_splitter():
    print("\n" + "=" * 60)
    print("1. RecursiveCharacterTextSplitter (recommended default)")
    print("=" * 60)

    # Tries to split on ["\n\n", "\n", " ", ""] in order
    # Falls back to smaller separators only when chunk_size isn't satisfied
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,      # target max chars per chunk
        chunk_overlap=50,    # overlap to preserve boundary context
        length_function=len, # char count (not token count)
        is_separator_regex=False,
    )

    chunks = splitter.split_text(LONG_TEXT)

    print(f"Original length : {len(LONG_TEXT)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Avg chunk size  : {sum(len(c) for c in chunks) // len(chunks)} chars")

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        print(f"  {chunk[:100].strip()}...")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Splitting Documents (not just strings) — metadata is preserved
# ─────────────────────────────────────────────────────────────────────────────
def demo_split_documents():
    print("\n" + "=" * 60)
    print("2. split_documents() — metadata is carried over to each chunk")
    print("=" * 60)

    docs = [
        Document(
            page_content=LONG_TEXT,
            metadata={"source": "kafka-guide.txt", "author": "team", "page": 1},
        )
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
    )

    # split_documents returns List[Document] with metadata copied to each chunk
    chunks = splitter.split_documents(docs)

    print(f"Input docs  : {len(docs)}")
    print(f"Output chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  Metadata    : {chunk.metadata}")
        print(f"  Content     : {chunk.page_content[:80].strip()}...")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CharacterTextSplitter — single-separator split
# ─────────────────────────────────────────────────────────────────────────────
def demo_character_splitter():
    print("\n" + "=" * 60)
    print("3. CharacterTextSplitter — explicit single separator")
    print("=" * 60)

    # Only splits on the given separator — less flexible than recursive
    splitter = CharacterTextSplitter(
        separator="\n\n",    # split only on double newlines (paragraphs)
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    chunks = splitter.split_text(LONG_TEXT)
    print(f"Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n  Chunk {i+1} ({len(chunk)} chars): {chunk[:80].strip()}...")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Token-based splitting — precise control for LLM context windows
# ─────────────────────────────────────────────────────────────────────────────
def demo_token_splitter():
    print("\n" + "=" * 60)
    print("4. Token-based splitting (tiktoken)")
    print("=" * 60)

    # For RAG: split by token count to match your LLM's context window exactly
    # from_tiktoken_encoder uses OpenAI's tokenizer (cl100k_base)
    # but token counts are similar enough for Ollama models
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",   # tokenizer to use for counting
        chunk_size=100,        # max TOKENS per chunk
        chunk_overlap=20,      # overlap in TOKENS
    )

    chunks = splitter.split_text(LONG_TEXT)
    print(f"Chunks with token-based splitting: {len(chunks)}")
    print(f"First chunk: {chunks[0][:150].strip()}...")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MarkdownHeaderTextSplitter — structure-aware splitting
# ─────────────────────────────────────────────────────────────────────────────
def demo_markdown_splitter():
    print("\n" + "=" * 60)
    print("5. MarkdownHeaderTextSplitter — hierarchy-aware")
    print("=" * 60)

    # Splits at markdown headers and adds them to metadata
    headers_to_split_on = [
        ("#",  "h1"),
        ("##", "h2"),
        ("###","h3"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # keep headers in content
    )

    chunks = splitter.split_text(MARKDOWN_TEXT)

    print(f"Chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"\n  Metadata : {chunk.metadata}")
        print(f"  Content  : {chunk.page_content[:80].strip()}...")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Code splitter — language-aware AST splitting
# ─────────────────────────────────────────────────────────────────────────────
def demo_code_splitter():
    print("\n" + "=" * 60)
    print("6. Code splitter — Python-aware splitting")
    print("=" * 60)

    python_code = '''
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number recursively."""
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)


class FibonacciCalculator:
    """Efficient Fibonacci calculator with memoization."""

    def __init__(self):
        self._cache: dict[int, int] = {}

    def calculate(self, n: int) -> int:
        """Calculate using dynamic programming."""
        if n in self._cache:
            return self._cache[n]
        if n <= 1:
            return n
        result = self.calculate(n - 1) + self.calculate(n - 2)
        self._cache[n] = result
        return result

    def clear_cache(self) -> None:
        """Clear the memoization cache."""
        self._cache.clear()


def main():
    calc = FibonacciCalculator()
    for i in range(10):
        print(f"fib({i}) = {calc.calculate(i)}")
'''

    # RecursiveCharacterTextSplitter with Language enum uses language-specific separators
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=200,
        chunk_overlap=30,
    )

    chunks = splitter.split_text(python_code)
    print(f"Code chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i+1}:\n{chunk[:150]}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Chunking strategy comparison — visualize the impact
# ─────────────────────────────────────────────────────────────────────────────
def demo_strategy_comparison():
    print("\n" + "=" * 60)
    print("7. Chunking strategy comparison")
    print("=" * 60)

    configs = [
        ("Small chunks, no overlap",    {"chunk_size": 200, "chunk_overlap": 0}),
        ("Small chunks, 20% overlap",   {"chunk_size": 200, "chunk_overlap": 40}),
        ("Large chunks, 10% overlap",   {"chunk_size": 600, "chunk_overlap": 60}),
    ]

    for name, kwargs in configs:
        splitter = RecursiveCharacterTextSplitter(**kwargs)
        chunks = splitter.split_text(LONG_TEXT)
        avg = sum(len(c) for c in chunks) // len(chunks)
        print(f"\n  {name}:")
        print(f"    chunk_size={kwargs['chunk_size']}, overlap={kwargs['chunk_overlap']}")
        print(f"    → {len(chunks)} chunks, avg size: {avg} chars")


if __name__ == "__main__":
    demo_recursive_splitter()
    demo_split_documents()
    demo_character_splitter()
    demo_token_splitter()
    demo_markdown_splitter()
    demo_code_splitter()
    demo_strategy_comparison()

    print("\n\nKey takeaways:")
    print("  - RecursiveCharacterTextSplitter is the go-to default for most text")
    print("  - chunk_overlap prevents losing context at chunk boundaries")
    print("  - split_documents() preserves metadata — use it instead of split_text()")
    print("  - Token-based splitting gives exact control for LLM context windows")
    print("  - MarkdownHeaderTextSplitter adds section structure to metadata")
    print("  - For RAG: typical values are chunk_size=500-1000, overlap=50-200 tokens")
