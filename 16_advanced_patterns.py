"""
16 - Advanced Patterns
=======================
Production patterns for building robust LangChain applications:

  1. Retry with exponential backoff
  2. Parallel multi-chain fan-out with aggregation
  3. Router chain — dynamic chain selection
  4. Self-critique chain — model reviews its own output
  5. Map-reduce over documents
  6. Hypothetical Document Embedding (HyDE) for better RAG
  7. Multi-query retrieval — generate multiple queries for richer retrieval
  8. Rate limiting and throttling

WHEN YOU NEED THIS:
  The basic RAG chain or agent is built and real production problems are
  appearing. This file is the hardening chapter — each pattern solves a
  specific problem that shows up at scale.

  Pattern → Problem it solves:

  RETRY (demo_retry):
    Cloud APIs (Groq, OpenAI) rate-limit requests under load. Without retry,
    the chain crashes on transient failures. .with_retry() adds exponential
    backoff in one line — same concept as Resilience4j's @Retry in Java or
    tenacity in Python.

  MAP-REDUCE (demo_map_reduce):
    50 documents need to be summarized. They don't fit in a single prompt.
    Map = summarize each doc independently. Reduce = combine summaries into
    a final output. Same mental model as MapReduce, Spark, or parallel streams.

  SELF-CRITIQUE (demo_self_critique):
    Output quality is inconsistent and errors are unacceptable (legal, medical,
    customer-facing content). Draft → critique → improve pipeline. Costs 3 LLM
    calls instead of 1, but output quality improves measurably.

  MULTI-QUERY RETRIEVAL (demo_multi_query_rag):
    RAG returns irrelevant results because query wording doesn't match how
    documents are phrased. Generate 3 alternative phrasings, retrieve for all,
    deduplicate results. More diverse coverage, better recall.

  HYDE (demo_hyde):
    Query embeddings and document embeddings sit in slightly different semantic
    spaces — queries are short questions, documents are long explanations.
    Generate a hypothetical answer first and embed that instead. The hypothetical
    answer is closer in embedding space to real documents.

  ASYNC PARALLEL (demo_async_parallel):
    5 sections need to be summarized. Sequential = 5x slower than necessary.
    asyncio.gather() fires all calls simultaneously — same idea as
    CompletableFuture.allOf() in Java or asyncio.gather() in Python async code.

Run:  python 16_advanced_patterns.py
"""

import asyncio
import time
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_ollama import ChatOllama, OllamaEmbeddings

MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
llm = ChatOllama(model=MODEL, temperature=0)
creative_llm = ChatOllama(model=MODEL, temperature=0.7)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Retry logic — handle transient failures gracefully
# ─────────────────────────────────────────────────────────────────────────────
def demo_retry():
    print("\n" + "=" * 60)
    print("1. Retry with .with_retry()")
    print("=" * 60)

    # .with_retry() adds automatic retry with exponential backoff
    # Essential for production use of cloud APIs (rate limits, transient errors)
    resilient_llm = llm.with_retry(
        retry_if_exception_type=(ConnectionError, TimeoutError),
        wait_exponential_jitter=True,
        stop_after_attempt=3,
    )

    chain = (
        ChatPromptTemplate.from_messages([("human", "What is {topic}?")])
        | resilient_llm
        | StrOutputParser()
    )

    result = chain.invoke({"topic": "idempotency in distributed systems"})
    print(f"Result (with auto-retry): {result[:150]}...")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Map-reduce over documents — process many docs, summarize
# ─────────────────────────────────────────────────────────────────────────────
def demo_map_reduce():
    print("\n" + "=" * 60)
    print("2. Map-Reduce pattern — summarize many documents")
    print("=" * 60)

    documents = [
        Document(page_content="Kafka enables real-time data streaming with high throughput. It decouples producers and consumers via topics and partitions.", metadata={"source": "kafka-docs"}),
        Document(page_content="Redis is an in-memory database excellent for caching. It supports TTL, pub/sub, and sorted sets for real-time leaderboards.", metadata={"source": "redis-docs"}),
        Document(page_content="PostgreSQL offers ACID compliance, JSONB support, and excellent indexing. It's the default choice for relational workloads.", metadata={"source": "postgres-docs"}),
        Document(page_content="Kubernetes orchestrates containers with auto-scaling, rolling updates, and health checks. It's the de-facto container orchestration platform.", metadata={"source": "k8s-docs"}),
    ]

    # MAP step — summarize each document independently
    map_chain = (
        ChatPromptTemplate.from_messages([
            ("human", "In one sentence, what is the core purpose of this technology?\n\n{content}")
        ])
        | llm
        | StrOutputParser()
    )

    print(f"MAP step — summarizing {len(documents)} documents:")
    summaries = []
    for doc in documents:
        summary = map_chain.invoke({"content": doc.page_content})
        summaries.append(summary)
        print(f"  [{doc.metadata['source']:15}]: {summary[:80]}...")

    # REDUCE step — combine all summaries into a cohesive final output
    reduce_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You synthesize technical summaries into cohesive insights."),
            ("human", (
                "Given these technology summaries, write a 3-sentence architecture recommendation:\n\n"
                "{summaries}"
            )),
        ])
        | llm
        | StrOutputParser()
    )

    combined = "\n".join(f"- {s}" for s in summaries)
    final = reduce_chain.invoke({"summaries": combined})

    print(f"\nREDUCE step — final synthesis:")
    print(final)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Self-critique chain — model critiques and improves its output
# ─────────────────────────────────────────────────────────────────────────────
def demo_self_critique():
    print("\n" + "=" * 60)
    print("3. Self-critique chain — draft → critique → improve")
    print("=" * 60)

    # Step 1: Draft initial answer
    draft_chain = (
        ChatPromptTemplate.from_messages([
            ("human", "Explain {topic} in 3 sentences for a junior developer.")
        ])
        | creative_llm
        | StrOutputParser()
    )

    # Step 2: Critique the draft
    critique_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a senior engineer doing a technical review."),
            ("human", (
                "Review this explanation:\n\n{draft}\n\n"
                "Identify: accuracy issues, missing context, unclear terms. "
                "Be specific. 2-3 bullet points max."
            )),
        ])
        | llm
        | StrOutputParser()
    )

    # Step 3: Improve based on critique
    improve_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Improve technical explanations based on review feedback."),
            ("human", (
                "Original:\n{draft}\n\n"
                "Review feedback:\n{critique}\n\n"
                "Write an improved version addressing the feedback. 3 sentences."
            )),
        ])
        | llm
        | StrOutputParser()
    )

    topic = "database connection pooling"

    draft    = draft_chain.invoke({"topic": topic})
    critique = critique_chain.invoke({"draft": draft})
    improved = improve_chain.invoke({"draft": draft, "critique": critique})

    print(f"Topic   : {topic}")
    print(f"\nDraft   :\n  {draft}")
    print(f"\nCritique:\n  {critique}")
    print(f"\nImproved:\n  {improved}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-query retrieval — richer RAG with diverse queries
# ─────────────────────────────────────────────────────────────────────────────
def demo_multi_query_rag():
    print("\n" + "=" * 60)
    print("4. Multi-query retrieval — richer RAG context")
    print("=" * 60)

    from langchain_community.vectorstores import FAISS
    from langchain_core.output_parsers import CommaSeparatedListOutputParser

    # Build a small vector store
    docs = [
        Document(page_content="Kafka topics are partitioned for parallel processing. Each partition is ordered.", metadata={"id": 1}),
        Document(page_content="Kafka consumer groups allow multiple consumers to divide topic partitions among themselves.", metadata={"id": 2}),
        Document(page_content="Kafka producers can be configured for at-least-once or exactly-once delivery semantics.", metadata={"id": 3}),
        Document(page_content="Kafka retention is configured per topic. Default is 7 days. Can be size-based too.", metadata={"id": 4}),
        Document(page_content="Kafka Connect integrates Kafka with external systems via source and sink connectors.", metadata={"id": 5}),
    ]

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Generate multiple variations of the query
    query_gen_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Generate 3 different search queries for the same information need. Output as comma-separated list."),
            ("human", "Original question: {question}\n\nGenerate 3 alternative phrasings:"),
        ])
        | llm
        | CommaSeparatedListOutputParser()
    )

    def multi_retrieve(question: str) -> List[Document]:
        """Retrieve docs using multiple query variations, deduplicate by content."""
        queries = query_gen_chain.invoke({"question": question})
        queries.append(question)  # include original

        print(f"  Generated {len(queries)} query variants:")
        for q in queries:
            print(f"    - {q.strip()[:80]}")

        seen_contents = set()
        all_docs = []
        for query in queries:
            for doc in retriever.invoke(query.strip()):
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    all_docs.append(doc)

        return all_docs

    question = "How does Kafka handle message delivery guarantees?"
    retrieved = multi_retrieve(question)

    print(f"\nRetrieved {len(retrieved)} unique chunks (vs {2} with single query):")
    for doc in retrieved:
        print(f"  [id={doc.metadata['id']}]: {doc.page_content[:80]}...")


# ─────────────────────────────────────────────────────────────────────────────
# 5. HyDE — Hypothetical Document Embedding for better semantic search
# ─────────────────────────────────────────────────────────────────────────────
def demo_hyde():
    print("\n" + "=" * 60)
    print("5. HyDE — Hypothetical Document Embedding")
    print("=" * 60)

    from langchain_community.vectorstores import FAISS

    print("HyDE explanation:")
    print("  Problem  : Query 'how does X work?' is semantically different from")
    print("             a document that explains X — they're in different 'embedding spaces'")
    print("  Solution : Generate a HYPOTHETICAL answer first, then embed THAT for retrieval")
    print("             The hypothetical answer is semantically closer to real documents")

    docs = [
        Document(page_content="PostgreSQL uses MVCC (Multi-Version Concurrency Control) to handle concurrent transactions without locking. Each transaction sees a consistent snapshot.", metadata={}),
        Document(page_content="In MVCC, each write creates a new row version rather than overwriting. Old versions are cleaned up by VACUUM.", metadata={}),
        Document(page_content="MVCC allows readers to not block writers and writers to not block readers in PostgreSQL.", metadata={}),
    ]

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # HyDE: generate hypothetical answer, embed it, use for retrieval
    hyde_chain = (
        ChatPromptTemplate.from_messages([
            ("human", (
                "Write a 2-sentence technical explanation that would answer: {question}\n"
                "Write as if you are the document that contains this answer."
            )),
        ])
        | llm
        | StrOutputParser()
    )

    question = "How does PostgreSQL handle concurrent reads and writes?"

    # Standard retrieval — embed the question directly
    standard_docs = vectorstore.similarity_search(question, k=2)

    # HyDE retrieval — embed the hypothetical answer
    hypothetical_answer = hyde_chain.invoke({"question": question})
    print(f"\nHypothetical answer:\n  {hypothetical_answer}")

    hyde_docs = vectorstore.similarity_search(hypothetical_answer, k=2)

    print(f"\nStandard retrieval (embed question):")
    for doc in standard_docs:
        print(f"  - {doc.page_content[:80]}...")

    print(f"\nHyDE retrieval (embed hypothetical answer):")
    for doc in hyde_docs:
        print(f"  - {doc.page_content[:80]}...")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Async parallel execution — fan-out to multiple LLM calls simultaneously
# ─────────────────────────────────────────────────────────────────────────────
async def demo_async_parallel():
    print("\n" + "=" * 60)
    print("6. Async parallel LLM calls — fan-out pattern")
    print("=" * 60)

    chain = (
        ChatPromptTemplate.from_messages([
            ("human", "In one sentence: what is {topic}?")
        ])
        | llm
        | StrOutputParser()
    )

    topics = ["Redis", "Kafka", "PostgreSQL", "Kubernetes", "Docker"]

    # Sequential timing
    start = time.time()
    sequential_results = []
    for topic in topics:
        result = await chain.ainvoke({"topic": topic})
        sequential_results.append(result)
    seq_time = time.time() - start

    # Parallel timing — all calls go out simultaneously
    start = time.time()
    parallel_results = await asyncio.gather(*[
        chain.ainvoke({"topic": topic}) for topic in topics
    ])
    par_time = time.time() - start

    print(f"Sequential: {seq_time:.2f}s for {len(topics)} calls")
    print(f"Parallel  : {par_time:.2f}s for {len(topics)} calls")
    print(f"Speedup   : {seq_time/par_time:.1f}x")

    print("\nResults:")
    for topic, result in zip(topics, parallel_results):
        print(f"  [{topic:12}]: {result[:80]}...")


if __name__ == "__main__":
    demo_retry()
    demo_map_reduce()
    demo_self_critique()
    demo_multi_query_rag()
    demo_hyde()
    asyncio.run(demo_async_parallel())

    print("\n\nKey takeaways:")
    print("  - .with_retry() adds production-grade resilience in one line")
    print("  - Map-Reduce: process N docs in parallel (map), then combine (reduce)")
    print("  - Self-critique: draft → critique → improve produces higher quality output")
    print("  - Multi-query: generate query variants → richer, more diverse retrieval")
    print("  - HyDE: embed a hypothetical answer instead of the query for better search")
    print("  - asyncio.gather() runs multiple LLM calls in parallel — huge speedup")
    print("  - These patterns compose: e.g., async + multi-query + self-critique")
