"""
09 - RAG Pipeline (Retrieval Augmented Generation)
====================================================
RAG = Retrieval + Augmented + Generation

Without RAG: LLM answers from training data → stale, hallucinated, no private data.
With RAG: LLM answers from YOUR retrieved docs → current, grounded, private-data-aware.

The RAG pipeline:
  1. Ingest   : load docs → split → embed → store in vector DB (done once)
  2. Retrieve : embed query → similarity search → top-k docs
  3. Augment  : inject retrieved docs into prompt as context
  4. Generate : LLM answers using the injected context

This file builds a complete RAG system from scratch.

WHEN YOU NEED THIS:
  The #1 use case for LangChain in production. RAG is needed whenever the LLM
  must answer questions about data it was not trained on — private, internal,
  or recent information.

  Real scenarios:
  - Internal knowledge base: "What is our incident response SLA?" — answer
    comes from runbooks, not from the model's training data
  - Customer support bot: answers grounded in actual product docs, not generic text
  - Codebase assistant: questions about a specific repo's architecture and decisions
  - Legal/compliance Q&A: answers sourced from the actual contracts and policies
  - Any domain where hallucination or outdated answers are unacceptable

WITHOUT RAG vs WITH RAG:
  Without: LLM answers from training data → may be outdated, hallucinated,
           or wrong for a specific internal context
  With:    LLM answers from retrieved document chunks → grounded, accurate,
           citable with source (e.g., "kafka-guide.pdf, page 3")

THE PROMPT WORDING IS CRITICAL:
  The model must be instructed: "Answer ONLY from the context below. If the
  answer is not in the context, say so." Without this constraint, the model
  mixes in training data and the grounding benefit of RAG is lost.

CONVERSATIONAL RAG (demo_conversational_rag):
  RAG + memory combined. Relevant docs are retrieved AND prior conversation
  turns are remembered. Required for any multi-turn RAG chatbot where follow-up
  questions refer back to earlier parts of the conversation.

Run:  python 09_rag_pipeline.py
Prereq: ollama pull nomic-embed-text  (or change EMBEDDING_MODEL below)
        ollama pull llama3.2
"""

import tempfile
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL       = "llama3.2"

# ─────────────────────────────────────────────────────────────────────────────
# Knowledge base — pretend this is loaded from files/DBs
# ─────────────────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = [
    """
    Apache Kafka is a distributed event streaming platform originally developed at LinkedIn.
    Kafka uses topics to organize messages. Topics are divided into partitions for parallelism.
    Each partition is an ordered, immutable log of messages. Kafka brokers store the messages.
    Producers publish to topics. Consumers read from topics via consumer groups.
    Kafka guarantees at-least-once delivery. With idempotent producers and transactions,
    it supports exactly-once semantics. Kafka retains messages for a configurable duration
    (default 7 days), even after they are consumed.
    Key use cases: real-time data pipelines, event sourcing, log aggregation, stream processing.
    """,
    """
    Redis is an open-source, in-memory data structure store used as a database, cache, and
    message broker. Redis supports strings, hashes, lists, sets, sorted sets, bitmaps,
    hyperloglogs, and streams. Redis is single-threaded for commands but uses I/O multiplexing.
    Cache patterns: Cache-Aside (read), Write-Through, Write-Behind.
    Redis Cluster provides horizontal scaling. Redis Sentinel provides high availability.
    Always set TTL on cache keys. Use SCAN instead of KEYS in production.
    Key naming convention: service:entity:id (e.g., myapp:user:123:profile).
    """,
    """
    PostgreSQL is a powerful open-source relational database management system.
    It supports ACID transactions, foreign keys, joins, views, triggers, and stored procedures.
    PostgreSQL supports advanced data types including JSONB, arrays, hstore, and geometric types.
    Use UUID for public-facing IDs and BIGSERIAL for internal primary keys.
    Use TIMESTAMPTZ (not TIMESTAMP) for all date/time columns.
    Indexes: B-tree (default), GIN (for JSONB/arrays), GiST, BRIN, Hash.
    Always use EXPLAIN ANALYZE to understand query plans.
    Use connection pooling (PgBouncer) in production. Migrations via Flyway or Alembic.
    """,
    """
    Spring Boot is a Java framework that simplifies building production-ready applications.
    It provides auto-configuration, embedded servers (Tomcat, Jetty), and starter dependencies.
    Use constructor injection (not field injection). Single constructor needs no @Autowired.
    Use @ConfigurationProperties with records for type-safe configuration.
    Use @Valid on controller inputs. Handle exceptions with @RestControllerAdvice.
    Spring Data JPA for relational databases, Spring Data MongoDB for document databases.
    Use @Transactional on service methods. Keep transactions short.
    Spring Boot Actuator provides health checks and metrics endpoints.
    """,
    """
    FastAPI is a modern, fast Python web framework for building APIs with Python 3.12+.
    It uses type hints and Pydantic for automatic data validation.
    FastAPI generates OpenAPI documentation automatically from code.
    Use Depends() for dependency injection (DB sessions, authentication).
    Async endpoints (async def) for I/O-bound work. Sync endpoints for CPU-bound.
    Background tasks with BackgroundTasks for fire-and-forget operations.
    Use Pydantic BaseModel for all request and response schemas.
    FastAPI integrates with SQLAlchemy, Motor (MongoDB), and Redis easily.
    """,
]

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: INGESTION — load, split, embed, store
# ─────────────────────────────────────────────────────────────────────────────
def build_vectorstore() -> FAISS:
    print("Phase 1: Ingesting documents into vector store...")

    # Convert raw text to Document objects
    raw_docs = [
        Document(page_content=text.strip(), metadata={"chunk_id": i})
        for i, text in enumerate(KNOWLEDGE_BASE)
    ]

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        length_function=len,
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"  Raw docs: {len(raw_docs)} → Chunks: {len(chunks)}")

    # Embed and store
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"  Vector store built with {vectorstore.index.ntotal} vectors")
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2+3+4: RETRIEVAL + AUGMENTATION + GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def build_rag_chain(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},  # retrieve top 3 most relevant chunks
    )

    # RAG prompt — instructs the model to use ONLY the provided context
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful technical assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have that information."
Be concise and accurate.

Context:
{context}
"""),
        ("human", "{question}"),
    ])

    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    def format_docs(docs: list) -> str:
        """Format retrieved documents into a single context string."""
        return "\n\n---\n\n".join(
            f"[Source chunk {i+1}]:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

    # LCEL RAG chain
    # Input: {"question": "..."} or just a plain string
    rag_chain = (
        {
            "context": retriever | format_docs,  # retrieve docs and format them
            "question": RunnablePassthrough(),    # pass question through unchanged
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# ─────────────────────────────────────────────────────────────────────────────
# Demo: query the RAG system
# ─────────────────────────────────────────────────────────────────────────────
def demo_basic_rag(rag_chain, retriever):
    print("\n" + "=" * 60)
    print("Basic RAG Q&A")
    print("=" * 60)

    questions = [
        "What is Kafka's retention policy for messages?",
        "What naming convention should I use for Redis keys?",
        "How should I handle exceptions in Spring Boot?",
        "What PostgreSQL column type should I use for timestamps?",
        "What is the difference between async and sync endpoints in FastAPI?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        answer = rag_chain.invoke(question)
        print(f"A: {answer}")
        print("-" * 40)


# ─────────────────────────────────────────────────────────────────────────────
# Advanced: RAG with source citations
# ─────────────────────────────────────────────────────────────────────────────
def demo_rag_with_sources(vectorstore):
    print("\n" + "=" * 60)
    print("RAG with source citations")
    print("=" * 60)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer using the provided context only.
At the end, list the chunk numbers you used as sources.

Context:
{context}
"""),
        ("human", "{question}"),
    ])

    def format_docs_with_ids(docs: list) -> str:
        return "\n\n".join(
            f"[Chunk {i+1} | chunk_id={doc.metadata.get('chunk_id', '?')}]:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

    # Use RunnablePassthrough.assign() to keep both docs and formatted context
    from langchain_core.runnables import RunnablePassthrough

    chain_with_sources = (
        {
            "context": retriever | format_docs_with_ids,
            "question": RunnablePassthrough(),
            "docs": retriever,   # also keep raw docs
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    question = "What are the key differences between Redis and Kafka?"
    print(f"Q: {question}")

    # Get the retrieved docs first
    retrieved = retriever.invoke(question)
    print(f"\nRetrieved {len(retrieved)} chunks:")
    for i, doc in enumerate(retrieved):
        print(f"  [{i+1}] chunk_id={doc.metadata.get('chunk_id')}: {doc.page_content[:70]}...")

    answer = chain_with_sources.invoke(question)
    print(f"\nAnswer:\n{answer}")


# ─────────────────────────────────────────────────────────────────────────────
# Advanced: Conversational RAG — RAG + memory
# ─────────────────────────────────────────────────────────────────────────────
def demo_conversational_rag(vectorstore):
    print("\n" + "=" * 60)
    print("Conversational RAG (RAG + memory)")
    print("=" * 60)

    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.prompts import MessagesPlaceholder

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

    # Prompt includes both history and retrieved context
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful technical assistant.
Use the following context to answer questions.
If unsure, say so.

Context: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": RunnablePassthrough(),  # will be injected by history wrapper
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Simpler approach: separate history from RAG for this demo
    session_store = {}

    def get_history(session_id):
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]

    # For conversational RAG, we build a simpler chain manually
    conv_prompt = ChatPromptTemplate.from_messages([
        ("system", "Technical assistant. Use context:\n\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])

    simple_chain = conv_prompt | llm | StrOutputParser()

    config = {"configurable": {"session_id": "rag-conv"}}
    history = get_history("rag-conv")

    turns = [
        "What is Redis used for?",
        "What data structures does it support?",
        "How should I name my Redis keys?",
    ]

    for question in turns:
        context_docs = retriever.invoke(question)
        context_text = format_docs(context_docs)

        response = simple_chain.invoke({
            "context": context_text,
            "history": history.messages,
            "question": question,
        })

        history.add_user_message(question)
        history.add_ai_message(response)

        print(f"\nQ: {question}")
        print(f"A: {response}")


if __name__ == "__main__":
    vectorstore = build_vectorstore()
    rag_chain, retriever = build_rag_chain(vectorstore)

    demo_basic_rag(rag_chain, retriever)
    demo_rag_with_sources(vectorstore)
    demo_conversational_rag(vectorstore)

    print("\n\nKey takeaways:")
    print("  - RAG = Retrieve relevant chunks + Augment the prompt + Generate answer")
    print("  - The prompt must instruct the model to use ONLY the context")
    print("  - format_docs() is the glue that serializes retrieved docs into context text")
    print("  - RunnablePassthrough() passes the question to both retriever and prompt")
    print("  - Add MessagesPlaceholder + RunnableWithMessageHistory for conversational RAG")
    print("  - Always tune: chunk_size, chunk_overlap, k (top-k), and the prompt wording")


# =============================================================================
# DEEP DIVE: RecursiveCharacterTextSplitter parameters explained
#
# The splitter is configured here in build_vectorstore():
#
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=400,
#         chunk_overlap=80,
#         length_function=len,
#     )
#
# Below each parameter is explained using the actual Kafka entry from KNOWLEDGE_BASE.
#
# The Kafka entry (after .strip()) is ~630 characters long:
#
#   "Apache Kafka is a distributed event streaming platform originally developed at LinkedIn.
#    Kafka uses topics to organize messages. Topics are divided into partitions for parallelism.
#    Each partition is an ordered, immutable log of messages. Kafka brokers store the messages.
#    Producers publish to topics. Consumers read from topics via consumer groups.
#    Kafka guarantees at-least-once delivery. With idempotent producers and transactions,
#    it supports exactly-once semantics. Kafka retains messages for a configurable duration
#    (default 7 days), even after they are consumed.
#    Key use cases: real-time data pipelines, event sourcing, log aggregation, stream processing."
#
# =============================================================================


# ── chunk_size=400 ────────────────────────────────────────────────────────────
#
#   The MAXIMUM number of characters each chunk is allowed to contain.
#   The splitter tries to split at natural boundaries first in this order:
#     paragraphs (\n\n) → newlines (\n) → sentences (". ") → words (" ") → characters
#   It picks the largest boundary that still keeps the chunk under the limit.
#
#   The Kafka text above is ~630 chars — it exceeds 400, so it splits into 2 chunks.
#   The first clean sentence boundary before char 400 is after "...consumer groups."
#   (~347 chars), so CHUNK 1 ends there:
#
#   CHUNK 1  (~347 chars)
#   ┌──────────────────────────────────────────────────────────────────────────┐
#   │ Apache Kafka is a distributed event streaming platform originally        │
#   │ developed at LinkedIn. Kafka uses topics to organize messages. Topics    │
#   │ are divided into partitions for parallelism. Each partition is an        │
#   │ ordered, immutable log of messages. Kafka brokers store the messages.    │
#   │ Producers publish to topics. Consumers read from topics via consumer     │
#   │ groups.                                                                  │
#   └──────────────────────────────────────────────────────────────────────────┘
#
#   CHUNK 2  (remainder of the text)
#   ┌──────────────────────────────────────────────────────────────────────────┐
#   │ Kafka guarantees at-least-once delivery. With idempotent producers and   │
#   │ transactions, it supports exactly-once semantics. Kafka retains messages │
#   │ for a configurable duration (default 7 days), even after they are        │
#   │ consumed. Key use cases: real-time data pipelines, event sourcing, log   │
#   │ aggregation, stream processing.                                           │
#   └──────────────────────────────────────────────────────────────────────────┘
#
#   Why 400 specifically?
#   - Too large (1000+): chunk contains too many sentences → retrieved chunk
#     floods the LLM context with irrelevant information, hurting answer quality.
#   - Too small (50-100): a chunk may not carry a complete thought → retrieval
#     finds technically matching text that lacks enough context to answer.
#   - 300-600 chars is the practical sweet spot for dense technical prose.
#     Tune this based on your document type (code, legal text, chat logs, etc.).


# ── chunk_overlap=80 ──────────────────────────────────────────────────────────
#
#   The number of characters from the END of the previous chunk that are REPEATED
#   at the START of the next chunk.
#   This creates a sliding-window effect so that context is not lost at boundaries.
#
#   In the Kafka example, chunk 1 ends with:
#
#     "...Consumers read from topics via consumer groups."
#                                                        ^
#                                             last sentence of chunk 1
#
#   The last ~80 characters of chunk 1 — roughly that sentence — are copied into
#   the beginning of chunk 2:
#
#   CHUNK 2 (with overlap shown explicitly):
#   ┌──────────────────────────────────────────────────────────────────────────┐
#   │ [OVERLAP]  Consumers read from topics via consumer groups.               │  ← repeated from chunk 1
#   │ [NEW]      Kafka guarantees at-least-once delivery. With idempotent ...  │
#   └──────────────────────────────────────────────────────────────────────────┘
#
#   Why does this matter?
#   Suppose a user asks: "How do consumer groups relate to Kafka guarantees?"
#   The answer spans the boundary between chunk 1 and chunk 2.
#   Without overlap: neither chunk alone contains both pieces of information.
#   With 80-char overlap: chunk 2 opens with the consumer-groups sentence AND
#   contains the guarantees sentence — so a single retrieved chunk answers it.
#
#   Rule of thumb: chunk_overlap = 15-25% of chunk_size.
#   Here: 80 / 400 = 20%.  Increase overlap when your docs have dense cross-
#   sentence dependencies; decrease it to save embedding storage.


# ── length_function=len ───────────────────────────────────────────────────────
#
#   Tells the splitter HOW to measure the "size" of a piece of text.
#   `len` is Python's built-in function — it counts CHARACTERS (bytes for ASCII).
#
#   Example from KNOWLEDGE_BASE:
#
#     text = "Apache Kafka is a distributed event streaming platform originally"
#     len(text)  →  65    (65 characters)
#
#   So with chunk_size=400 and length_function=len, a chunk is "full" when it
#   reaches 400 CHARACTERS, regardless of how many tokens or words that is.
#
#   The alternative: count TOKENS instead of characters.
#
#     from langchain_text_splitters import RecursiveCharacterTextSplitter
#     import tiktoken
#
#     enc = tiktoken.get_encoding("cl100k_base")   # GPT-4 tokenizer
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=150,                           # now means 150 TOKENS
#         chunk_overlap=30,
#         length_function=lambda text: len(enc.encode(text)),
#     )
#
#   Why does the distinction matter?
#   LLMs have TOKEN limits, not character limits.
#   For English prose, 1 token ≈ 4 characters (rough average).
#   So chunk_size=400 chars ≈ 100 tokens.
#
#   With len():   chunk_size=400  →  ~80-120 tokens  (varies by vocabulary)
#   With tokens:  chunk_size=400  →  exactly 400 tokens
#
#   When to use each:
#   - len() (default): fine for most use cases; simple, fast, no tokenizer needed.
#   - token counting: use when you need precise control over how much of the
#     model's context window each chunk consumes (e.g., model has a 512-token
#     input limit and you're injecting top-k=5 chunks into the prompt).


# =============================================================================
# CHUNKING STRATEGY — by file size and document type
#
# The goal of chunking is to break a document into pieces small enough to embed
# and retrieve individually, but large enough to carry a complete thought.
#
# Core problem:
#   You cannot feed an entire document into an LLM prompt.
#   You split it into chunks, embed each chunk, store it in a vector DB.
#   At query time: embed the question → find the most similar chunk vectors
#   → inject only those chunks into the prompt.
#
# Flow:
#   [Full document]
#         |
#         v
#   [RecursiveCharacterTextSplitter]
#         |
#         v
#   [chunk 1] [chunk 2] [chunk 3] ... [chunk N]
#         |
#         v
#   embed each chunk → store as vector → retrieve top-k at query time
#
# =============================================================================


# ── Small files (< 2 KB) ──────────────────────────────────────────────────────
#
#   Examples: a FAQ page, a short config doc, a single API reference section.
#
#   Strategy: keep as one chunk or split into 2-3 large chunks.
#   Settings:
#     chunk_size=800-1000
#     chunk_overlap=100-150
#
#   Why:
#   Small files have tightly coupled content — every sentence relates to the
#   others. Splitting too finely destroys those relationships. A small file
#   that gets cut into 10 tiny chunks means each retrieved chunk is a fragment
#   without enough context to form a useful answer.
#
#   Example: a 900-char FAQ entry on Redis TTL.
#   With chunk_size=400 it splits into 3 chunks. The question "what happens
#   when a Redis key expires and should I worry about it?" needs all 3 chunks
#   to answer properly. With chunk_size=900 it stays as one chunk and a single
#   retrieval hit answers the question completely.


# ── Medium files (2 KB – 50 KB) ───────────────────────────────────────────────
#
#   Examples: a runbook, a product spec, a technical guide section, a policy doc.
#
#   Strategy: standard chunking — one chunk = one coherent thought.
#   Settings:
#     chunk_size=400-600
#     chunk_overlap=80-120  (20% of chunk_size)
#
#   This is the sweet spot used in build_vectorstore() above:
#
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=400,
#         chunk_overlap=80,
#         length_function=len,
#     )
#
#   The splitter respects natural boundaries in this priority order:
#     paragraphs (\n\n) → newlines (\n) → sentences (". ") → words (" ") → chars
#   It takes the largest boundary that still fits under chunk_size.
#
#   The Kafka entry (~630 chars) splits into exactly 2 chunks at chunk_size=400,
#   with the overlap carrying the "consumer groups" sentence into chunk 2 so
#   a question spanning that boundary is still answerable from one chunk.
#   See the chunk_size and chunk_overlap sections above for the full breakdown.


# ── Large files (50 KB – MB range) ───────────────────────────────────────────
#
#   Examples: a PDF manual, a legal contract, an architecture RFC, a textbook.
#
#   Strategy: load page-by-page first, chunk each page, preserve page number
#   in metadata. This is the only way to produce citable answers:
#   "kafka-guide.pdf, page 7" — which is the whole point of RAG in production.
#
#   Settings:
#     chunk_size=500
#     chunk_overlap=100
#
#   How to do it:
#
#     from langchain_community.document_loaders import PyPDFLoader
#
#     loader = PyPDFLoader("kafka-guide.pdf")
#     pages = loader.load()
#     # Each page is a Document with metadata={"page": 3, "source": "kafka-guide.pdf"}
#
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     chunks = splitter.split_documents(pages)
#     # Each chunk inherits {"page": 3, "source": "kafka-guide.pdf"} from its page
#
#   The metadata survives into the vector store. When you retrieve chunks you
#   can surface citations directly:
#
#     for doc in retrieved_chunks:
#         source = doc.metadata.get("source")   # "kafka-guide.pdf"
#         page   = doc.metadata.get("page")     # 7
#         print(f"[{source}, page {page}]: {doc.page_content[:80]}...")
#
#   Why page numbers matter:
#   Without page metadata, a correct answer has no citation — unverifiable and
#   unusable in regulated domains (legal, compliance, internal policy Q&A).
#   With page metadata, the answer becomes: "According to kafka-guide.pdf page 7,
#   consumer groups allow parallel reads across partitions."
#
#   For very large PDFs (100+ pages), consider chunking per section heading
#   instead of per page using MarkdownHeaderTextSplitter or similar, so that
#   the metadata carries a section name ("3.2 Consumer Groups") in addition
#   to the page number.


# ── Failure modes to avoid ────────────────────────────────────────────────────
#
#   Chunks too large (1000+ chars):
#     Each chunk floods the LLM with irrelevant sentences.
#     The LLM sees the right passage buried in noise → answer quality drops.
#
#   Chunks too small (< 100 chars):
#     Each chunk is a fragment without a complete thought.
#     Retrieval finds the right area of the document but can't form an answer.
#
#   Zero overlap:
#     A fact that spans a boundary disappears from both chunks.
#     The retriever finds neither chunk as the best match → the answer is missed.
#
#   No metadata on large files:
#     Correct answer, no citation. Unverifiable in production.
#     Always preserve source and page at ingestion time, not after.


# ── Quick reference ───────────────────────────────────────────────────────────
#
#   File size     chunk_size    chunk_overlap    notes
#   ──────────    ──────────    ─────────────    ──────────────────────────────
#   Small         800-1000      100-150          fewer, larger chunks
#   Medium        400-600       80-120           standard; one chunk = one idea
#   Large (PDF)   400-500       80-100           page metadata required
#
#   chunk_overlap = ~20% of chunk_size  (rule of thumb for all sizes)
#   k (top-k retrieved at query time)  = 3-5
#
#   Tune chunk_size first. If answers are noisy → smaller chunks.
#   If answers are incomplete → larger chunks or more overlap.
# =============================================================================
