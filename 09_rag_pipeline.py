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
