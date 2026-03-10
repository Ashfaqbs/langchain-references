"""
08 - Embeddings & Vector Stores
=================================
Embeddings convert text into dense vectors (lists of floats).
Similar texts have similar vectors — this is what makes semantic search possible.

Vector stores index these vectors and let you:
  - add_documents()     : store text + vectors
  - similarity_search() : find most similar docs to a query
  - as_retriever()      : wrap as a LangChain Retriever for RAG chains

Embedding models (used here):
  - OllamaEmbeddings         : local, free, no API key needed
  - HuggingFaceEmbeddings    : local sentence-transformers

Vector stores (covered here):
  - Chroma    : persistent, full-featured, great for local dev
  - FAISS     : in-memory, blazing fast, good for smaller corpora

WHEN YOU NEED THIS:
  Step 3 of any RAG pipeline — after splitting, every chunk is embedded and
  stored. Think of it like building a search index (Elasticsearch, Solr), but
  instead of keyword matching it does semantic (meaning-based) search.

  Real scenario: "how does Kafka handle failures?" — knowledge base has 500
  chunks. Without embeddings, all 500 would need to be read. With embeddings,
  the 3 most relevant chunks are found in milliseconds.

WHY SEMANTIC SEARCH MATTERS OVER KEYWORD SEARCH:
  "message bus" and "event streaming platform" both match a Kafka document
  even though neither phrase appears word-for-word. A SQL LIKE '%kafka%' would
  miss this. Semantic search understands meaning, not just character patterns.

FAISS vs CHROMA:
  FAISS  → in-memory only, no persistence, rebuilds on restart. Good for
           scripts, one-off analysis, prototypes, small corpora (<100K docs)
  Chroma → persists to disk, supports metadata filtering (topic='kafka'),
           survives restarts. Better for ongoing apps and production use

CRITICAL RULE — always use the same embedding model:
  The model used to embed documents MUST be the same model used to embed queries.
  Embedding with nomic-embed-text and querying with llama3.2 produces vectors in
  different spaces — similarity search returns garbage. Same model, always.

as_retriever() — why it matters:
  Wraps any vector store as a LangChain Runnable so it plugs directly into
  LCEL chains with |. This is the bridge between the vector store and the
  RAG chain built in file 09.

Run:  python 08_embeddings_vectorstores.py
"""

import os
import tempfile
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "nomic-embed-text"  # pull with: ollama pull nomic-embed-text
# Fallback: "mxbai-embed-large", "llama3.2" (some models support embedding)

# ─────────────────────────────────────────────────────────────────────────────
# Sample documents — our "knowledge base"
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_DOCS = [
    Document(
        page_content="Kafka is a distributed event streaming platform for high-throughput data pipelines. It uses topics and partitions for scalability.",
        metadata={"source": "kafka-guide", "topic": "kafka"},
    ),
    Document(
        page_content="Redis is an in-memory data structure store. It supports strings, hashes, lists, sets, and sorted sets. Used for caching and session management.",
        metadata={"source": "redis-guide", "topic": "redis"},
    ),
    Document(
        page_content="PostgreSQL is a powerful open-source relational database. It supports JSONB, full-text search, and advanced indexing strategies like GIN and GiST.",
        metadata={"source": "postgres-guide", "topic": "postgresql"},
    ),
    Document(
        page_content="Spring Boot simplifies Java microservice development with auto-configuration. It includes embedded servers and production-ready actuator endpoints.",
        metadata={"source": "spring-guide", "topic": "spring-boot"},
    ),
    Document(
        page_content="FastAPI is a modern Python web framework for building APIs. It offers automatic OpenAPI documentation and type-based validation using Pydantic.",
        metadata={"source": "fastapi-guide", "topic": "fastapi"},
    ),
    Document(
        page_content="Docker is a containerization platform. It packages applications and dependencies into portable containers that run consistently across environments.",
        metadata={"source": "docker-guide", "topic": "docker"},
    ),
    Document(
        page_content="Kubernetes orchestrates containerized applications. It manages scaling, rolling updates, service discovery, and self-healing for distributed systems.",
        metadata={"source": "k8s-guide", "topic": "kubernetes"},
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. OllamaEmbeddings — generate embedding vectors
# ─────────────────────────────────────────────────────────────────────────────
def demo_embeddings():
    print("\n" + "=" * 60)
    print("1. OllamaEmbeddings — generate embedding vectors")
    print("=" * 60)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Embed a single query
    query = "How does Kafka handle high throughput?"
    vector = embeddings.embed_query(query)

    print(f"Query         : {query}")
    print(f"Vector dims   : {len(vector)}")
    print(f"First 5 values: {[round(v, 4) for v in vector[:5]]}")

    # Embed multiple documents at once
    texts = [doc.page_content for doc in SAMPLE_DOCS[:3]]
    doc_vectors = embeddings.embed_documents(texts)

    print(f"\nDocuments embedded: {len(doc_vectors)}")
    print(f"Each vector dims  : {len(doc_vectors[0])}")

    # Compute cosine similarity manually to understand what's happening
    import math

    def cosine_similarity(a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x ** 2 for x in a))
        mag_b = math.sqrt(sum(x ** 2 for x in b))
        return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

    query_vec = embeddings.embed_query("streaming platform for events")
    kafka_vec = embeddings.embed_query("Kafka topics and partitions")
    redis_vec  = embeddings.embed_query("Redis caching and session store")

    sim_kafka = cosine_similarity(query_vec, kafka_vec)
    sim_redis = cosine_similarity(query_vec, redis_vec)

    print(f"\nCosine similarity 'streaming platform' ↔ 'Kafka topics': {sim_kafka:.4f}")
    print(f"Cosine similarity 'streaming platform' ↔ 'Redis caching': {sim_redis:.4f}")
    print("→ Higher similarity = more semantically related")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FAISS vector store — in-memory, fast
# ─────────────────────────────────────────────────────────────────────────────
def demo_faiss():
    print("\n" + "=" * 60)
    print("2. FAISS — in-memory vector store")
    print("=" * 60)

    from langchain_community.vectorstores import FAISS

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Build index from documents
    vectorstore = FAISS.from_documents(SAMPLE_DOCS, embeddings)

    print(f"Documents indexed: {len(SAMPLE_DOCS)}")

    # Basic similarity search
    query = "distributed messaging and event streaming"
    results = vectorstore.similarity_search(query, k=3)

    print(f"\nQuery: '{query}'")
    print(f"Top {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"\n  [{i}] Source: {doc.metadata['source']}")
        print(f"       Content: {doc.page_content[:100]}...")

    # Similarity search with scores
    print("\n\nWith relevance scores (lower = more similar for L2, higher for cosine):")
    results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
    for doc, score in results_with_scores:
        print(f"  Score {score:.4f} | {doc.metadata['topic']:12} | {doc.page_content[:60]}...")

    # Save and reload (FAISS supports local persistence)
    with tempfile.TemporaryDirectory() as tmp:
        vectorstore.save_local(tmp)
        loaded = FAISS.load_local(tmp, embeddings, allow_dangerous_deserialization=True)
        reload_results = loaded.similarity_search(query, k=1)
        print(f"\nAfter save/reload — top result: {reload_results[0].metadata['topic']}")

    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# 3. Chroma vector store — persistent, full-featured
# ─────────────────────────────────────────────────────────────────────────────
def demo_chroma():
    print("\n" + "=" * 60)
    print("3. Chroma — persistent vector store")
    print("=" * 60)

    from langchain import Chroma

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    with tempfile.TemporaryDirectory() as persist_dir:
        # Create and populate the store
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings,
            persist_directory=persist_dir,     # saves to disk automatically
            collection_name="tech-knowledge",
        )

        print(f"Collection   : tech-knowledge")
        print(f"Doc count    : {vectorstore._collection.count()}")

        # Similarity search
        query = "container orchestration and scaling"
        results = vectorstore.similarity_search(query, k=2)
        print(f"\nQuery: '{query}'")
        for doc in results:
            print(f"  → {doc.metadata['topic']}: {doc.page_content[:80]}...")

        # Filtered search — only search within a specific topic subset
        print("\nFiltered search (only docker/kubernetes topics):")
        filtered = vectorstore.similarity_search(
            "deployment and orchestration",
            k=3,
            filter={"topic": {"$in": ["docker", "kubernetes"]}},
        )
        for doc in filtered:
            print(f"  → {doc.metadata['topic']}: {doc.page_content[:80]}...")

        # Add more documents to existing store
        new_docs = [
            Document(
                page_content="MongoDB is a document database with flexible schema. Use it for JSON-like data with nested structures.",
                metadata={"source": "mongo-guide", "topic": "mongodb"},
            )
        ]
        vectorstore.add_documents(new_docs)
        print(f"\nAfter add — doc count: {vectorstore._collection.count()}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Retriever interface — bridge between vector store and chains
# ─────────────────────────────────────────────────────────────────────────────
def demo_retriever(vectorstore):
    print("\n" + "=" * 60)
    print("4. Retriever interface — plug into LCEL chains")
    print("=" * 60)

    # as_retriever() wraps any vector store as a Retriever
    # A Retriever is a Runnable: takes a string query, returns List[Document]
    retriever = vectorstore.as_retriever(
        search_type="similarity",      # "similarity", "mmr", "similarity_score_threshold"
        search_kwargs={"k": 3},        # return top 3 docs
    )

    query = "Python web framework for REST APIs"
    docs = retriever.invoke(query)

    print(f"Query  : {query}")
    print(f"Results: {len(docs)} documents")
    for doc in docs:
        print(f"  → {doc.metadata['topic']}")

    # MMR retriever — Maximum Marginal Relevance (reduces redundancy)
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
    )
    mmr_docs = mmr_retriever.invoke("databases and data stores")
    print(f"\nMMR results for 'databases and data stores':")
    for doc in mmr_docs:
        print(f"  → {doc.metadata['topic']}")


if __name__ == "__main__":
    demo_embeddings()
    faiss_store = demo_faiss()
    demo_chroma()
    demo_retriever(faiss_store)

    print("\n\nKey takeaways:")
    print("  - Embeddings convert text → vectors; similar text → similar vectors")
    print("  - FAISS: best for in-memory speed; Chroma: best for persistence + filtering")
    print("  - Always embed queries with the SAME model used to embed documents")
    print("  - as_retriever() converts any vector store into a Runnable for LCEL chains")
    print("  - MMR retrieval reduces result redundancy — good for diverse knowledge bases")
    print("  - Metadata filters let you scope search to subsets of your corpus")
    print(f"\n  Embedding model used: {EMBEDDING_MODEL}")
    print("  Pull it with: ollama pull nomic-embed-text")
