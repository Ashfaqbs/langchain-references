"""
06 - Document Loaders
======================
Before you can do RAG, you need to ingest documents.
LangChain's document loaders handle 50+ source types and normalize
everything into a list of Document objects.

Document object has:
  - page_content : str   — the actual text
  - metadata     : dict  — source info, page number, etc.

Common loaders:
  - TextLoader           : plain .txt files
  - PyPDFLoader          : PDF documents
  - WebBaseLoader        : scrape a web page
  - DirectoryLoader      : load all files in a directory
  - CSVLoader            : CSV rows as documents
  - JSONLoader           : JSON with jq-like field extraction
  - UnstructuredMarkdownLoader : Markdown files

WHEN YOU NEED THIS:
  Step 1 of any RAG pipeline. Before documents can be searched, they must be
  ingested into Python. Think of it like an ETL reader stage, a Kafka consumer,
  or a JDBC ResultSet — reads from a source and returns normalized objects.
  Here the normalized object is Document(page_content, metadata).

  Real scenarios:
  - Load all PDFs from a "company-policies/" folder → build internal Q&A bot
  - Load Confluence/Notion pages → searchable internal knowledge base
  - Load a product catalog CSV → answer "do you have X in stock?" questions
  - Pull records from PostgreSQL → wrap as Document objects manually
    (use demo_manual_documents() for this pattern)

WHY METADATA MATTERS:
  Every Document carries metadata (source file, page number, author).
  This is what enables showing "Source: kafka-guide.pdf, page 3" alongside
  an answer. Without metadata, there's no way to cite or verify the source.

LAZY LOADING (demo_lazy_loading):
  Use .lazy_load() instead of .load() when ingesting thousands of documents.
  .load() pulls everything into RAM at once. .lazy_load() streams one doc at
  a time — like a database cursor vs loading an entire table into memory.

Run:  python 06_document_loaders.py
Note: Some loaders need extra packages — see requirements.txt
"""

import os
import json
import tempfile
from pathlib import Path
from langchain_core.documents import Document

# ─────────────────────────────────────────────────────────────────────────────
# Helper: create sample files for demos
# ─────────────────────────────────────────────────────────────────────────────
def create_sample_files(tmp_dir: str):
    # Text file
    Path(f"{tmp_dir}/kafka-overview.txt").write_text("""
Apache Kafka is a distributed event streaming platform.
Originally developed at LinkedIn, it was open-sourced in 2011.
Kafka is used for building real-time data pipelines and streaming applications.
Key concepts: topics, partitions, producers, consumers, consumer groups.
Kafka guarantees at-least-once delivery by default.
With transactions enabled, it supports exactly-once semantics.
""")

    # CSV file
    Path(f"{tmp_dir}/engineers.csv").write_text(
        "name,role,language,experience\n"
        "Alice,Backend Engineer,Java,5\n"
        "Bob,ML Engineer,Python,3\n"
        "Carol,Frontend Engineer,TypeScript,4\n"
        "David,DevOps Engineer,Go,6\n"
    )

    # JSON file
    data = [
        {"id": 1, "title": "Intro to Kafka", "author": "Alice", "tags": ["streaming", "kafka"]},
        {"id": 2, "title": "Spring Boot REST", "author": "Bob",  "tags": ["java", "spring"]},
        {"id": 3, "title": "Python FastAPI",   "author": "Carol","tags": ["python", "api"]},
    ]
    Path(f"{tmp_dir}/articles.json").write_text(json.dumps(data, indent=2))

    # Markdown file
    Path(f"{tmp_dir}/readme.md").write_text("""
# LangChain Reference Project

## Overview
This project demonstrates LangChain from basics to advanced.

## Stack
- Python 3.12
- LangChain 1.1.3
- Ollama (local LLMs)
- Groq (cloud inference)

## Topics Covered
1. LLM basics
2. Prompt templates
3. LCEL chains
4. RAG pipeline
5. Agents and tools
""")

    return tmp_dir


# ─────────────────────────────────────────────────────────────────────────────
# 1. TextLoader — plain text files
# ─────────────────────────────────────────────────────────────────────────────
def demo_text_loader(tmp_dir: str):
    print("\n" + "=" * 60)
    print("1. TextLoader — plain .txt files")
    print("=" * 60)

    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(f"{tmp_dir}/kafka-overview.txt", encoding="utf-8")
    docs = loader.load()

    print(f"Number of docs : {len(docs)}")
    print(f"Content length : {len(docs[0].page_content)} chars")
    print(f"Metadata       : {docs[0].metadata}")
    print(f"Preview        : {docs[0].page_content[:150].strip()}...")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CSVLoader — one Document per row
# ─────────────────────────────────────────────────────────────────────────────
def demo_csv_loader(tmp_dir: str):
    print("\n" + "=" * 60)
    print("2. CSVLoader — rows as documents")
    print("=" * 60)

    from langchain_community.document_loaders.csv_loader import CSVLoader

    loader = CSVLoader(
        file_path=f"{tmp_dir}/engineers.csv",
        csv_args={"delimiter": ","},
        source_column="name",   # used as source in metadata
    )
    docs = loader.load()

    print(f"Number of docs: {len(docs)}")
    for doc in docs:
        print(f"  Source: {doc.metadata.get('source')} | Content: {doc.page_content[:60]}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. JSONLoader — extract specific fields from JSON
# ─────────────────────────────────────────────────────────────────────────────
def demo_json_loader(tmp_dir: str):
    print("\n" + "=" * 60)
    print("3. JSONLoader — JSON with field extraction")
    print("=" * 60)

    from langchain_community.document_loaders import JSONLoader

    loader = JSONLoader(
        file_path=f"{tmp_dir}/articles.json",
        jq_schema=".[].title",   # jq-style: extract .title from each array item
        text_content=True,
    )
    docs = loader.load()

    print(f"Number of docs: {len(docs)}")
    for doc in docs:
        print(f"  Title: {doc.page_content} | Meta: {doc.metadata}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. DirectoryLoader — load all files matching a pattern
# ─────────────────────────────────────────────────────────────────────────────
def demo_directory_loader(tmp_dir: str):
    print("\n" + "=" * 60)
    print("4. DirectoryLoader — load all .txt files in a directory")
    print("=" * 60)

    from langchain_community.document_loaders import DirectoryLoader

    loader = DirectoryLoader(
        tmp_dir,
        glob="**/*.txt",              # only .txt files
        loader_cls=TextLoader,        # use TextLoader for each file
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
        use_multithreading=True,      # parallel loading
    )
    docs = loader.load()

    print(f"Documents loaded: {len(docs)}")
    for doc in docs:
        print(f"  Source: {Path(doc.metadata['source']).name} | {len(doc.page_content)} chars")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Creating Documents manually — useful for custom pipelines
# ─────────────────────────────────────────────────────────────────────────────
def demo_manual_documents():
    print("\n" + "=" * 60)
    print("5. Creating Documents manually")
    print("=" * 60)

    # Sometimes you pull data from an API, DB, or custom source
    # Just wrap it in Document objects and the rest of the pipeline works the same
    docs = [
        Document(
            page_content="Redis is an in-memory data structure store used as cache, DB, and message broker.",
            metadata={"source": "internal-wiki", "topic": "redis", "page": 1, "author": "team"},
        ),
        Document(
            page_content="Spring Boot simplifies Java microservice development with auto-configuration.",
            metadata={"source": "internal-wiki", "topic": "spring-boot", "page": 1, "author": "team"},
        ),
        Document(
            page_content="PostgreSQL is a powerful open-source relational database with JSONB support.",
            metadata={"source": "internal-wiki", "topic": "postgresql", "page": 2, "author": "team"},
        ),
    ]

    print(f"Documents created: {len(docs)}")
    for doc in docs:
        print(f"  [{doc.metadata['topic']}]: {doc.page_content[:60]}...")
        print(f"   Metadata: {doc.metadata}")

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# 6. Lazy loading — memory-efficient for large document sets
# ─────────────────────────────────────────────────────────────────────────────
def demo_lazy_loading(tmp_dir: str):
    print("\n" + "=" * 60)
    print("6. Lazy loading — memory-efficient for large corpora")
    print("=" * 60)

    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    loader = DirectoryLoader(
        tmp_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    # .lazy_load() returns a generator — loads one doc at a time
    # Use this instead of .load() for large directories
    doc_count = 0
    for doc in loader.lazy_load():
        doc_count += 1
        print(f"  Lazily loaded: {Path(doc.metadata['source']).name}")

    print(f"Total lazy-loaded: {doc_count} documents")


if __name__ == "__main__":
    # Create temp directory with sample files
    with tempfile.TemporaryDirectory() as tmp_dir:
        from langchain_community.document_loaders import TextLoader
        create_sample_files(tmp_dir)

        demo_text_loader(tmp_dir)
        demo_csv_loader(tmp_dir)
        demo_json_loader(tmp_dir)
        demo_directory_loader(tmp_dir)
        demo_manual_documents()
        demo_lazy_loading(tmp_dir)

    print("\n\nKey takeaways:")
    print("  - All loaders return List[Document] with page_content + metadata")
    print("  - metadata tracks source, page, author — critical for RAG citations")
    print("  - DirectoryLoader + glob pattern loads entire knowledge bases")
    print("  - Use .lazy_load() for large corpora to avoid memory blowout")
    print("  - You can always create Document objects manually from any data source")
    print("  - WebBaseLoader, PyPDFLoader, NotionLoader, etc. follow the same interface")
