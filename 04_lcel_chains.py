"""
04 - LCEL: LangChain Expression Language
=========================================
LCEL is the core composition system in modern LangChain (v0.3+).
Every component (prompt, model, parser, retriever, tool) is a "Runnable".
You compose Runnables with the | (pipe) operator — same mental model as Unix pipes.

LCEL gives you for free:
  - Streaming support on any chain
  - Async support (.ainvoke, .astream, .abatch)
  - Parallel execution
  - Automatic retry and fallback
  - LangSmith tracing

Key Runnables:
  - RunnablePassthrough     : passes input through unchanged (useful as a splitter)
  - RunnableParallel        : runs multiple chains on the same input simultaneously
  - RunnableLambda          : wraps any Python function as a Runnable
  - RunnableBranch          : conditional routing based on input

WHEN YOU NEED THIS:
  Always — this is how all LangChain components are wired together.
  Think of it like Java Streams (.filter().map().collect()), Python's functools
  pipeline, or Unix pipes. Each step takes the output of the previous.

  Real scenarios by component:

  RunnableParallel:
    Generate pros AND cons of a technology simultaneously — two LLM calls run
    in parallel, halving response time vs sequential.

  RunnableBranch:
    Route a query to a "technical expert" chain or a "simple explanation" chain
    based on detected expertise level. Same idea as a switch/match dispatch.

  RunnableLambda:
    Inject a mid-chain DB call — fetch account status from PostgreSQL and pass
    it into the prompt as context. Any plain Python function becomes a chain step.

  Chain-of-chains:
    Step 1 LLM generates an outline → Step 2 LLM expands it into a full doc.
    Output of one chain becomes input of the next — multi-step LLM pipelines.

Run:  python 04_lcel_chains.py
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
    RunnableBranch,
)
from langchain_ollama import ChatOllama

MODEL = "qwen3:0.6b"
llm = ChatOllama(model=MODEL, temperature=0.3)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic pipe chain — the foundation
# ─────────────────────────────────────────────────────────────────────────────
def demo_basic_chain():
    print("\n" + "=" * 60)
    print("1. Basic pipe chain (prompt | llm | parser)")
    print("=" * 60)

    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a concise assistant."),
            ("human", "What is {topic}?"),
        ])
        | llm
        | StrOutputParser()
    )

    # .invoke() — single call
    result = chain.invoke({"topic": "event sourcing"})
    print("Result:", result)

    # .batch() — parallel calls over a list of inputs
    print("\n.batch() with multiple inputs:")
    results = chain.batch([
        {"topic": "CQRS"},
        {"topic": "Saga pattern"},
    ])
    for i, r in enumerate(results):
        print(f"  [{i}]: {r[:80]}...")


# ─────────────────────────────────────────────────────────────────────────────
# 2. RunnablePassthrough — pass input through, optionally merge keys
# ─────────────────────────────────────────────────────────────────────────────
def demo_passthrough():
    print("\n" + "=" * 60)
    print("2. RunnablePassthrough — forwarding input unchanged")
    print("=" * 60)

    # Classic RAG pattern: you need both the retrieved context AND the question.
    # RunnablePassthrough lets the question flow through while context is computed.
    def fake_retriever(question: str) -> str:
        return f"[Retrieved docs about: {question}]"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use this context to answer:\n\n{context}"),
        ("human", "{question}"),
    ])

    # Dict on the left side of | creates a RunnableParallel implicitly:
    #   "question" gets the raw input passed through
    #   "context"  gets the result of fake_retriever(input)
    chain = (
        {
            "question": RunnablePassthrough(),
            "context": RunnableLambda(fake_retriever),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke("What is Kafka?")
    print("Result:", result[:200])


# ─────────────────────────────────────────────────────────────────────────────
# 3. RunnableParallel — run multiple branches simultaneously
# ─────────────────────────────────────────────────────────────────────────────
def demo_parallel():
    print("\n" + "=" * 60)
    print("3. RunnableParallel — multiple chains on same input")
    print("=" * 60)

    # Both branches get the SAME input and run concurrently
    pros_chain = (
        ChatPromptTemplate.from_messages([
            ("human", "List 3 pros of {technology} in one line each.")
        ])
        | llm
        | StrOutputParser()
    )

    cons_chain = (
        ChatPromptTemplate.from_messages([
            ("human", "List 3 cons of {technology} in one line each.")
        ])
        | llm
        | StrOutputParser()
    )

    parallel = RunnableParallel(
        pros=pros_chain,
        cons=cons_chain,
    )

    result = parallel.invoke({"technology": "microservices"})
    print("PROS:\n", result["pros"])
    print("\nCONS:\n", result["cons"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. RunnableLambda — wrap any Python function
# ─────────────────────────────────────────────────────────────────────────────
def demo_lambda():
    print("\n" + "=" * 60)
    print("4. RunnableLambda — inline Python functions")
    print("=" * 60)

    def uppercase_input(text: str) -> str:
        return text.upper()

    def count_words(text: str) -> dict:
        return {"text": text, "word_count": len(text.split())}

    chain = (
        RunnableLambda(uppercase_input)
        | ChatPromptTemplate.from_messages([
            ("human", "Summarize this topic in one sentence: {text}")
        ])
        | llm
        | StrOutputParser()
        | RunnableLambda(count_words)
    )

    result = chain.invoke("event driven architecture")
    print(f"Result : {result['text'][:100]}...")
    print(f"Words  : {result['word_count']}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. RunnableBranch — conditional routing
# ─────────────────────────────────────────────────────────────────────────────
def demo_branch():
    print("\n" + "=" * 60)
    print("5. RunnableBranch — conditional chain routing")
    print("=" * 60)

    # Route the query to a different chain based on content
    technical_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a senior engineer. Be technical and precise."),
            ("human", "{query}"),
        ])
        | llm
        | StrOutputParser()
    )

    simple_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Explain like I'm 5 years old."),
            ("human", "{query}"),
        ])
        | llm
        | StrOutputParser()
    )

    # RunnableBranch: list of (condition_fn, chain) pairs + default chain
    branch = RunnableBranch(
        (lambda x: "advanced" in x["query"].lower(), technical_chain),
        (lambda x: "simple" in x["query"].lower(),   simple_chain),
        technical_chain,  # default fallback
    )

    print("Technical query:")
    r1 = branch.invoke({"query": "advanced: how does Kafka replication work?"})
    print(r1[:200])

    print("\nSimple query:")
    r2 = branch.invoke({"query": "simple: what is a database?"})
    print(r2[:200])


# ─────────────────────────────────────────────────────────────────────────────
# 6. Chaining chains — composing sub-chains
# ─────────────────────────────────────────────────────────────────────────────
def demo_chain_of_chains():
    print("\n" + "=" * 60)
    print("6. Chain-of-thought: chaining two LLM calls")
    print("=" * 60)

    # Step 1: generate a topic outline
    outline_chain = (
        ChatPromptTemplate.from_messages([
            ("human", "Create a 3-point outline for a blog post about {topic}.")
        ])
        | llm
        | StrOutputParser()
    )

    # Step 2: expand that outline into a post
    expand_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a technical writer."),
            ("human", "Expand this outline into a short blog post:\n\n{outline}"),
        ])
        | llm
        | StrOutputParser()
    )

    # Compose: output of step 1 becomes "outline" input of step 2
    full_chain = (
        outline_chain
        | RunnableLambda(lambda outline: {"outline": outline})
        | expand_chain
    )

    result = full_chain.invoke({"topic": "Redis caching strategies"})
    print(result[:400], "...")


if __name__ == "__main__":
    demo_basic_chain()
    demo_passthrough()
    demo_parallel()
    demo_lambda()
    demo_branch()
    demo_chain_of_chains()

    print("\n\nKey takeaways:")
    print("  - Every LangChain component is a Runnable: .invoke() .stream() .batch()")
    print("  - | (pipe) is the LCEL composition operator — left output → right input")
    print("  - A dict on the left of | auto-creates a RunnableParallel")
    print("  - RunnablePassthrough forwards input; use with dicts for multi-input prompts")
    print("  - RunnableBranch enables conditional routing without if/else boilerplate")
    print("  - Any chain can be used as a step inside another chain")
