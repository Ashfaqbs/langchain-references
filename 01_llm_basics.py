"""
01 - LLM Basics
===============
The foundational building block in LangChain is the Language Model.
LangChain wraps every LLM/Chat model behind a common interface so you can
swap providers (Ollama, Groq, OpenAI, Anthropic) without changing your chain logic.

Two model types you'll use constantly:
  - LLM        : takes a plain string → returns a plain string   (older style)
  - ChatModel  : takes a list of messages → returns a message    (modern, preferred)

This file focuses on the raw model interface before we add prompts or chains.

WHEN YOU NEED THIS:
  Always. This is the entry point to calling any AI model.
  Think of it like java.net.HttpClient or Python's requests.Session — the raw
  connection layer. Nothing runs in isolation in production, but everything else
  (chains, agents, RAG) builds on top of this.

  Real scenario: call a local Ollama model or Groq cloud API and get a response.
  No templates, no parsing — just "send message, get reply."

WHY ChatModel OVER LLM:
  ChatModel (ChatOllama) supports system prompts, conversation history, tool
  calling, and structured output. Plain LLM (OllamaLLM) does not. Always use
  ChatModel for any real feature.

Run:  python 01_llm_basics.py
Prereq: Ollama running locally with at least one model pulled
        e.g.  ollama pull llama3.2
"""

from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ── Configuration ─────────────────────────────────────────────────────────────
# Change this to whichever model you have pulled in Ollama:
#   ollama list  →  shows installed models
MODEL = "deepseek-r1:latest"

# ─────────────────────────────────────────────────────────────────────────────
# 1. OllamaLLM  — plain text in, plain text out
# ─────────────────────────────────────────────────────────────────────────────
def demo_plain_llm():
    print("\n" + "=" * 60)
    print("1. OllamaLLM (plain string interface)")
    print("=" * 60)

    llm = OllamaLLM(
        model=MODEL,
        temperature=0.7,       # 0 = deterministic, 1 = creative
        num_predict=256,       # max tokens to generate
    )

    # .invoke() is the standard synchronous call
    response = llm.invoke("What is LangChain in one sentence?")
    print(f"Response type : {type(response)}")
    print(f"Response      : {response}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ChatOllama — message list in, AIMessage out
# ─────────────────────────────────────────────────────────────────────────────
def demo_chat_model():
    print("\n" + "=" * 60)
    print("2. ChatOllama (chat / message interface)")
    print("=" * 60)

    chat = ChatOllama(
        model=MODEL,
        temperature=0,        # 0 for consistent demo output
        num_predict=256,
    )

    # Messages are structured objects — not raw strings
    messages = [
        SystemMessage(content="You are a concise Python expert."),
        HumanMessage(content="What is a list comprehension? Give a 2-line example."),
    ]

    response = chat.invoke(messages)

    print(f"Response type    : {type(response)}")
    print(f"Response content : {response.content}")
    print(f"Metadata         : {response.response_metadata}")  # model, usage stats


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-turn conversation — building message history manually
# ─────────────────────────────────────────────────────────────────────────────
def demo_multi_turn():
    print("\n" + "=" * 60)
    print("3. Multi-turn conversation (manual history)")
    print("=" * 60)

    chat = ChatOllama(model=MODEL, temperature=0.3)

    # Start with a system instruction
    history = [
        SystemMessage(content="You are a math tutor. Keep answers short."),
    ]

    turns = [
        "What is the Pythagorean theorem?",
        "Give me a numerical example with sides 3 and 4.",
    ]

    for user_input in turns:
        history.append(HumanMessage(content=user_input))
        response = chat.invoke(history)
        history.append(response)  # AIMessage gets appended so context grows
        print(f"\nUser : {user_input}")
        print(f"AI   : {response.content}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model metadata & token usage
# ─────────────────────────────────────────────────────────────────────────────
def demo_metadata():
    print("\n" + "=" * 60)
    print("4. Inspecting model metadata and usage")
    print("=" * 60)

    chat = ChatOllama(model=MODEL)
    response = chat.invoke([HumanMessage(content="Say 'hello' and nothing else.")])

    print(f"Content          : {response.content}")
    print(f"Type             : {response.type}")          # 'ai'
    print(f"Response metadata: {response.response_metadata}")

    # usage_metadata is populated by most providers
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        print(f"Input tokens     : {response.usage_metadata.get('input_tokens')}")
        print(f"Output tokens    : {response.usage_metadata.get('output_tokens')}")


if __name__ == "__main__":
    demo_plain_llm()
    demo_chat_model()
    demo_multi_turn()
    demo_metadata()

    print("\n\nKey takeaways:")
    print("  - OllamaLLM / ChatOllama both expose .invoke(), .stream(), .batch()")
    print("  - Always prefer ChatModel (chat interface) for modern LangChain apps")
    print("  - Messages: SystemMessage, HumanMessage, AIMessage are the core types")
    print("  - The response from ChatModel is an AIMessage object, not a plain string")
