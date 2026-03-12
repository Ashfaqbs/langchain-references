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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INFERENCE PARAMETERS — THE FUNDAMENTALS FOR AI ENGINEERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW TOKEN GENERATION WORKS (THE MENTAL MODEL):
  At each step the model produces "logits" — one raw score per token in its
  vocabulary (~32k-128k tokens). These get turned into probabilities via
  softmax, then one token is *sampled* from that distribution. The parameters
  below all control how that sampling happens.

  Logits → temperature scaling → top-k filter → top-p filter → softmax → sample

──────────────────────────────────────────────────────────────
1. temperature
──────────────────────────────────────────────────────────────
  What it does : Scales the logits before softmax.
                 Low  → probability mass concentrates on the top token (focused).
                 High → probability spreads out (more surprising / creative).

  Min     : 0.0  (greedy — always picks the single highest-probability token,
                  fully deterministic given the same input + seed)
  Default : 0.8  (Ollama) / 1.0 (OpenAI) / 1.0 (Anthropic)
  Max     : 2.0  (anything above ~1.5 usually produces incoherent output)

  Output effect:
    temperature=0.0  → same answer every run, factual, robotic
    temperature=0.3  → stable but slightly varied phrasing
    temperature=0.7  → good balance for most tasks
    temperature=1.0  → noticeably creative / conversational
    temperature=1.5+ → chaotic, good for creative brainstorming only

  Rule of thumb:
    Facts / code / data extraction → 0.0–0.3
    Summarisation / Q&A            → 0.3–0.7
    Creative writing / chat        → 0.7–1.2

──────────────────────────────────────────────────────────────
2. num_predict  (= max_tokens / max_completion_tokens elsewhere)
──────────────────────────────────────────────────────────────
  Yes — num_predict IS the output/completion token limit.
  Different providers call it different things:
    Ollama      : num_predict
    OpenAI      : max_tokens  (legacy) / max_completion_tokens  (new)
    Anthropic   : max_tokens
    Groq        : max_tokens

  What it does : Hard cap on tokens the model will generate in one call.
                 Generation stops when this limit is hit OR when the model
                 outputs its end-of-sequence token (<eos>), whichever comes first.

  Min     : -2  (fill entire remaining context window — use carefully)
            -1  (no limit, generate until <eos>)
             1  (absolute minimum — one token)
  Default : -1  (Ollama default — unlimited until <eos>)
             128 / 256 in some older configs
  Max     : context_window − input_tokens  (you can't generate more tokens than
            the context window has room for after your prompt)

  Is there a param to limit INPUT tokens?
    No direct "max_input_tokens" param exists in inference APIs.
    You control input size by:
      a) Trimming / truncating your prompt before sending it.
      b) Setting num_ctx (Ollama) — the total context window size.
         If your prompt exceeds num_ctx the model silently truncates from
         the beginning (oldest tokens dropped first).
    num_ctx  min: 512  default: 2048 (Ollama, model-dependent)  max: model limit
    (e.g. llama3.2 supports 128k, deepseek-r1 supports 128k)

  Tokens vs words (rough guide):
    1 token ≈ 0.75 English words  ≈ 4 characters
    "Hello world" = 2 tokens
    A typical paragraph (~100 words) ≈ 130 tokens
    GPT-4 / Claude 3 / Llama 3 context windows: 8k – 200k tokens

──────────────────────────────────────────────────────────────
3. top_p  (nucleus sampling)
──────────────────────────────────────────────────────────────
  What it does : After temperature scaling, sort tokens by probability
                 descending and keep only the smallest set whose cumulative
                 probability ≥ top_p. Sample only from that set.
                 This removes the "long tail" of low-probability garbage tokens.

  Min     : 0.0  (only the single most-probable token — same as temperature=0)
  Default : 0.9  (Ollama) / 1.0 (OpenAI, effectively disabled) / 0.999 (Anthropic)
  Max     : 1.0  (all tokens considered — top_p disabled)

  Output effect:
    top_p=0.1  → very focused, only highest-probability tokens considered
    top_p=0.9  → cuts off the bottom 10% probability mass (recommended default)
    top_p=1.0  → no filtering, temperature alone controls diversity

  Interaction with temperature:
    They work in sequence. Tune ONE at a time — adjusting both simultaneously
    makes it hard to reason about behaviour.
    Common practice: fix top_p=0.9, tune temperature.

──────────────────────────────────────────────────────────────
4. top_k
──────────────────────────────────────────────────────────────
  What it does : Keep only the K most-probable next tokens. Everything outside
                 the top K is zeroed out before sampling.
                 Simpler than top_p — a hard count instead of a probability budget.

  Min     : 0  (disabled — all tokens eligible)
             1  (greedy, same as temperature=0)
  Default : 40  (Ollama)  /  0 (OpenAI — disabled)  /  not exposed (Anthropic)
  Max     : vocabulary size (~32k–128k), practically anything above 200 is
            equivalent to disabled

  Output effect:
    top_k=1   → always picks the most likely token (deterministic)
    top_k=10  → conservative, focused, less surprising
    top_k=40  → balanced (Ollama default, works well in most cases)
    top_k=0   → disabled, top_p does the filtering instead

  top_k vs top_p:
    top_k = fixed count of candidate tokens
    top_p = dynamic count based on cumulative probability
    top_p is generally preferred because it adapts: when the model is confident
    (few high-prob tokens) it samples from fewer candidates automatically.

──────────────────────────────────────────────────────────────
5. OTHER PARAMETERS YOU NEED TO KNOW
──────────────────────────────────────────────────────────────

  seed (int | None)
    Fix the random seed for reproducible output.
    Default: None (random each run).
    Use seed=42 (any fixed int) during testing / evals for determinism.
    Note: only fully deterministic when temperature=0 too.

  stop  (list[str])
    Generation halts when any of these strings appears in the output.
    Example: stop=["###", "\n\nUser:"]
    Essential for chat loops so the model doesn't roleplay both sides.

  repeat_penalty / frequency_penalty / presence_penalty
    Discourage repetition. The model tends to loop without this.
    Ollama   : repeat_penalty  (default 1.1, range 0.5–2.0; 1.0 = disabled)
    OpenAI   : frequency_penalty (default 0, range -2.0–2.0)
               presence_penalty  (default 0, range -2.0–2.0)
    > 1.0 penalises repeated tokens.  < 1.0 encourages repetition (rarely useful).

  num_ctx  (Ollama-specific context window)
    Sets the total token budget: input + output combined.
    Default: 2048 (model-dependent; increase for long documents / chat history).
    Higher num_ctx = more RAM / VRAM required.

  format  (Ollama-specific)
    format="json" forces the model to output valid JSON.
    Equivalent to OpenAI's response_format={"type": "json_object"}.

──────────────────────────────────────────────────────────────
6. QUICK REFERENCE TABLE
──────────────────────────────────────────────────────────────

  Parameter       | Min  | Default (Ollama) | Max      | Controls
  ────────────────┼──────┼──────────────────┼──────────┼──────────────────────────
  temperature     | 0.0  | 0.8              | 2.0      | randomness / creativity
  num_predict     | -2   | -1 (unlimited)   | ctx size | output token count
  num_ctx         | 512  | 2048+            | model max| total context window
  top_p           | 0.0  | 0.9              | 1.0      | nucleus sampling cutoff
  top_k           | 0    | 40               | vocab sz | candidate token count
  repeat_penalty  | 0.5  | 1.1              | 2.0      | repetition suppression
  seed            | 0    | None (random)    | any int  | reproducibility

──────────────────────────────────────────────────────────────
7. MENTAL MODEL FOR SETTING PARAMS IN PRODUCTION
──────────────────────────────────────────────────────────────

  Extraction / classification / code generation:
    temperature=0, top_p=0.9, top_k=40, seed=42 (for evals)

  Q&A / summarisation:
    temperature=0.3, top_p=0.9, top_k=40

  Chatbot / assistant:
    temperature=0.7, top_p=0.9, top_k=40

  Creative writing:
    temperature=1.0-1.2, top_p=0.95, top_k=0 (let top_p do the work)

  Always set num_predict explicitly in production to cap cost and latency.
  Always set stop sequences for chat-loop scenarios.
  Use seed in eval/testing pipelines for reproducible benchmarks.
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
