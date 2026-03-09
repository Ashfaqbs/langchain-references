"""
05 - Memory & Chat History
===========================
LLMs are stateless — every call is independent. Memory is the mechanism
for maintaining conversation context across turns.

LangChain's modern approach (v0.3+):
  - Store history externally in a ChatMessageHistory object
  - Inject it into the chain via RunnableWithMessageHistory
  - The chain fetches and updates history automatically per session

History stores:
  - InMemoryChatMessageHistory  : in-process dict (dev/testing)
  - FileChatMessageHistory      : JSON file on disk (simple persistence)
  - RedisChatMessageHistory     : Redis (production)
  - SQLChatMessageHistory       : any SQL DB via SQLAlchemy

WHEN YOU NEED THIS:
  Any multi-turn conversation where context from earlier turns matters.
  LLMs are stateless — like a REST API with no session. State must be stored
  externally and injected on every request, the same way a web app reads a
  session cookie from Redis or a DB on each request.

  Real scenarios:
  - Chatbot: user says "I use Kafka" on turn 1, asks "how to handle errors in
    that?" on turn 3 — without memory the model has no idea what "that" refers to
  - Interview bot: remembers earlier answers to ask meaningful follow-ups
  - Onboarding assistant: builds a user profile progressively across turns
  - Code review assistant: remembers earlier decisions within the same session

SESSION_ID is critical:
  In a multi-user app, each user gets a distinct session_id so histories are
  isolated. Same concept as a session token — one per user, scoped to their
  conversation only.

PRODUCTION UPGRADE PATH:
  Dev   → ChatMessageHistory (in-memory, lost on restart)
  Prod  → RedisChatMessageHistory (persistent, survives restarts, scalable)

TRIM HISTORY — always implement for long conversations:
  A long conversation accumulates messages until they exceed the model's context
  window and the call fails. trim_messages() keeps only the most recent N tokens,
  same idea as a circular buffer or a sliding window.

Run:  python 05_memory_chat_history.py
"""

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

MODEL = "llama3.2"
llm = ChatOllama(model=MODEL, temperature=0.3)

# ─────────────────────────────────────────────────────────────────────────────
# 1. ChatMessageHistory — the history store
# ─────────────────────────────────────────────────────────────────────────────
def demo_message_history_basics():
    print("\n" + "=" * 60)
    print("1. ChatMessageHistory — raw history manipulation")
    print("=" * 60)

    history = ChatMessageHistory()

    # Add messages manually
    history.add_user_message("Hi, my name is Ashfa.")
    history.add_ai_message("Hello Ashfa! How can I help you today?")
    history.add_user_message("I work with Spring Boot and Python.")
    history.add_ai_message("Great stack! Spring Boot for Java microservices and Python for AI?")

    print(f"Total messages: {len(history.messages)}")
    for msg in history.messages:
        prefix = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  [{prefix}]: {msg.content}")

    # Access messages
    print(f"\nLast AI message: {history.messages[-1].content}")

    # Clear history
    history.clear()
    print(f"After clear: {len(history.messages)} messages")


# ─────────────────────────────────────────────────────────────────────────────
# 2. RunnableWithMessageHistory — automatic history management
# ─────────────────────────────────────────────────────────────────────────────
def demo_runnable_with_history():
    print("\n" + "=" * 60)
    print("2. RunnableWithMessageHistory — automatic per-session memory")
    print("=" * 60)

    # In-memory store: maps session_id → ChatMessageHistory
    # In production, replace this with Redis or DB store
    session_store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]

    # Build the chain — note the MessagesPlaceholder for "chat_history"
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer concisely."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()

    # Wrap chain with history management
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",          # which key is the user message
        history_messages_key="chat_history", # which key maps to MessagesPlaceholder
    )

    config = {"configurable": {"session_id": "user-123"}}

    # Turn 1
    r1 = chain_with_history.invoke(
        {"input": "My name is Ashfa and I love Kafka."},
        config=config,
    )
    print("Turn 1:", r1)

    # Turn 2 — model should remember the context from turn 1
    r2 = chain_with_history.invoke(
        {"input": "What messaging system did I just mention?"},
        config=config,
    )
    print("Turn 2:", r2)

    # Turn 3
    r3 = chain_with_history.invoke(
        {"input": "And what is my name?"},
        config=config,
    )
    print("Turn 3:", r3)

    # Inspect stored history
    print(f"\nStored messages for session 'user-123': {len(session_store['user-123'].messages)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multiple sessions — history is scoped by session_id
# ─────────────────────────────────────────────────────────────────────────────
def demo_multiple_sessions():
    print("\n" + "=" * 60)
    print("3. Multiple sessions — isolated histories")
    print("=" * 60)

    session_store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    chain_with_history = RunnableWithMessageHistory(
        prompt | llm | StrOutputParser(),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # Session A
    chain_with_history.invoke(
        {"input": "I am building a RAG system with LangChain."},
        config={"configurable": {"session_id": "session-A"}},
    )

    # Session B — completely separate context
    chain_with_history.invoke(
        {"input": "I am working on a React frontend."},
        config={"configurable": {"session_id": "session-B"}},
    )

    # Ask both about what they said
    r_a = chain_with_history.invoke(
        {"input": "What am I building?"},
        config={"configurable": {"session_id": "session-A"}},
    )
    r_b = chain_with_history.invoke(
        {"input": "What am I working on?"},
        config={"configurable": {"session_id": "session-B"}},
    )

    print("Session A:", r_a)
    print("Session B:", r_b)
    print(f"\nSession A message count: {len(session_store['session-A'].messages)}")
    print(f"Session B message count: {len(session_store['session-B'].messages)}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Trimming history — avoid token limit overflow
# ─────────────────────────────────────────────────────────────────────────────
def demo_trim_history():
    print("\n" + "=" * 60)
    print("4. Trimming history — keeping context within token limits")
    print("=" * 60)

    from langchain_core.messages import trim_messages

    # Create a large history to demonstrate trimming
    history = ChatMessageHistory()
    for i in range(10):
        history.add_user_message(f"Turn {i}: What is Python feature {i}?")
        history.add_ai_message(f"Python feature {i} is about {'generators' if i % 2 == 0 else 'decorators'}.")

    print(f"Before trim: {len(history.messages)} messages")

    # trim_messages keeps the most recent N tokens
    trimmed = trim_messages(
        history.messages,
        max_tokens=200,           # keep last ~200 tokens
        strategy="last",          # keep from the end
        token_counter=llm,        # use the model's tokenizer
        include_system=True,      # never drop system message
        allow_partial=False,      # don't split a message in half
        start_on="human",         # always start with a human message
    )

    print(f"After trim : {len(trimmed)} messages")
    for msg in trimmed:
        prefix = "H" if isinstance(msg, HumanMessage) else "A"
        print(f"  [{prefix}]: {msg.content[:60]}")


if __name__ == "__main__":
    demo_message_history_basics()
    demo_runnable_with_history()
    demo_multiple_sessions()
    demo_trim_history()

    print("\n\nKey takeaways:")
    print("  - History is stored OUTSIDE the chain, not inside it")
    print("  - session_id scopes history — essential for multi-user apps")
    print("  - RunnableWithMessageHistory auto-loads and saves history around the chain")
    print("  - MessagesPlaceholder is the bridge between history store and prompt")
    print("  - Always implement history trimming for long conversations (token limits!)")
    print("  - In production: swap InMemoryChatMessageHistory for Redis/SQL store")
