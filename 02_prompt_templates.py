"""
02 - Prompt Templates
=====================
Raw strings are fragile — you'd concatenate inputs, forget variables, break format.
LangChain's PromptTemplate solves this with typed, reusable, composable prompts.

Template types:
  - PromptTemplate           : plain text with {variable} placeholders
  - ChatPromptTemplate       : list of role-tagged messages with placeholders
  - MessagesPlaceholder      : slot for injecting a dynamic list of messages (used for memory)
  - FewShotChatMessagePromptTemplate : inject examples automatically

Run:  python 02_prompt_templates.py
"""

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

MODEL = "qwen3:0.6b"

# ─────────────────────────────────────────────────────────────────────────────
# 1. PromptTemplate — simplest form, single string
# ─────────────────────────────────────────────────────────────────────────────
def demo_prompt_template():
    print("\n" + "=" * 60)
    print("1. PromptTemplate (plain string)")
    print("=" * 60)

    template = PromptTemplate.from_template(
        "Explain {concept} to a {audience} in {sentences} sentences."
    )

    # .format() returns a plain string — no LLM call yet
    formatted = template.format(
        concept="recursion",
        audience="10-year-old",
        sentences="3",
    )
    print("Formatted prompt:", formatted)

    # .invoke() returns a StringPromptValue (wraps the string)
    prompt_value = template.invoke({
        "concept": "APIs",
        "audience": "senior engineer",
        "sentences": "2",
    })
    print("Prompt value type:", type(prompt_value))
    print("Prompt text:", prompt_value.text)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ChatPromptTemplate — the workhorse for chat models
# ─────────────────────────────────────────────────────────────────────────────
def demo_chat_prompt_template():
    print("\n" + "=" * 60)
    print("2. ChatPromptTemplate (for chat models)")
    print("=" * 60)

    # from_messages() accepts tuples of (role, template_string)
    # Roles: "system", "human", "ai"
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in {domain}. Respond in {language}."),
        ("human", "{question}"),
    ])

    # .invoke() returns a ChatPromptValue containing a list of messages
    messages = chat_prompt.invoke({
        "domain": "distributed systems",
        "language": "English",
        "question": "What is eventual consistency?",
    })

    print("Messages type:", type(messages))
    for msg in messages.messages:
        print(f"  [{msg.type}]: {msg.content[:80]}...")

    # Wire it directly to a chat model — this is the heart of LCEL
    llm = ChatOllama(model=MODEL, temperature=0)
    chain = chat_prompt | llm
    response = chain.invoke({
        "domain": "Python",
        "language": "English",
        "question": "What is a generator?",
    })
    print("\nLLM response:", response.content)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MessagesPlaceholder — inject dynamic message lists (critical for memory)
# ─────────────────────────────────────────────────────────────────────────────
def demo_messages_placeholder():
    print("\n" + "=" * 60)
    print("3. MessagesPlaceholder (dynamic history injection)")
    print("=" * 60)

    # MessagesPlaceholder reserves a slot where you inject a list of messages.
    # This is how conversation memory gets wired into prompts.
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),  # <-- dynamic slot
        ("human", "{input}"),
    ])

    # Simulate a prior conversation history
    history = [
        HumanMessage(content="My name is Ashfa."),
        AIMessage(content="Nice to meet you, Ashfa!"),
    ]

    messages = chat_prompt.invoke({
        "chat_history": history,
        "input": "What is my name?",
    })

    print("Total messages in prompt:", len(messages.messages))
    for msg in messages.messages:
        print(f"  [{msg.type}]: {msg.content}")

    llm = ChatOllama(model=MODEL, temperature=0)
    response = (chat_prompt | llm).invoke({
        "chat_history": history,
        "input": "What is my name?",
    })
    print("\nLLM response:", response.content)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FewShotChatMessagePromptTemplate — teach with examples
# ─────────────────────────────────────────────────────────────────────────────
def demo_few_shot():
    print("\n" + "=" * 60)
    print("4. FewShotChatMessagePromptTemplate (in-context examples)")
    print("=" * 60)

    # Define the shape of each example
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    # Your actual examples — teach the model the pattern
    examples = [
        {"input": "2 + 2",   "output": "4"},
        {"input": "10 / 2",  "output": "5"},
        {"input": "3 * 7",   "output": "21"},
    ]

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    # Wrap it in a full ChatPromptTemplate
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a calculator. Only answer with the numeric result."),
        few_shot_prompt,        # examples get expanded here
        ("human", "{question}"),
    ])

    messages = final_prompt.invoke({"question": "15 - 8"})
    print("Prompt messages injected:")
    for msg in messages.messages:
        print(f"  [{msg.type}]: {msg.content}")

    llm = ChatOllama(model=MODEL, temperature=0)
    response = (final_prompt | llm).invoke({"question": "15 - 8"})
    print("\nModel answer:", response.content)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Partial templates — pre-fill some variables
# ─────────────────────────────────────────────────────────────────────────────
def demo_partial_templates():
    print("\n" + "=" * 60)
    print("5. Partial templates (pre-fill variables)")
    print("=" * 60)

    template = ChatPromptTemplate.from_messages([
        ("system", "You are a {role} assistant."),
        ("human", "{question}"),
    ])

    # Lock in the role, leave question open — returns a new template
    support_template = template.partial(role="customer support")

    messages = support_template.invoke({"question": "How do I reset my password?"})
    print("System message:", messages.messages[0].content)
    print("Human message :", messages.messages[1].content)


if __name__ == "__main__":
    demo_prompt_template()
    demo_chat_prompt_template()
    demo_messages_placeholder()
    demo_few_shot()
    demo_partial_templates()

    print("\n\nKey takeaways:")
    print("  - PromptTemplate is for plain LLMs; ChatPromptTemplate is for chat models")
    print("  - MessagesPlaceholder is essential for injecting conversation memory")
    print("  - FewShotChatMessagePromptTemplate adds in-context examples cleanly")
    print("  - .partial() lets you pre-bake variables into a reusable template")
    print("  - Templates are Runnables — they support | (pipe) for chaining")
