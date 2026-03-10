"""
10 - Tools & Agents
=====================
An Agent is an LLM that can DECIDE which tools to use and in what order.
Instead of a fixed chain, agents dynamically choose their steps.

Core concepts:
  - Tool     : a function the agent can call (with name, description, input schema)
  - Agent    : the LLM + reasoning logic (decides what to do next)
  - AgentExecutor: runtime loop that runs the agent until it produces a final answer

ReAct (Reason + Act) is the most common agent pattern:
  1. Reason: "I need to find the weather, I'll use the get_weather tool"
  2. Act   : calls get_weather("London")
  3. Observe: gets the result
  4. Repeat until final answer

In LangChain v1, the modern approach uses LangGraph for agents.
Here we cover both the classic AgentExecutor AND the modern approach.

WHEN YOU NEED THIS:
  When an answer requires real-time data, computation, or actions an LLM cannot
  perform on its own. Think of it like a command dispatcher or strategy pattern —
  the model reads intent and selects which function to invoke, rather than that
  decision being hardcoded in application logic.

  Real scenarios:
  - "What is my account balance?" → agent calls get_account_balance(user_id)
  - "What's 15% of 230,000?" → agent calls a calculator tool (LLMs are unreliable at math)
  - "Create a support ticket for this bug" → agent calls create_ticket() API
  - "Look up order #12345" → agent calls database_query(order_id=12345)
  - "Send a Slack message to the on-call team" → agent calls send_slack()

CHAINS vs AGENTS — key difference:
  Chain → fixed sequence of steps, the flow is decided at build time
  Agent → dynamic, the LLM decides which step to take next based on context
  Use chains when the flow is predictable and fixed.
  Use agents when the required steps depend on what the user asks.

THE AGENT LOOP (what demo_manual_agent_loop shows):
  1. LLM sees user message + list of available tools
  2. LLM decides: call tool X with args Y
  3. Tool X runs, produces a result
  4. Result is fed back to the LLM as a ToolMessage
  5. LLM decides: call another tool, or produce the final answer
  This repeats until no more tool calls → final answer returned.

LANGGRAPH AGENT (demo_langgraph_agent) — use this in production:
  create_react_agent() from langgraph manages the loop automatically. With
  MemorySaver, agent state persists across conversation turns.

Run:  python 10_tools_and_agents.py
"""

import math
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

MODEL = "llama3.2"  # Needs to support tool calling / function calling

# ─────────────────────────────────────────────────────────────────────────────
# 1. Defining tools with @tool decorator
# ─────────────────────────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Input must be a valid Python math expression string.
    Examples: '2 + 2', '10 * 5', 'math.sqrt(144)', '2 ** 10'
    """
    try:
        # Safe eval with math module available
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def word_counter(text: str) -> str:
    """
    Count the number of words in the given text.
    Returns word count as a string.
    """
    count = len(text.split())
    return f"The text contains {count} words."


@tool
def string_reverser(text: str) -> str:
    """
    Reverse the given string and return it.
    """
    return text[::-1]


@tool
def temperature_converter(celsius: float) -> str:
    """
    Convert a temperature from Celsius to Fahrenheit and Kelvin.
    Input: temperature in Celsius as a float.
    Returns: conversion results as a string.
    """
    fahrenheit = (celsius * 9 / 5) + 32
    kelvin = celsius + 273.15
    return f"{celsius}°C = {fahrenheit}°F = {kelvin}K"


@tool
def list_python_keywords(category: str) -> str:
    """
    Return Python keywords or built-in functions for a given category.
    Valid categories: 'control_flow', 'exceptions', 'functions', 'classes'
    """
    keywords = {
        "control_flow": ["if", "elif", "else", "for", "while", "break", "continue", "pass", "return"],
        "exceptions":   ["try", "except", "finally", "raise", "assert"],
        "functions":    ["def", "lambda", "yield", "async", "await"],
        "classes":      ["class", "self", "super", "property", "staticmethod", "classmethod"],
    }
    result = keywords.get(category.lower())
    if result is None:
        return f"Unknown category '{category}'. Valid: {list(keywords.keys())}"
    return f"Python {category} keywords: {', '.join(result)}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Inspect tool metadata — what the agent sees
# ─────────────────────────────────────────────────────────────────────────────
def demo_tool_metadata():
    print("\n" + "=" * 60)
    print("1. Tool metadata — what the agent sees")
    print("=" * 60)

    tools = [calculator, word_counter, string_reverser, temperature_converter]

    for t in tools:
        print(f"\nTool   : {t.name}")
        print(f"Desc   : {t.description}")
        print(f"Schema : {t.args}")

    # Direct tool invocation (without agent)
    print("\n\nDirect tool calls:")
    print(f"  calculator('2 ** 10')     = {calculator.invoke({'expression': '2 ** 10'})}")
    print(f"  temperature_converter(100)= {temperature_converter.invoke({'celsius': 100})}")
    print(f"  word_counter('hello world')= {word_counter.invoke({'text': 'hello world'})}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bind tools to LLM — LLM decides WHEN to call which tool
# ─────────────────────────────────────────────────────────────────────────────
def demo_tool_binding():
    print("\n" + "=" * 60)
    print("2. Binding tools to LLM — tool-calling interface")
    print("=" * 60)

    tools = [calculator, temperature_converter, word_counter]
    llm = ChatOllama(model=MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # The LLM now knows about the tools — it can decide to call them
    from langchain_core.messages import HumanMessage

    response = llm_with_tools.invoke([
        HumanMessage(content="What is 25 multiplied by 16?")
    ])

    print(f"Response type: {type(response)}")
    print(f"Content      : {response.content}")

    if response.tool_calls:
        print(f"Tool calls   : {response.tool_calls}")
    else:
        print("No tool calls — model answered directly")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Manual agent loop — understand what's happening under the hood
# ─────────────────────────────────────────────────────────────────────────────
def demo_manual_agent_loop():
    print("\n" + "=" * 60)
    print("3. Manual agent loop (ReAct pattern, unrolled)")
    print("=" * 60)

    from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

    tools = [calculator, temperature_converter, word_counter, list_python_keywords]
    tool_map = {t.name: t for t in tools}

    llm = ChatOllama(model=MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        HumanMessage(content=(
            "What is 2 to the power of 8? "
            "Also convert 37 degrees Celsius to Fahrenheit."
        ))
    ]

    print(f"User: {messages[0].content}")

    # Agent loop — runs until no more tool calls
    max_iterations = 5
    for iteration in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"\nAgent final answer: {response.content}")
            break

        print(f"\n[Iteration {iteration + 1}] Agent wants to call {len(response.tool_calls)} tool(s):")

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id   = tool_call["id"]

            print(f"  → {tool_name}({tool_args})")

            # Execute the tool
            tool_fn = tool_map[tool_name]
            tool_result = tool_fn.invoke(tool_args)

            print(f"  ← Result: {tool_result}")

            # Feed result back as a ToolMessage
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_id,
            ))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Using create_react_agent from langgraph (modern approach)
# ─────────────────────────────────────────────────────────────────────────────
def demo_langgraph_agent():
    print("\n" + "=" * 60)
    print("4. LangGraph ReAct agent (modern recommended approach)")
    print("=" * 60)

    try:
        from langgraph.prebuilt import create_react_agent

        tools = [calculator, temperature_converter, word_counter, string_reverser]
        llm = ChatOllama(model=MODEL, temperature=0)

        # create_react_agent returns a compiled graph (stateful agent)
        agent = create_react_agent(
            llm,
            tools,
            state_modifier="You are a helpful assistant with access to tools. "
                           "Always use tools when you need to compute something.",
        )

        # .invoke() runs the full agent loop and returns the final state
        result = agent.invoke({
            "messages": [("human", "What is the square root of 256? And reverse the word 'LangChain'.")]
        })

        # The last message in the state is the agent's final answer
        final_message = result["messages"][-1]
        print(f"Agent answer: {final_message.content}")

        # Show all messages in the trace
        print(f"\nFull trace ({len(result['messages'])} messages):")
        for msg in result["messages"]:
            msg_type = type(msg).__name__
            content  = str(msg.content)[:80] if msg.content else "[tool calls]"
            print(f"  [{msg_type:15}]: {content}")

    except ImportError:
        print("  langgraph not installed. Install with: pip install langgraph")
        print("  Falling back to manual loop (see demo_manual_agent_loop)")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Agent with memory — stateful multi-turn
# ─────────────────────────────────────────────────────────────────────────────
def demo_agent_with_memory():
    print("\n" + "=" * 60)
    print("5. Agent with memory (multi-turn)")
    print("=" * 60)

    try:
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver

        tools = [calculator, temperature_converter]
        llm = ChatOllama(model=MODEL, temperature=0)

        # MemorySaver stores agent state between turns
        memory = MemorySaver()
        agent = create_react_agent(llm, tools, checkpointer=memory)

        config = {"configurable": {"thread_id": "session-1"}}

        turns = [
            "My favorite number is 42. What is 42 squared?",
            "Now multiply that by 2. And what was my favorite number?",
        ]

        for user_input in turns:
            result = agent.invoke(
                {"messages": [("human", user_input)]},
                config=config,
            )
            final = result["messages"][-1].content
            print(f"\nUser : {user_input}")
            print(f"Agent: {final}")

    except ImportError:
        print("  langgraph not installed. Install with: pip install langgraph")


if __name__ == "__main__":
    demo_tool_metadata()
    demo_tool_binding()
    demo_manual_agent_loop()
    demo_langgraph_agent()
    demo_agent_with_memory()

    print("\n\nKey takeaways:")
    print("  - @tool decorator creates a LangChain tool from any Python function")
    print("  - The docstring IS the tool description — write it clearly for the agent")
    print("  - llm.bind_tools(tools) registers tools so the LLM knows about them")
    print("  - The agent loop: LLM → tool call? → run tool → feed result back → repeat")
    print("  - create_react_agent (langgraph) is the modern, recommended approach")
    print("  - MemorySaver gives agents persistent state across conversation turns")
    print("  - Tool calling requires a model that supports function/tool calling")
