"""
18 - Human in the Loop
======================
"Human in the loop" (HITL) means the AI workflow PAUSES and waits for a
real human to provide input, approval, or feedback before continuing.

Without HITL an agent runs autonomously start-to-finish.
With HITL you get control points where you can:
  - Steer the conversation (interactive chat)
  - Approve or reject AI output before it's used
  - Provide feedback so the AI can refine its answer

FOUR PATTERNS covered here:

  Pattern 1 — Interactive Chat REPL
    The simplest HITL: a loop where you type a message, the AI replies,
    and the full conversation history grows. Type 'quit' to exit.
    This is the foundation of every chatbot.

  Pattern 2 — Human Approval Gate
    AI generates something (e.g. a draft email), then PAUSES.
    Human types 'yes' to accept, 'no' to regenerate, or provides
    free-text feedback. The loop continues until the human accepts.
    Pattern used in: code review bots, content generation pipelines.

  Pattern 3 — Iterative Refinement
    AI produces output, human gives specific feedback ("make it shorter",
    "add an example"), AI improves it. Repeats until human is satisfied.
    Pattern used in: writing assistants, report generators, code explainers.

  Pattern 4 — Tool Call Permission Gate  *** THE MOST IMPORTANT ONE ***
    AI decides it needs to call a real-world tool (send email, schedule
    meeting, delete file). Before executing, it PAUSES and asks the human
    "can I run this?". Human sees the exact tool name + arguments.
    Yes → tool runs, result fed back to AI.
    No  → AI is told it was denied, responds gracefully.

    This is EXACTLY what Claude Code does when it asks:
      "Run bash command: rm -rf ./dist?" [yes/no]
    The AI proposes the action; a human gate decides if it executes.
    Pattern used in: agents, AI assistants with external integrations.

WHY THIS MATTERS IN PRODUCTION:
  Fully autonomous agents are great for well-defined tasks, but for anything
  touching real users, finances, or external systems you need control points.
  LangGraph (covered in later files) formalises this with interrupt() nodes.
  These demos teach the CONCEPT before you reach that abstraction.

Run:   python 18_human_in_the_loop.py
Model: qwen3:4b  (ollama pull qwen3:4b)
"""

import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

MODEL = "qwen3:4b"

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

# qwen3 is a "thinking" model — it wraps its internal reasoning in <think>...</think>
# before giving the actual answer. We strip those tags before displaying output.
# Other models (llama3, mistral, etc.) never emit these tags so this is a no-op for them.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that qwen3 emits before its real answer."""
    return _THINK_RE.sub("", text).strip()


def build_chat(temperature: float = 0.7, think: bool = True) -> ChatOllama:
    # think=False disables qwen3's chain-of-thought mode.
    # IMPORTANT for tool calling: thinking tokens and tool-call JSON cannot coexist
    # in the same response — if thinking is on, the model outputs <think> blocks
    # and then emits empty tool_calls, so nothing executes.
    return ChatOllama(
        model=MODEL,
        temperature=temperature,
        num_predict=512,
        top_p=0.9,
        top_k=40,
        num_ctx=4096,
        think=think,           # passed through to Ollama's API
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 1 — Interactive Chat REPL
# ─────────────────────────────────────────────────────────────────────────────
def demo_chat_repl():
    """
    Simplest HITL pattern: a read-eval-print loop.

    Concept:
      history = [SystemMessage, HumanMessage, AIMessage, HumanMessage, ...]
      Each turn:
        1. Human types → append HumanMessage to history
        2. Call model with full history → get AIMessage
        3. Append AIMessage to history
        4. Repeat

    The history list IS the "memory" here. No magic — just a growing list.
    """
    print("\n" + "=" * 60)
    print("PATTERN 1 — Interactive Chat REPL")
    print("=" * 60)
    print("Chat with qwen3:4b. Type 'quit' or 'exit' to stop.")
    print("-" * 60)

    chat = build_chat(temperature=0.7)

    # System message sets the AI's persona for the whole conversation.
    # It is always the FIRST message and never changes.
    history = [
        SystemMessage(content=(
            "You are a concise, helpful AI assistant. "
            "Keep answers under 3 sentences unless the user asks for more detail."
        ))
    ]

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Ending chat.")
            break

        # Append user turn to history
        history.append(HumanMessage(content=user_input))

        # Model sees the FULL history — that's how it knows context
        response = chat.invoke(history)

        # Append AI turn so next call includes this exchange
        history.append(response)

        print(f"\nAI: {strip_thinking(response.content)}")

        # Debug: show how many tokens are in play (most providers return this)
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            print(
                f"    [tokens — in: {meta.get('input_tokens', '?')} "
                f"out: {meta.get('output_tokens', '?')}]"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 2 — Human Approval Gate
# ─────────────────────────────────────────────────────────────────────────────
def demo_approval_gate():
    """
    AI generates output → human reviews → approve or reject.

    This is the pattern behind:
      - "Approve this PR description before posting"
      - "Review this SQL query before executing"
      - "Confirm this email draft before sending"

    Flow:
      generate → show to human → yes: continue | no/feedback: regenerate
    """
    print("\n" + "=" * 60)
    print("PATTERN 2 — Human Approval Gate")
    print("=" * 60)
    print("AI will draft a short product description.")
    print("You decide: accept it or give feedback to improve it.")
    print("-" * 60)

    chat = build_chat(temperature=0.9)  # slightly creative for drafting

    product = input("\nWhat product should the AI write about? > ").strip()
    if not product:
        product = "a smart water bottle that tracks hydration"

    # The task prompt — fixed, does not change across iterations
    task = (
        f"Write a 2-sentence marketing description for: {product}. "
        "Be punchy and benefit-focused. No bullet points."
    )

    messages = [HumanMessage(content=task)]
    iteration = 0

    while True:
        iteration += 1
        print(f"\n[Generating draft #{iteration}...]")

        response = chat.invoke(messages)
        draft = strip_thinking(response.content)
        print(f"\nDraft:\n  {draft}")

        # ----- HUMAN CONTROL POINT -----
        decision = input(
            "\nAccept? [yes / no / or type feedback to improve it]: "
        ).strip()

        if decision.lower() in ("yes", "y", ""):
            print("\nDraft accepted. In a real pipeline this would be sent/saved.")
            break

        # Human gave feedback — inject it so the AI can improve
        # We append the draft as an AIMessage + the feedback as HumanMessage
        messages.append(AIMessage(content=draft))
        messages.append(HumanMessage(content=(
            f"That wasn't quite right. Here's my feedback: {decision}. "
            f"Please rewrite the 2-sentence description for: {product}."
        )))

        # Safety: stop after 5 iterations to avoid infinite loops in demos
        if iteration >= 5:
            print("\n[Max iterations reached. Exiting approval loop.]")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 3 — Iterative Refinement
# ─────────────────────────────────────────────────────────────────────────────
def demo_iterative_refinement():
    """
    AI explains a concept → human gives targeted feedback → AI refines.

    Different from the approval gate:
      - Approval gate: binary yes/no with optional feedback
      - Iterative refinement: focused improvement loop, human steers quality

    Practical use cases:
      - Code explainer: "explain this function" → "now for a beginner" → "add analogy"
      - Report writer: "write summary" → "more concise" → "add numbers"
    """
    print("\n" + "=" * 60)
    print("PATTERN 3 — Iterative Refinement")
    print("=" * 60)
    print("AI will explain a technical concept.")
    print("Give feedback each round. Type 'done' when satisfied.")
    print("-" * 60)

    chat = build_chat(temperature=0.5)  # lower temp = more consistent rewrites

    topic = input("\nWhat topic should the AI explain? > ").strip()
    if not topic:
        topic = "how a database index works"

    # Seed message: the initial explanation request
    history = [
        SystemMessage(content=(
            "You are a patient technical educator. "
            "Adjust your explanation based on the human's feedback."
        )),
        HumanMessage(content=f"Explain {topic} in 3-4 sentences."),
    ]

    iteration = 0

    while True:
        iteration += 1
        print(f"\n[Generating explanation #{iteration}...]")

        response = chat.invoke(history)
        history.append(response)

        print(f"\nExplanation:\n{strip_thinking(response.content)}")

        # ----- HUMAN CONTROL POINT -----
        feedback = input(
            "\nFeedback (or 'done' to finish): "
        ).strip()

        if feedback.lower() in ("done", ""):
            print("\nRefinement complete.")
            break

        # Add human feedback as a new turn — model will see full history
        history.append(HumanMessage(content=feedback))

        if iteration >= 6:
            print("\n[Max refinement iterations reached.]")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 4 — Tool Call Permission Gate
# ─────────────────────────────────────────────────────────────────────────────

# --- Mock tools ---
# @tool turns a plain function into a LangChain Tool.
# The docstring becomes the tool description the AI reads to decide when to use it.
# The type hints become the argument schema the AI must fill in.

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient with a subject and body."""
    # In production this would call an SMTP client / SendGrid / SES etc.
    print(f"    [MOCK EXECUTED] send_email → to={to}, subject={subject}")
    return f"Email delivered to '{to}' | subject='{subject}' | body length={len(body)} chars"


@tool
def schedule_meeting(attendees: str, when: str, topic: str) -> str:
    """Schedule a calendar meeting with given attendees, time, and topic."""
    print(f"    [MOCK EXECUTED] schedule_meeting → topic={topic}, when={when}")
    return f"Meeting '{topic}' scheduled for {when} with {attendees}"


@tool
def delete_file(filename: str) -> str:
    """Permanently delete a file from the filesystem. This action is irreversible."""
    # Marked dangerous below — notice it gets an extra warning in the UI
    print(f"    [MOCK EXECUTED] delete_file → {filename}")
    return f"File '{filename}' permanently deleted"


# Registry: tool name → tool object so we can look up and call by name
TOOL_REGISTRY = {
    "send_email":       send_email,
    "schedule_meeting": schedule_meeting,
    "delete_file":      delete_file,
}

# Tools flagged as destructive get an extra warning prompt in the UI,
# the same way Claude Code shows a red warning before running rm commands.
DANGEROUS_TOOLS = {"delete_file"}


def demo_tool_permission_gate():
    """
    The AI is given tools it can call. When it decides to use one it emits
    an AIMessage with a .tool_calls list instead of plain text.

    Message flow when a tool call happens:
      HumanMessage("send an email to alice...")
        → AIMessage(tool_calls=[{name: "send_email", args: {...}, id: "abc"}])
        → [PAUSE — human sees tool name + args, says yes or no]
        → ToolMessage(content="<result or denial>", tool_call_id="abc")
        → AIMessage("I've sent the email...")   ← model summarises

    ToolMessage is mandatory — the model always expects a result for every
    tool call it made before it will generate a final text response.
    If you skip it, the API returns an error.

    Why this matters:
      This is the exact mechanism behind Claude Code's permission prompts,
      Copilot Workspace's action approvals, and every responsible AI agent.
      The AI proposes; the human decides; the result flows back.
    """
    print("\n" + "=" * 60)
    print("PATTERN 4 — Tool Call Permission Gate")
    print("=" * 60)
    print("The AI has 3 tools: send_email, schedule_meeting, delete_file")
    print("It will ask YOUR permission before running any of them.")
    print("Type 'quit' to exit.")
    print("-" * 60)
    print("Try asking:")
    print('  "send an email to alice@example.com about the Q3 report"')
    print('  "schedule a meeting with bob and carol tomorrow at 2pm to discuss the roadmap"')
    print('  "delete the file old_backup.zip"')
    print("-" * 60)

    # think=False is mandatory here.
    # qwen3's thinking tokens and tool-call JSON cannot coexist in one response:
    # with thinking on, the model outputs <think> blocks then emits empty tool_calls.
    # Disabling it forces the model to output structured tool-call JSON directly.
    chat = build_chat(temperature=0, think=False).bind_tools(list(TOOL_REGISTRY.values()))

    history = [
        SystemMessage(content=(
            "You are a helpful assistant with access to tools. "
            "Use the appropriate tool when the user asks you to send emails, "
            "schedule meetings, or manage files. "
            "After a tool result is returned, give a short confirmation to the user."
        ))
    ]

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Exiting tool demo.")
            break

        history.append(HumanMessage(content=user_input))
        response = chat.invoke(history)
        history.append(response)

        # ── Did the AI request one or more tool calls? ──────────────────────
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name    = tool_call["name"]
                tool_args    = tool_call["args"]
                tool_call_id = tool_call["id"]

                # ── HUMAN CONTROL POINT ─────────────────────────────────────
                print(f"\n  [AI wants to use a tool]")
                print(f"  Tool : {tool_name}")
                print(f"  Args : {tool_args}")

                # Extra warning for irreversible/dangerous tools —
                # same UX as Claude Code's red warning before destructive bash commands
                if tool_name in DANGEROUS_TOOLS:
                    print("  *** WARNING: this tool is irreversible ***")

                permission = input("  Allow? [yes/no]: ").strip().lower()

                if permission in ("yes", "y"):
                    # Execute the mock tool and capture its return value
                    result = TOOL_REGISTRY[tool_name].invoke(tool_args)
                    print(f"  Result: {result}")
                    tool_result_message = ToolMessage(
                        content=result,
                        tool_call_id=tool_call_id,
                    )
                else:
                    # Denied — inject a ToolMessage so the conversation stays valid.
                    # The model needs a result for every tool_call_id it emitted.
                    print("  [Permission denied]")
                    tool_result_message = ToolMessage(
                        content="Permission denied by the user. Do not attempt this action again.",
                        tool_call_id=tool_call_id,
                    )

                history.append(tool_result_message)

            # Call the model once more so it can summarise what happened
            # (it now has the ToolMessage results in its history)
            final_response = chat.invoke(history)
            history.append(final_response)
            print(f"\nAI: {strip_thinking(final_response.content)}")

        else:
            # Plain text response — no tool was needed
            print(f"\nAI: {strip_thinking(response.content)}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — choose which pattern to run
# ─────────────────────────────────────────────────────────────────────────────
PATTERNS = {
    "1": ("Interactive Chat REPL",        demo_chat_repl),
    "2": ("Human Approval Gate",          demo_approval_gate),
    "3": ("Iterative Refinement",         demo_iterative_refinement),
    "4": ("Tool Call Permission Gate",    demo_tool_permission_gate),
}

if __name__ == "__main__":
    print("\nHuman in the Loop — Demo")
    print("========================")
    print("Which pattern do you want to explore?")
    for key, (name, _) in PATTERNS.items():
        print(f"  {key}. {name}")
    print("  a. Run all four")

    choice = input("\nChoice [1/2/3/4/a]: ").strip().lower()

    if choice == "a":
        for _, (_, fn) in PATTERNS.items():
            fn()
    elif choice in PATTERNS:
        _, fn = PATTERNS[choice]
        fn()
    else:
        print("Invalid choice. Running pattern 1 (Chat REPL).")
        demo_chat_repl()

    print("\n\nKey takeaways:")
    print("  - HITL = a loop with an input() that pauses execution for a human")
    print("  - history list grows each turn — that IS the conversation memory")
    print("  - SystemMessage sets persona once; never repeat it in the loop")
    print("  - Approval gate: inject AIMessage + feedback HumanMessage to steer")
    print("  - Iterative refinement: same idea, just more targeted feedback turns")
    print("  - Tool call gate: AI emits tool_calls → human gates execution →")
    print("    ToolMessage feeds result back so AI can summarise")
    print("  - ToolMessage is mandatory for every tool_call_id — skip it and API errors")
    print("  - DANGEROUS_TOOLS pattern = extra warning UI for irreversible actions")
    print("  - LangGraph formalises all of this with interrupt() + checkpoints")
    print("    (covered in advanced files)")
