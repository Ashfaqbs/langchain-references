"""
14 - Callbacks & Tracing
=========================
Callbacks let you hook into every step of a LangChain run:
  - When a chain starts/ends
  - When an LLM generates tokens
  - When a tool is called/returns
  - When an error occurs

Use cases:
  - Logging and monitoring (track cost, latency, errors)
  - Token counting and budgeting
  - Progress indicators in UIs
  - LangSmith tracing for debugging
  - Custom metrics and alerting

Callback types:
  - StdOutCallbackHandler   : print everything to stdout
  - FileCallbackHandler     : write logs to file
  - Custom handler          : subclass BaseCallbackHandler

WHEN YOU NEED THIS:
  When an AI feature goes to production and observability is required.
  Think of it like middleware/interceptors — hooks that fire before/after each
  operation without modifying the operation itself. Same pattern as servlet
  filters in Java, middleware in Express, or decorators in Python.

  Real scenarios:
  - PerformanceCallbackHandler (demo_performance_callback):
    Track p95 latency per LLM call. Alert when avg latency exceeds 3s.
    Emit metrics to Prometheus/Grafana.

  - TokenBudgetCallback (demo_budget_callback):
    Cloud LLM APIs charge per token. Set a per-request or per-session budget
    and raise an error before it's exceeded. A runaway agent loop can exhaust
    API credits quickly without this guard.

  - JSONLogCallbackHandler (demo_json_logger):
    Emit structured JSON logs (chain_start, llm_start, tool_call, chain_end)
    to ELK, Datadog, or CloudWatch. Full audit trail of every AI interaction —
    what was asked, what was retrieved, what was returned.

CONFIG vs CONSTRUCTOR callbacks:
  Constructor: callbacks=[handler] on the LLM object → fires for every call
               made through that instance, regardless of call site.
  Config:      config={"callbacks": [handler]} on .invoke() → fires only for
               that specific invocation. Use in web apps for per-request
               tracing where each request gets its own trace context.

THE LANGSMITH ALTERNATIVE:
  Set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY and a full trace UI appears
  with zero extra code — chain inputs/outputs, latency, token usage, errors.
  Use custom callbacks only for behavior LangSmith doesn't cover (budget limits,
  custom metric sinks, proprietary logging pipelines).

Run:  python 14_callbacks_tracing.py
"""

import time
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler, StdOutCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

MODEL = "qwen3:4b"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Built-in StdOutCallbackHandler
# ─────────────────────────────────────────────────────────────────────────────
def demo_stdout_callback():
    print("\n" + "=" * 60)
    print("1. StdOutCallbackHandler — built-in verbose logging")
    print("=" * 60)

    llm = ChatOllama(model=MODEL, temperature=0, callbacks=[StdOutCallbackHandler()])

    prompt = ChatPromptTemplate.from_messages([
        ("human", "What is a goroutine in Go? One sentence.")
    ])

    chain = prompt | llm | StrOutputParser()

    # The callback will print start/end events automatically
    result = chain.invoke({})
    print(f"\nFinal result: {result}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Custom callback — token counter and latency tracker
# ─────────────────────────────────────────────────────────────────────────────
class PerformanceCallbackHandler(BaseCallbackHandler):
    """Tracks token usage, latency, and call counts per run."""

    def __init__(self):
        self.runs: List[Dict] = []
        self._current_run: Dict = {}

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        **kwargs,
    ) -> None:
        self._current_run = {
            "run_id": str(run_id),
            "model": serialized.get("name", "unknown"),
            "start_time": time.time(),
            "prompt_chars": sum(len(p) for p in prompts),
        }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs,
    ) -> None:
        end_time = time.time()
        latency  = end_time - self._current_run.get("start_time", end_time)

        # Extract token usage from response metadata if available
        token_usage = {}
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})

        self._current_run.update({
            "latency_ms": round(latency * 1000),
            "token_usage": token_usage,
            "output_chars": sum(
                len(gen.text) for gen_list in response.generations for gen in gen_list
            ),
        })
        self.runs.append(dict(self._current_run))

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs,
    ) -> None:
        self._current_run["error"] = str(error)
        self.runs.append(dict(self._current_run))

    def summary(self) -> str:
        if not self.runs:
            return "No runs recorded."

        total_latency = sum(r.get("latency_ms", 0) for r in self.runs)
        avg_latency   = total_latency / len(self.runs)

        return (
            f"Runs: {len(self.runs)} | "
            f"Avg latency: {avg_latency:.0f}ms | "
            f"Total latency: {total_latency:.0f}ms"
        )


def demo_performance_callback():
    print("\n" + "=" * 60)
    print("2. Custom PerformanceCallbackHandler — latency + token tracking")
    print("=" * 60)

    perf = PerformanceCallbackHandler()

    llm = ChatOllama(model=MODEL, temperature=0)

    chain = (
        ChatPromptTemplate.from_messages([("human", "What is {topic}? One sentence.")])
        | llm
        | StrOutputParser()
    )

    topics = ["Redis", "Kafka", "Kubernetes"]

    for topic in topics:
        # Pass callback via config (preferred — doesn't pollute chain definition)
        result = chain.invoke(
            {"topic": topic},
            config={"callbacks": [perf]},
        )
        print(f"  [{topic}]: {result[:80]}...")

    print(f"\nPerformance summary: {perf.summary()}")
    print(f"\nDetailed run data:")
    for run in perf.runs:
        print(f"  latency={run['latency_ms']}ms, "
              f"output_chars={run['output_chars']}, "
              f"run_id={run['run_id'][:8]}...")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Custom callback — structured JSON logger
# ─────────────────────────────────────────────────────────────────────────────
class JSONLogCallbackHandler(BaseCallbackHandler):
    """Logs all LangChain events as structured JSON."""

    def __init__(self, log_file: str = "langchain_events.jsonl"):
        self.log_file = log_file
        self.events: List[Dict] = []

    def _log_event(self, event_type: str, data: Dict):
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            **data,
        }
        self.events.append(event)
        # Uncomment to also write to file:
        # with open(self.log_file, "a") as f:
        #     f.write(json.dumps(event) + "\n")

    def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        self._log_event("chain_start", {
            "chain": serialized.get("name", "unknown"),
            "inputs": str(inputs)[:200],
            "run_id": str(run_id),
        })

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._log_event("chain_end", {
            "run_id": str(run_id),
            "output_keys": list(outputs.keys()) if isinstance(outputs, dict) else "str",
        })

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._log_event("llm_start", {
            "model": serialized.get("name", "unknown"),
            "prompt_count": len(prompts),
            "run_id": str(run_id),
        })

    def on_llm_end(self, response, *, run_id, **kwargs):
        self._log_event("llm_end", {
            "run_id": str(run_id),
            "generation_count": len(response.generations),
        })

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self._log_event("tool_start", {
            "tool": serialized.get("name", "unknown"),
            "input": input_str[:200],
            "run_id": str(run_id),
        })

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._log_event("tool_end", {
            "run_id": str(run_id),
            "output": str(output)[:200],
        })


def demo_json_logger():
    print("\n" + "=" * 60)
    print("3. JSON structured logging — production observability pattern")
    print("=" * 60)

    json_logger = JSONLogCallbackHandler()

    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{question}"),
        ])
        | ChatOllama(model=MODEL, temperature=0)
        | StrOutputParser()
    )

    chain.invoke(
        {"question": "What is the CAP theorem?"},
        config={"callbacks": [json_logger]},
    )

    print(f"Captured {len(json_logger.events)} events:")
    for event in json_logger.events:
        print(f"  [{event['event']:15}] {event['timestamp']}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Token budget callback — stop when cost threshold is reached
# ─────────────────────────────────────────────────────────────────────────────
class TokenBudgetCallback(BaseCallbackHandler):
    """Raises an exception if token usage exceeds budget."""

    def __init__(self, max_total_chars: int = 5000):
        self.max_total_chars = max_total_chars
        self.total_chars_used = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        chars_used = sum(
            len(gen.text) for gen_list in response.generations for gen in gen_list
        )
        self.total_chars_used += chars_used

        if self.total_chars_used > self.max_total_chars:
            raise RuntimeError(
                f"Token budget exceeded: {self.total_chars_used} chars used "
                f"(limit: {self.max_total_chars})"
            )


def demo_budget_callback():
    print("\n" + "=" * 60)
    print("4. Token budget callback — cost control")
    print("=" * 60)

    budget = TokenBudgetCallback(max_total_chars=2000)

    chain = (
        ChatPromptTemplate.from_messages([
            ("human", "Write a very detailed explanation of {topic} in 500 words.")
        ])
        | ChatOllama(model=MODEL, temperature=0.5)
        | StrOutputParser()
    )

    topics = ["Kubernetes networking", "Kafka exactly-once semantics", "PostgreSQL MVCC"]

    for topic in topics:
        try:
            result = chain.invoke(
                {"topic": topic},
                config={"callbacks": [budget]},
            )
            used_pct = (budget.total_chars_used / budget.max_total_chars) * 100
            print(f"  [{topic[:30]:30}]: OK ({used_pct:.0f}% budget used)")
        except RuntimeError as e:
            print(f"  [{topic[:30]:30}]: BUDGET EXCEEDED — {e}")
            break

    print(f"\nTotal chars used: {budget.total_chars_used} / {budget.max_total_chars}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Callbacks via config vs constructor — when to use each
# ─────────────────────────────────────────────────────────────────────────────
def demo_callback_scoping():
    print("\n" + "=" * 60)
    print("5. Callback scoping — constructor vs config")
    print("=" * 60)

    perf1 = PerformanceCallbackHandler()
    perf2 = PerformanceCallbackHandler()

    llm_with_callback = ChatOllama(
        model=MODEL, temperature=0,
        callbacks=[perf1],        # fires for ALL invocations of this llm instance
    )

    chain = (
        ChatPromptTemplate.from_messages([("human", "{q}")])
        | ChatOllama(model=MODEL, temperature=0)  # no callback on the LLM
        | StrOutputParser()
    )

    # per-invoke callback — only fires for this specific invocation
    chain.invoke({"q": "What is Docker?"}, config={"callbacks": [perf2]})

    print("Constructor callback (fires always): attached to llm object")
    print("Config callback (per-invoke): attached at call time — more flexible")
    print("Best practice: use config callbacks for per-request tracing in web apps")


if __name__ == "__main__":
    demo_performance_callback()
    demo_json_logger()
    demo_budget_callback()
    demo_callback_scoping()

    print("\n\nKey takeaways:")
    print("  - Callbacks are hooks: on_llm_start/end, on_chain_start/end, on_tool_start/end")
    print("  - Pass via config={'callbacks': [...]} for per-request scoping (web apps)")
    print("  - Pass via constructor callbacks=[...] for always-on handlers")
    print("  - Use callbacks for: latency tracking, token budgets, structured logging")
    print("  - LangSmith gives you all of this automatically with zero extra code")
    print("    Set: LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY=<key>")

# Output
#
# WHAT ARE CALLBACKS?
# -------------------
# Callbacks are hooks (event listeners) that LangChain fires automatically at each
# step of an LLM pipeline. You never call them yourself — LangChain calls them for you.
#
# Lifecycle events that fire callbacks (in order for a chain.invoke() call):
#   1. on_chain_start  — the chain begins, receives inputs
#   2. on_llm_start    — the LLM receives the formatted prompt
#   3. on_llm_end      — the LLM returns its response
#   4. on_chain_end    — the chain finishes, outputs are ready
#   5. on_tool_start/end — if an agent calls a tool
#   6. on_llm_error / on_chain_error — if anything throws
#
# Think of it like a filter chain in Java (javax.servlet.Filter) or middleware in
# Express — each callback handler wraps the operation without modifying it.
#
# WHAT IS TRACING?
# ----------------
# Tracing is recording the full execution tree of a LangChain run:
# which chains fired, what they received as input, what the LLM said, how long
# each step took. Callbacks are the mechanism that makes tracing possible —
# every trace event (chain_start, llm_end, etc.) is just a callback being fired.
# LangSmith is the hosted tracing backend that captures all this automatically.

# ❯❯ langchain-references git:(main) 20:30 & c:\Users\ashfa\Downloads\langchain-references\.venv\Scripts\Activate.ps1
#  (langchain-references) ❯❯ langchain-references git:(main) 20:30 & c:\Users\ashfa\Downloads\langchain-references\.venv\Scripts\python.exe c:/Users/ashfa/Downloads/langchain-references/14_callbacks_tracing.py

# ============================================================
# 2. Custom PerformanceCallbackHandler — latency + token tracking
# ============================================================
#
# For each topic, LangChain fires on_llm_start (recorded start_time) then
# on_llm_end (recorded end_time). The handler computes latency = end - start.
# The INFO line below is the HTTP layer confirming Ollama returned 200 OK.
# The bracketed line is the truncated LLM answer printed by demo_performance_callback().
#
# 2026-03-12 20:31:04,848 | INFO | HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
#   [Redis]: Redis is an open-source, in-memory data structure store used as a database, cach...
# 2026-03-12 20:32:23,189 | INFO | HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
#   [Kafka]: Apache Kafka is a distributed event streaming platform that enables real-time da...
# 2026-03-12 20:32:42,404 | INFO | HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
#   [Kubernetes]: Kubernetes is an open-source system for automating deployment, scaling, and mana...

# Performance summary: Runs: 3 | Avg latency: 43585ms | Total latency: 130755ms
#
# The handler accumulated one dict per LLM call in self.runs[].
# summary() calculates avg over all runs — this is how you'd feed a Prometheus gauge
# or trigger a Slack alert when p95 latency exceeds your SLA.

# Detailed run data:
#   latency=43621ms, output_chars=102, run_id=019ce290...
#   latency=43208ms, output_chars=155, run_id=019ce291...
#   latency=43926ms, output_chars=117, run_id=019ce292...
#
# Each run_id is a UUID generated by LangChain per invocation — it lets you
# correlate the on_llm_start and on_llm_end events for the SAME call even when
# multiple chains run concurrently. In a web app, you'd store this in a request-
# scoped context and log it alongside your trace ID for distributed tracing.

# ============================================================
# 3. JSON structured logging — production observability pattern
# ============================================================
#
# WARNING line: on_chain_start received serialized=None for some inner chain nodes
# (StrOutputParser, etc.) — those components don't provide a serialized dict.
# LangChain catches the AttributeError in the callback so the pipeline keeps running.
# Fix: guard with `if serialized` before calling .get() in on_chain_start.
#
# 2026-03-12 20:33:15,876 | WARNING | Error in JSONLogCallbackHandler.on_chain_start callback: AttributeError("'NoneType' object has no attribute 'get'")
# 2026-03-12 20:33:32,839 | INFO | HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
# 2026-03-12 20:34:17,498 | WARNING | Error in JSONLogCallbackHandler.on_chain_start callback: AttributeError("'NoneType' object has no attribute 'get'")
#
# Despite the warnings, 6 events were still captured — callbacks are fire-and-forget;
# an error in a handler does NOT abort the chain.
#
# Captured 6 events:
#   [chain_start    ] 2026-03-12T15:03:15.877419   <- outer chain (ChatPromptTemplate | LLM | Parser) started
#   [chain_end      ] 2026-03-12T15:03:15.877419   <- a sub-chain (e.g. the prompt template step) finished instantly
#   [llm_start      ] 2026-03-12T15:03:15.878133   <- LLM received the formatted prompt, clock starts here
#   [llm_end        ] 2026-03-12T15:04:17.497667   <- LLM returned; ~62 seconds elapsed (local Ollama model)
#   [chain_end      ] 2026-03-12T15:04:17.498724   <- parser/output step finished
#   [chain_end      ] 2026-03-12T15:04:17.498724   <- outer chain finished
#
# This event stream is the raw material for a distributed trace. In production you'd
# write these JSON lines to a log sink (ELK, Datadog, CloudWatch) and build a
# "chain_start → llm_start → llm_end → chain_end" waterfall view per request.

# ============================================================
# 4. Token budget callback — cost control
# ============================================================
#
# BUG OBSERVED HERE: LangChain's callback dispatcher catches exceptions raised inside
# callbacks and logs them as WARNINGs instead of re-raising them. That's why the
# RuntimeError fired in on_llm_end but the chain still returned "OK" to the caller.
# The budget check ran, but the exception was swallowed by LangChain's dispatcher.
# Real fix: check the budget BEFORE calling the LLM (e.g. in on_llm_start), or use
# a pre-call guard in the chain logic rather than relying on a callback exception.
#
# 2026-03-12 20:34:51,875 | INFO | HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
# 2026-03-12 20:35:28,668 | WARNING | Error in TokenBudgetCallback.on_llm_end callback: RuntimeError('Token budget exceeded: 3606 chars used (limit: 2000)')
#   [Kubernetes networking         ]: OK (180% budget used)   <- should have stopped here but exception was swallowed
# 2026-03-12 20:36:06,468 | INFO | HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
# 2026-03-12 20:36:45,401 | WARNING | Error in TokenBudgetCallback.on_llm_end callback: RuntimeError('Token budget exceeded: 7790 chars used (limit: 2000)')
#   [Kafka exactly-once semantics  ]: OK (390% budget used)   <- same, kept going
# 2026-03-12 20:37:26,458 | INFO | HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
# 2026-03-12 20:38:04,630 | WARNING | Error in TokenBudgetCallback.on_llm_end callback: RuntimeError('Token budget exceeded: 11578 chars used (limit: 2000)')
#   [PostgreSQL MVCC               ]: OK (579% budget used)   <- all 3 topics ran, budget was never enforced

# Total chars used: 11578 / 2000
# All 3 invocations completed; the budget limit was never enforced because LangChain
# swallowed the RuntimeError from inside the callback. Use callbacks for observability
# (logging, metrics) and implement hard budget enforcement as a pre-call check instead.

# ============================================================
# 5. Callback scoping — constructor vs config
# ============================================================
#
# Only one HTTP call happened here (the config-scoped perf2 chain).
# perf1 was attached to llm_with_callback but that LLM instance was never invoked
# in the demo — illustrating that constructor callbacks are idle until the specific
# instance is called. config callbacks fire only for that single chain.invoke() call.
#
# 2026-03-12 20:38:45,846 | INFO | HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
# Constructor callback (fires always): attached to llm object
# Config callback (per-invoke): attached at call time — more flexible
# Best practice: use config callbacks for per-request tracing in web apps
#
# Why config > constructor in web apps:
#   Each HTTP request should get its own trace context (user ID, request ID, etc.).
#   Attaching to the LLM constructor means all requests share the same handler
#   instance, which causes data mixing in multi-threaded/async environments.
#   config={"callbacks": [per_request_handler]} gives each request an isolated handler.


# Key takeaways:
#   - Callbacks are hooks: on_llm_start/end, on_chain_start/end, on_tool_start/end
#   - Pass via config={'callbacks': [...]} for per-request scoping (web apps)
#   - Pass via constructor callbacks=[...] for always-on handlers
#   - Use callbacks for: latency tracking, token budgets, structured logging
#   - LangSmith gives you all of this automatically with zero extra code
#     Set: LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY=<key>
#  (langchain-references) ❯❯ langchain-references git:(main) 20:39