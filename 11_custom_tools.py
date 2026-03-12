"""
11 - Custom Tools (Advanced)
=============================
Building production-quality tools for agents.
Real tools hit APIs, query databases, run code, call services.

Patterns covered:
  1. @tool with complex return types
  2. StructuredTool — Pydantic input schema for multi-arg tools
  3. Async tools — non-blocking I/O in agents
  4. Tools with error handling and validation
  5. Tools that call external APIs (simulated)
  6. Toolkit — grouping related tools

WHEN YOU NEED THIS:
  File 10 shows the concept with toy tools (calculator, string reverser).
  This file is the production version — needed when agent tools must:
  - Accept multiple typed parameters with validation (StructuredTool + Pydantic)
  - Call async external APIs without blocking (_arun)
  - Connect to a real database and return structured query results
  - Create tickets, send notifications, or mutate state in external systems

  Real scenarios:
  - search_knowledge_base: agent searches an internal vector DB
  - create_ticket_tool: agent files a Jira/GitHub issue from a user description
  - DatabaseQueryTool: agent queries PostgreSQL to answer data questions
  - fetch_weather: agent calls an external REST API asynchronously

@tool vs StructuredTool vs BaseTool — when to use each:
  @tool          → single-arg or simple multi-arg functions, fastest to write
  StructuredTool → multi-field inputs with per-field descriptions/validation,
                   needed when the agent must fill a rich input schema
  BaseTool       → full control: custom sync + async, callbacks, error handling;
                   use for production-grade tools that need retry or audit logging

THE DOCSTRING IS THE TOOL'S CONTRACT:
  The agent reads the docstring to decide WHEN to call the tool and HOW to
  format the arguments. A vague docstring leads to the agent misusing the tool.
  Write it precisely — state the inputs, outputs, and when to use this tool
  vs other tools available.

ASYNC TOOLS (_arun) — when this matters:
  A tool that calls an external HTTP API or a database should be async to avoid
  blocking the event loop in an async agent or FastAPI service. Sync tools
  in an async context serialize what could run concurrently.

Run:  python 11_custom_tools.py
"""

import asyncio
import json
import random
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import tool, StructuredTool, BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

# ─────────────────────────────────────────────────────────────────────────────
# 1. @tool with type hints — input schema is auto-generated from signature
# ─────────────────────────────────────────────────────────────────────────────
@tool
def search_knowledge_base(query: str, max_results: int = 3) -> str:
    """
    Search the internal knowledge base for information.
    Returns the top matching documents as a formatted string.

    Args:
        query: The search query to find relevant documents
        max_results: Maximum number of results to return (default: 3, max: 10)
    """
    # Simulated knowledge base
    knowledge = {
        "kafka":      "Kafka is a distributed event streaming platform using topics and partitions.",
        "redis":      "Redis is an in-memory data store for caching, sessions, and pub/sub.",
        "postgres":   "PostgreSQL is an ACID-compliant relational database with advanced features.",
        "docker":     "Docker containers package apps and their dependencies for consistent deployment.",
        "kubernetes": "Kubernetes orchestrates containers with scaling, routing, and self-healing.",
        "spring":     "Spring Boot simplifies Java microservices with auto-configuration.",
        "fastapi":    "FastAPI is a Python framework with automatic validation and OpenAPI docs.",
    }

    max_results = min(max_results, 10)  # cap at 10
    query_lower = query.lower()

    matches = [
        f"[{k}]: {v}" for k, v in knowledge.items()
        if any(word in k or word in v.lower() for word in query_lower.split())
    ]

    if not matches:
        return f"No results found for query: '{query}'"

    return "\n".join(matches[:max_results])


# ─────────────────────────────────────────────────────────────────────────────
# 2. StructuredTool — Pydantic schema for complex multi-field inputs
# ─────────────────────────────────────────────────────────────────────────────

class CreateTicketInput(BaseModel):
    """Input schema for creating a support ticket."""
    title: str = Field(description="Short title describing the issue")
    description: str = Field(description="Detailed description of the problem")
    priority: str = Field(
        default="medium",
        description="Ticket priority: 'low', 'medium', 'high', 'critical'"
    )
    assignee: Optional[str] = Field(
        default=None,
        description="Username to assign the ticket to (optional)"
    )


def create_ticket(title: str, description: str, priority: str = "medium", assignee: Optional[str] = None) -> str:
    """Create a support ticket in the ticketing system."""
    valid_priorities = {"low", "medium", "high", "critical"}
    if priority not in valid_priorities:
        return f"Error: Invalid priority '{priority}'. Must be one of: {valid_priorities}"

    ticket_id = f"TICKET-{random.randint(1000, 9999)}"
    ticket = {
        "id": ticket_id,
        "title": title,
        "priority": priority,
        "status": "open",
        "assignee": assignee or "unassigned",
    }
    return f"Ticket created: {json.dumps(ticket, indent=2)}"


# Wrap as StructuredTool with explicit schema
create_ticket_tool = StructuredTool.from_function(
    func=create_ticket,
    name="create_support_ticket",
    description="Create a support ticket for tracking issues. Use this when asked to log or report a problem.",
    args_schema=CreateTicketInput,
    return_direct=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. BaseTool subclass — full control, best for complex tools
# ─────────────────────────────────────────────────────────────────────────────

class DatabaseQueryInput(BaseModel):
    """Input for the database query tool."""
    table: str = Field(description="Table name to query")
    condition: Optional[str] = Field(default=None, description="WHERE clause condition")
    limit: int = Field(default=10, description="Max rows to return")


class DatabaseQueryTool(BaseTool):
    """
    Query a simulated database.
    Inheriting from BaseTool gives you full control over sync/async,
    error handling, callbacks, and metadata.
    """

    name: str = "database_query"
    description: str = (
        "Query the internal database for records. "
        "Useful for finding user data, orders, products, etc."
    )
    args_schema: Type[BaseModel] = DatabaseQueryInput

    # Simulated DB data
    _fake_db = {
        "users": [
            {"id": 1, "name": "Ashfa", "role": "admin", "active": True},
            {"id": 2, "name": "Alice", "role": "engineer", "active": True},
            {"id": 3, "name": "Bob",   "role": "analyst",  "active": False},
        ],
        "orders": [
            {"id": 101, "user_id": 1, "product": "Kafka License", "amount": 999},
            {"id": 102, "user_id": 2, "product": "Redis License",  "amount": 499},
            {"id": 103, "user_id": 1, "product": "K8s Support",    "amount": 1999},
        ],
    }

    def _run(
        self,
        table: str,
        condition: Optional[str] = None,
        limit: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if table not in self._fake_db:
            return f"Table '{table}' not found. Available: {list(self._fake_db.keys())}"

        rows = self._fake_db[table][:limit]

        if condition:
            # Simple filter: "active=True" or "user_id=1"
            try:
                key, value = condition.split("=", 1)
                key, value = key.strip(), value.strip()
                # Convert value type
                if value.lower() == "true":   value = True
                elif value.lower() == "false": value = False
                elif value.isdigit():           value = int(value)
                rows = [r for r in rows if str(r.get(key)) == str(value)]
            except ValueError:
                return f"Invalid condition format. Use: key=value"

        return json.dumps(rows, indent=2) if rows else "No records found."

    async def _arun(
        self,
        table: str,
        condition: Optional[str] = None,
        limit: int = 10,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        # Async version — same logic but non-blocking
        await asyncio.sleep(0.01)  # simulate async DB call
        return self._run(table, condition, limit)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Async tool — for non-blocking external API calls
# ─────────────────────────────────────────────────────────────────────────────
@tool
async def fetch_weather(city: str) -> str:
    """
    Fetch the current weather for a given city.
    Returns temperature, humidity, and conditions as a string.
    """
    # Simulated async API call
    await asyncio.sleep(0.1)

    weather_data = {
        "london": {"temp": 12, "humidity": 78, "condition": "cloudy"},
        "tokyo":  {"temp": 22, "humidity": 65, "condition": "sunny"},
        "dubai":  {"temp": 38, "humidity": 45, "condition": "hot and sunny"},
    }

    city_lower = city.lower()
    data = weather_data.get(city_lower, {"temp": 20, "humidity": 60, "condition": "clear"})
    return f"{city}: {data['temp']}°C, humidity {data['humidity']}%, {data['condition']}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Tool with retry and error handling
# ─────────────────────────────────────────────────────────────────────────────
@tool
def call_external_api(endpoint: str, method: str = "GET") -> str:
    """
    Call an external API endpoint.
    Args:
        endpoint: The API endpoint path (e.g., '/users', '/orders/123')
        method: HTTP method: GET, POST, PUT, DELETE
    """
    valid_methods = {"GET", "POST", "PUT", "DELETE"}
    method = method.upper()

    if method not in valid_methods:
        return f"Error: Invalid HTTP method '{method}'. Must be one of {valid_methods}"

    if not endpoint.startswith("/"):
        return f"Error: Endpoint must start with '/'. Got: '{endpoint}'"

    # Simulate 20% failure rate for resilience demo
    if random.random() < 0.2:
        raise ConnectionError(f"Simulated connection failure for {endpoint}")

    # Simulated response
    return json.dumps({
        "status": 200,
        "method": method,
        "endpoint": endpoint,
        "data": f"Simulated response for {method} {endpoint}",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Demo: show all tools working
# ─────────────────────────────────────────────────────────────────────────────
def demo_tool_usage():
    print("\n" + "=" * 60)
    print("Tool Demonstrations")
    print("=" * 60)

    # 1. @tool with optional args
    print("\n1. search_knowledge_base:")
    print(search_knowledge_base.invoke({"query": "distributed streaming", "max_results": 2}))

    # 2. StructuredTool
    print("\n2. create_ticket_tool:")
    print(create_ticket_tool.invoke({
        "title": "Login page broken",
        "description": "Users cannot log in after the last deployment.",
        "priority": "high",
        "assignee": "ashfa",
    }))

    # 3. BaseTool subclass
    print("\n3. DatabaseQueryTool:")
    db_tool = DatabaseQueryTool()
    print(db_tool.invoke({"table": "users", "condition": "active=True"}))

    print("\n4. DB query — orders table:")
    print(db_tool.invoke({"table": "orders", "condition": "user_id=1"}))

    # 4. Async tool
    print("\n5. fetch_weather (async, run synchronously):")
    result = asyncio.run(fetch_weather.ainvoke({"city": "Tokyo"}))
    print(result)

    # 5. Tool metadata
    print("\n6. Tool schemas (what the agent sees):")
    all_tools = [search_knowledge_base, create_ticket_tool, db_tool]
    for t in all_tools:
        print(f"\n  {t.name}:")
        print(f"    {t.description[:80]}...")


# ─────────────────────────────────────────────────────────────────────────────
# Demo: use tools inside an agent
# ─────────────────────────────────────────────────────────────────────────────
def demo_agent_with_custom_tools():
    print("\n" + "=" * 60)
    print("Agent using custom tools")
    print("=" * 60)

    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, ToolMessage

    db_tool = DatabaseQueryTool()
    tools = [search_knowledge_base, create_ticket_tool, db_tool]
    tool_map = {t.name: t for t in tools}

    llm = ChatOllama(model="llama3.2", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [HumanMessage(content="Search for information about Redis in the knowledge base.")]

    print(f"User: {messages[0].content}")

    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"Agent: {response.content}")
            break

        for call in response.tool_calls:
            print(f"  [Calling tool: {call['name']} with {call['args']}]")
            tool_fn = tool_map.get(call["name"])
            result  = tool_fn.invoke(call["args"]) if tool_fn else "Tool not found"
            print(f"  [Result: {str(result)[:100]}]")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))


if __name__ == "__main__":
    demo_tool_usage()
    demo_agent_with_custom_tools()

    print("\n\nKey takeaways:")
    print("  - @tool is the fastest way to make a tool — docstring = description")
    print("  - StructuredTool + Pydantic schema gives you type validation + rich descriptions")
    print("  - BaseTool subclass gives full control: sync/async, callbacks, error handling")
    print("  - Async tools (_arun) are essential for I/O-heavy tools in async agents")
    print("  - Tool description quality directly impacts agent decision making — be explicit")
    print("  - Include error messages in tool return values so the agent can recover")
