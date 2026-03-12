"""
13 - Structured Output
========================
Getting LLMs to return structured, typed data instead of freeform text.
Critical for building reliable pipelines where downstream code needs to
process the model's output programmatically.

Approaches:
  1. .with_structured_output(PydanticModel)  — cleanest, most reliable
  2. JsonOutputParser                         — raw dict, no type safety
  3. PydanticOutputParser + format_instructions — for models without tool calling
  4. Function/tool calling schema             — explicit JSON schema

WHEN YOU NEED THIS:
  Whenever application logic needs to branch or act on the LLM's output.
  Free-form text is fine for display. Structured output is needed when
  downstream code must read specific fields — same reason JSON is preferred
  over plain text for API responses.

  Think of it like deserialization — JSON → typed object in Jackson/Pydantic.
  Code can call result.severity or result.intent directly, with type safety.

  Real scenarios:
  - Bug triage (demo_with_structured_output):
    User describes a bug → extract severity, component, steps_to_reproduce →
    auto-assign to the right team in Jira based on those fields
  - Architecture analysis (demo_nested_structured):
    Paste a system description → extract technologies, scalability concerns,
    recommendations as typed lists → populate a structured report
  - Message routing (demo_classification):
    Incoming support message → classify intent + urgency →
    if urgency=True and requires_human=True → escalate immediately
  - Data extraction pipeline (demo_extraction_pipeline):
    Parse a resume or bio → extract name, skills[], years_experience →
    insert into DB without writing any manual parsing logic

FIELD DESCRIPTIONS ARE THE INSTRUCTIONS:
  Pydantic Field(description="...") is what the model reads to fill each field.
  Be explicit and specific: "Severity level: must be one of 'low', 'medium',
  'high', 'critical'" produces better output than just "severity".

ENUM FIELDS — constrain model choices:
  severity: Severity (Enum) forces the model to return exactly one of the defined
  values. Without Enum, output varies ("HIGH", "High", "high") and equality
  checks in downstream code break unpredictably.

Run:  python 13_structured_output.py
"""

from enum import Enum
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama

MODEL = "qwen3:4b"
llm = ChatOllama(model=MODEL, temperature=0)

# ─────────────────────────────────────────────────────────────────────────────
# 1. .with_structured_output() — the cleanest approach
# ─────────────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    low      = "low"
    medium   = "medium"
    high     = "high"
    critical = "critical"


class BugReport(BaseModel):
    """Structured bug report extracted from a description."""
    title: str = Field(description="Short title for the bug, max 10 words")
    component: str = Field(description="The software component affected (e.g., 'auth', 'database', 'UI')")
    severity: Severity = Field(description="Bug severity level")
    steps_to_reproduce: List[str] = Field(description="Ordered list of steps to reproduce the bug")
    expected_behavior: str = Field(description="What should happen")
    actual_behavior: str = Field(description="What actually happens")
    suggested_fix: Optional[str] = Field(default=None, description="Suggested fix if known")


def demo_with_structured_output():
    print("\n" + "=" * 60)
    print("1. .with_structured_output() — Pydantic model extraction")
    print("=" * 60)

    # .with_structured_output() wraps the LLM to always return the Pydantic model
    structured_llm = llm.with_structured_output(BugReport)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a QA engineer. Extract structured bug information from the description."),
        ("human", "{bug_description}"),
    ])

    chain = prompt | structured_llm

    bug_desc = """
    When users try to login with their Google OAuth account after the 2.3.1 release,
    they get a 500 error. The page just shows 'Internal Server Error'. This worked fine
    in 2.3.0. It seems to only affect users who have special characters in their email.
    I think the OAuth callback URL parsing is broken.
    """

    result = chain.invoke({"bug_description": bug_desc})

    print(f"Type              : {type(result)}")
    print(f"Title             : {result.title}")
    print(f"Component         : {result.component}")
    print(f"Severity          : {result.severity.value}")
    print(f"Expected          : {result.expected_behavior}")
    print(f"Actual            : {result.actual_behavior}")
    print(f"Steps to reproduce:")
    for i, step in enumerate(result.steps_to_reproduce, 1):
        print(f"  {i}. {step}")
    print(f"Suggested fix     : {result.suggested_fix}")

    # Full dict representation
    print(f"\nAs dict: {result.model_dump()}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Nested Pydantic models — complex structured output
# ─────────────────────────────────────────────────────────────────────────────

class Technology(BaseModel):
    name: str = Field(description="Technology or tool name")
    version: Optional[str] = Field(default=None, description="Version if mentioned")
    purpose: str = Field(description="Its role in the architecture")


class ArchitectureAnalysis(BaseModel):
    """Analysis of a system architecture description."""
    system_name: str = Field(description="Name or identifier of the system")
    architecture_style: str = Field(description="e.g., microservices, monolith, event-driven, serverless")
    technologies: List[Technology] = Field(description="List of technologies mentioned")
    scalability_concerns: List[str] = Field(description="Identified scalability issues or notes")
    recommendations: List[str] = Field(description="Actionable recommendations")
    complexity_score: int = Field(description="Complexity from 1 (simple) to 10 (highly complex)", ge=1, le=10)


def demo_nested_structured():
    print("\n" + "=" * 60)
    print("2. Nested Pydantic models — complex extraction")
    print("=" * 60)

    structured_llm = llm.with_structured_output(ArchitectureAnalysis)

    architecture_desc = """
    Our payment service is a Java Spring Boot 3 application running on Kubernetes.
    It uses PostgreSQL 15 for transaction records, Redis 7 for idempotency keys,
    and publishes events to Kafka 3.6 for downstream processing. The service handles
    about 5000 TPS at peak. We're currently running 3 replicas but seeing high DB
    connection pool contention. We use Flyway for migrations and JUnit 5 for testing.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior architect. Analyze the architecture description and extract structured insights."),
        ("human", "{description}"),
    ])

    chain = prompt | structured_llm
    result = chain.invoke({"description": architecture_desc})

    print(f"System       : {result.system_name}")
    print(f"Style        : {result.architecture_style}")
    print(f"Complexity   : {result.complexity_score}/10")
    print(f"\nTechnologies ({len(result.technologies)}):")
    for tech in result.technologies:
        print(f"  - {tech.name} {tech.version or ''}: {tech.purpose}")
    print(f"\nScalability concerns:")
    for concern in result.scalability_concerns:
        print(f"  ! {concern}")
    print(f"\nRecommendations:")
    for rec in result.recommendations:
        print(f"  → {rec}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Classification / routing with structured output
# ─────────────────────────────────────────────────────────────────────────────

class MessageClassification(BaseModel):
    """Classification of an incoming user message."""
    intent: str = Field(description="Primary intent: 'question', 'complaint', 'feature_request', 'bug_report', 'greeting', 'other'")
    topic: str = Field(description="Main topic of the message")
    sentiment: str = Field(description="Sentiment: 'positive', 'neutral', 'negative'")
    urgency: bool = Field(description="True if the message indicates urgency")
    requires_human: bool = Field(description="True if this should be escalated to a human agent")
    confidence: float = Field(description="Classification confidence from 0.0 to 1.0")


def demo_classification():
    print("\n" + "=" * 60)
    print("3. Structured classification — routing and intent detection")
    print("=" * 60)

    structured_llm = llm.with_structured_output(MessageClassification)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify the user support message accurately."),
        ("human", "Classify: {message}"),
    ])

    chain = prompt | structured_llm

    messages = [
        "Hi! Just wanted to say your product is amazing, love the new dashboard.",
        "MY ACCOUNT HAS BEEN HACKED! All my data is gone! HELP ME NOW!",
        "Would it be possible to add dark mode to the settings page?",
        "The export to CSV button doesn't work on Firefox 120. No error shown.",
    ]

    for msg in messages:
        result = chain.invoke({"message": msg})
        print(f"\nMessage  : {msg[:60]}...")
        print(f"Intent   : {result.intent}")
        print(f"Sentiment: {result.sentiment}")
        print(f"Urgency  : {result.urgency}")
        print(f"Escalate : {result.requires_human}")
        print(f"Confidence: {result.confidence:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Union types — dynamic return type based on content
# ─────────────────────────────────────────────────────────────────────────────

class CodeAnswer(BaseModel):
    """Answer that includes code."""
    language: str = Field(description="Programming language")
    code: str = Field(description="The code snippet")
    explanation: str = Field(description="Explanation of the code")


class TextAnswer(BaseModel):
    """Conceptual text answer without code."""
    answer: str = Field(description="The answer text")
    key_points: List[str] = Field(description="Key points as a list")


def demo_extraction_pipeline():
    print("\n" + "=" * 60)
    print("4. Data extraction pipeline — pull structured data from text")
    print("=" * 60)

    class PersonInfo(BaseModel):
        """Extracted person information."""
        name: str = Field(description="Full name of the person")
        job_title: str = Field(description="Job title or role")
        technologies: List[str] = Field(description="Technologies they work with")
        years_experience: Optional[int] = Field(default=None, description="Years of experience if mentioned")
        location: Optional[str] = Field(default=None, description="Location if mentioned")

    structured_llm = llm.with_structured_output(PersonInfo)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract person information from the bio."),
        ("human", "Bio: {bio}"),
    ])

    chain = prompt | structured_llm

    bios = [
        "Sarah Chen is a Principal Backend Engineer at Stripe with 8 years of experience. She specializes in Java, Kotlin, and Kafka-based event streaming from her office in San Francisco.",
        "Carlos is a full-stack dev who loves Python, React, and PostgreSQL. He's been building SaaS products for 4 years.",
    ]

    for bio in bios:
        result = chain.invoke({"bio": bio})
        print(f"\nBio     : {bio[:70]}...")
        print(f"Name    : {result.name}")
        print(f"Title   : {result.job_title}")
        print(f"Stack   : {', '.join(result.technologies)}")
        print(f"Years   : {result.years_experience}")
        print(f"Location: {result.location}")


if __name__ == "__main__":
    demo_with_structured_output()
    demo_nested_structured()
    demo_classification()
    demo_extraction_pipeline()

    print("\n\nKey takeaways:")
    print("  - .with_structured_output(Model) is the cleanest way to get typed output")
    print("  - Pydantic Field(description=...) is the instruction the model uses to fill each field")
    print("  - Nested models work — great for complex hierarchical extraction")
    print("  - Use Enum fields to constrain choices (severity, intent, etc.)")
    print("  - Classification with structured output → deterministic routing logic")
    print("  - Field validators (ge=, le=, min_length=) enforce constraints at parse time")

# Output:
# (langchain-references) ❯❯ langchain-references git:(main) 20:25 python .\13_structured_output.py

# ============================================================
# 1. .with_structured_output() — Pydantic model extraction
# ============================================================
# Type              : <class '__main__.BugReport'>
# Title             : Google OAuth login fails with 500 error for users with special characters in email after 2.3.1 release
# Component         : Authentication
# Severity          : medium
# Expected          : Successful login via Google OAuth without errors
# Actual            : 500 Internal Server Error response
# Steps to reproduce:
#   1. Attempt to log in with Google OAuth account
#   2. Ensure user email contains special characters (e.g., 'user@domain!.com')
#   3. Observe 500 Internal Server Error response
# Suggested fix     : Fix OAuth callback URL parsing to handle special characters in email addresses (e.g., validate email format before processing callback URL parameters in 2.3.1 release).

# As dict: {'title': 'Google OAuth login fails with 500 error for users with special characters in email after 2.3.1 release', 'component': 'Authentication', 'severity': <Severity.medium: 'medium'>, 'steps_to_reproduce': ['Attempt to log in with Google OAuth account', "Ensure user email contains special characters (e.g., 'user@domain!.com')", 'Observe 500 Internal Server Error response'], 'expected_behavior': 'Successful login via Google OAuth without errors', 'actual_behavior': '500 Internal Server Error response', 'suggested_fix': 'Fix OAuth callback URL parsing to handle special characters in email addresses (e.g., validate email format before processing callback URL parameters in 2.3.1 release).'}

# ============================================================
# 2. Nested Pydantic models — complex extraction
# ============================================================
# System       : Payment Service Architecture Analysis
# Style        : Microservices (Spring Boot 3) with Event-Driven Pattern
# Complexity   : 7/10

# Technologies (7):
#   - Application Spring Boot 3.0+: Payment processing service
#   - Database PostgreSQL 15: Transaction records
#   - Caching Redis 7: Idempotency key management
#   - Event Streaming Kafka 3.6: Downstream processing
#   - Orchestration Kubernetes: Cluster management
#   - Migration Flyway: Database schema management
#   - Testing JUnit 5: Unit/integration testing

# Scalability concerns:
#   ! High DB connection pool contention at 5000 TPS peak
#   ! 3 replicas insufficient for current load profile

# Recommendations:
#   → Increase HikariCP connection pool size to 500-1000 per replica (current default is 10)
#   → Implement connection pool metrics monitoring (HikariCP metrics endpoint)
#   → Add horizontal scaling to 5-7 replicas based on connection pool utilization
#   → Optimize PostgreSQL connection parameters (max_connections, work_mem)
#   → Add connection pool warm-up strategy for Kubernetes pod startup

# ============================================================
# 3. Structured classification — routing and intent detection
# ============================================================

# Message  : Hi! Just wanted to say your product is amazing, love the new...
# Intent   : Positive feedback
# Sentiment: Positive
# Urgency  : False
# Escalate : False
# Confidence: 0.95

# Message  : MY ACCOUNT HAS BEEN HACKED! All my data is gone! HELP ME NOW...
# Intent   : Security Breach
# Sentiment: Urgent
# Urgency  : True
# Escalate : True
# Confidence: 0.95

# Message  : Would it be possible to add dark mode to the settings page?...
# Intent   : Feature Request
# Sentiment: Neutral
# Urgency  : False
# Escalate : False
# Confidence: 0.95

# Message  : The export to CSV button doesn't work on Firefox 120. No err...
# Intent   : Bug report
# Sentiment: Negative
# Urgency  : True
# Escalate : True
# Confidence: 0.95

# ============================================================
# 4. Data extraction pipeline — pull structured data from text
# ============================================================

# Bio     : Sarah Chen is a Principal Backend Engineer at Stripe with 8 years of e...
# Name    : Sarah Chen
# Title   : Principal Backend Engineer
# Stack   : Java, Kotlin, Kafka-based event streaming
# Years   : 8
# Location: San Francisco

# Bio     : Carlos is a full-stack dev who loves Python, React, and PostgreSQL. He...
# Name    : Carlos
# Title   : full-stack developer
# Stack   : Python, React, PostgreSQL
# Years   : 4
# Location: N/A (not specified in bio)


# Key takeaways:
#   - .with_structured_output(Model) is the cleanest way to get typed output
#   - Pydantic Field(description=...) is the instruction the model uses to fill each field
#   - Nested models work — great for complex hierarchical extraction
#   - Use Enum fields to constrain choices (severity, intent, etc.)
#   - Classification with structured output → deterministic routing logic
#   - Field validators (ge=, le=, min_length=) enforce constraints at parse time
#  (langchain-references) ❯❯ langchain-references git:(main) 20:29
