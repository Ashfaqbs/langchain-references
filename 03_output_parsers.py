"""
03 - Output Parsers
====================
The LLM outputs a raw AIMessage. Output parsers transform that into the
exact Python type you need: a string, a dict, a list, or a typed Pydantic object.

Parsers to cover:
  - StrOutputParser       : AIMessage → str         (most common)
  - JsonOutputParser      : AIMessage → dict         (JSON extraction)
  - PydanticOutputParser  : AIMessage → Pydantic obj (typed, validated)
  - CommaSeparatedListOutputParser : AIMessage → List[str]
  - StructuredOutputParser        : custom field schema

WHEN YOU NEED THIS:
  Whenever downstream code needs to act on the model's output programmatically.
  Think of it like JSON deserialization — Jackson in Java, Pydantic in Python,
  json.loads() in a script. A typed object comes back instead of raw text, so
  code can call result.severity or result.intent without regex parsing.

  Real scenarios:
  - Bug triage bot: extract severity, component, steps → route to correct team
  - Resume parser: extract name, skills[], years_experience → insert into DB
  - Support classifier: extract intent + urgency → trigger escalation logic
  - Code generator: extract language + code block → run in sandbox

WHICH PARSER TO USE:
  - StrOutputParser           → always, as the final step of any text chain
  - JsonOutputParser          → quick dict extraction, no type safety needed
  - .with_structured_output() → production use, typed Pydantic model, validated
  - CommaSeparatedListParser  → when a simple Python list is all that's needed

Run:  python 03_output_parsers.py
"""

from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List

MODEL = "qwen3:0.6b"
llm = ChatOllama(model=MODEL, temperature=0)

# ─────────────────────────────────────────────────────────────────────────────
# 1. StrOutputParser — AIMessage → plain string
# ─────────────────────────────────────────────────────────────────────────────
def demo_str_parser():
    print("\n" + "=" * 60)
    print("1. StrOutputParser (AIMessage → str)")
    print("=" * 60)

    # Without parser: you get an AIMessage object
    raw = llm.invoke("What color is the sky?")
    print(f"Raw type    : {type(raw)}")
    print(f"Raw content : {raw.content}")

    # With StrOutputParser: you get a plain string
    chain = ChatPromptTemplate.from_messages([
        ("human", "{question}")
    ]) | llm | StrOutputParser()

    result = chain.invoke({"question": "What color is the sky?"})
    print(f"\nParsed type : {type(result)}")
    print(f"Parsed      : {result}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. JsonOutputParser — extract structured JSON from the model
# ─────────────────────────────────────────────────────────────────────────────
def demo_json_parser():
    print("\n" + "=" * 60)
    print("2. JsonOutputParser (AIMessage → dict)")
    print("=" * 60)

    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You always respond with valid JSON only. No extra text."),
        ("human", (
            "Return a JSON object with these fields: "
            "name (string), age (int), skills (list of strings). "
            "Create a fictional Python developer profile."
        )),
    ])

    chain = prompt | llm | parser

    result = chain.invoke({})
    print(f"Result type : {type(result)}")
    print(f"Result      : {result}")
    print(f"Name field  : {result.get('name')}")
    print(f"Skills      : {result.get('skills')}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PydanticOutputParser — JSON → typed Pydantic model
# ─────────────────────────────────────────────────────────────────────────────
def demo_pydantic_parser():
    print("\n" + "=" * 60)
    print("3. PydanticOutputParser (AIMessage → Pydantic model)")
    print("=" * 60)

    # Define your expected output schema
    class MovieReview(BaseModel):
        title: str = Field(description="Title of the movie")
        year: int = Field(description="Year of release")
        rating: float = Field(description="Rating out of 10")
        summary: str = Field(description="One-sentence summary")
        recommended: bool = Field(description="Whether you recommend it")

    # This is the new LangChain v1 way — use .with_structured_output()
    # (PydanticOutputParser also works but with_structured_output is cleaner)
    structured_llm = llm.with_structured_output(MovieReview)

    prompt = ChatPromptTemplate.from_messages([
        ("human", "Give me a review of the movie: {movie}")
    ])

    chain = prompt | structured_llm

    result = chain.invoke({"movie": "Inception"})
    print(f"Result type  : {type(result)}")
    print(f"Title        : {result.title}")
    print(f"Year         : {result.year}")
    print(f"Rating       : {result.rating}")
    print(f"Recommended  : {result.recommended}")
    print(f"Summary      : {result.summary}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CommaSeparatedListOutputParser — "a, b, c" → ["a", "b", "c"]
# ─────────────────────────────────────────────────────────────────────────────
def demo_list_parser():
    print("\n" + "=" * 60)
    print("4. CommaSeparatedListOutputParser (str → List[str])")
    print("=" * 60)

    parser = CommaSeparatedListOutputParser()

    # get_format_instructions() returns the instruction to inject into the prompt
    format_instructions = parser.get_format_instructions()
    print("Format instructions injected into prompt:")
    print(f"  {format_instructions}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "{format_instructions}"),
        ("human", "List 5 popular Python web frameworks."),
    ])

    chain = prompt | llm | parser

    result = chain.invoke({"format_instructions": format_instructions})
    print(f"\nResult type : {type(result)}")
    print(f"Result      : {result}")
    for i, item in enumerate(result, 1):
        print(f"  {i}. {item.strip()}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Chaining multiple parsers / transforms with RunnableLambda
# ─────────────────────────────────────────────────────────────────────────────
def demo_custom_parse():
    print("\n" + "=" * 60)
    print("5. Custom post-processing with RunnableLambda")
    print("=" * 60)

    from langchain_core.runnables import RunnableLambda

    # Custom parser: extract just the number from the response
    def extract_number(text: str) -> int:
        import re
        numbers = re.findall(r'\d+', text)
        return int(numbers[0]) if numbers else 0

    chain = (
        ChatPromptTemplate.from_messages([
            ("human", "How many days are in a year? Reply with just the number.")
        ])
        | llm
        | StrOutputParser()
        | RunnableLambda(extract_number)  # custom transform step
    )

    result = chain.invoke({})
    print(f"Result type : {type(result)}")
    print(f"Days in year: {result}")


if __name__ == "__main__":
    demo_str_parser()
    demo_json_parser()
    demo_pydantic_parser()
    demo_list_parser()
    demo_custom_parse()

    print("\n\nKey takeaways:")
    print("  - StrOutputParser is the most common — always add it to text chains")
    print("  - JsonOutputParser handles JSON extraction but isn't type-safe")
    print("  - .with_structured_output(PydanticModel) is the cleanest typed approach")
    print("  - Parser.get_format_instructions() generates the prompt injection text")
    print("  - Parsers are Runnables — they slot into | chains naturally")
