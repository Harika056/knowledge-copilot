"""Decides whether a question needs the documents or the structured database."""
import re

SQL_WORDS = re.compile(
    r"\b(how many|count|total|sum|average|avg|median|top \d+|highest|lowest|most|least|"
    r"revenue|per (month|region|customer|product)|by (month|region|customer|product)|number of)\b", re.I)
DATA_WORDS = re.compile(r"\b(orders?|customers?|revenue|sales|units|region|amount)\b", re.I)


def route(question: str) -> str:
    """'sql' only when the question both asks for an aggregate and mentions our data tables."""
    return "sql" if SQL_WORDS.search(question) and DATA_WORDS.search(question) else "docs"
