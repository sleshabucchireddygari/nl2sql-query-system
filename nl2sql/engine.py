"""Natural language -> SQL engine.

Pipeline for each question:

    question
      -> schema grounding   (LlamaIndex ObjectIndex picks relevant tables)
      -> SQL generation     (LlamaIndex LLM + prompt with the real schema)
      -> validation         (sqlglot static checks + EXPLAIN dry run)
      -> self-correction    (validation errors fed back to the LLM, up to N retries)
      -> execution          (read-only, row-limited) -> pandas DataFrame
      -> optional plain-English answer summarising the result
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import pandas as pd
from llama_index.core import PromptTemplate, SQLDatabase
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from sqlalchemy import create_engine, text

from .config import Settings
from .schema import SchemaCatalog, TableRetriever
from .validator import SQLValidator

TEXT_TO_SQL_PROMPT = PromptTemplate(
    """You are an expert data analyst who writes {dialect} SQL.
Write ONE read-only SQL query that answers the question, using ONLY the tables and
columns listed in the schema below. Rules:
- Use exact table/column names from the schema; never invent columns.
- When filtering text columns, use the exact listed values (case-sensitive).
- Use explicit JOINs on the documented foreign keys.
- Give aggregated columns readable aliases (e.g. AS avg_length_of_stay).
- Round averages/ratios to 2 decimals.
- Do not add a LIMIT unless the question asks for top/bottom N.
- Return only the SQL inside a ```sql code block, with no explanation.

Schema:
{schema}

Question: {question}
{feedback}"""
)

FEEDBACK_TEMPLATE = """
Your previous query was:
```sql
{sql}
```
It failed validation with these errors:
{errors}
Fix every error and return the corrected SQL."""

ANSWER_PROMPT = PromptTemplate(
    """Question: {question}
SQL used: {sql}
Result (first rows, CSV):
{rows}

Answer the question in one or two plain-English sentences using only the result above.
If the result is empty, say no matching records were found."""
)


@dataclass
class Attempt:
    sql: str
    errors: list[str]


@dataclass
class QueryResult:
    question: str
    sql: str = ""
    success: bool = False
    tables_used: list[str] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    data: pd.DataFrame | None = None
    truncated: bool = False
    answer: str = ""
    error: str = ""
    seconds: float = 0.0

    @property
    def retries(self) -> int:
        return max(0, len(self.attempts) - 1)


def extract_sql(response: str) -> str:
    """Pull the SQL out of an LLM response (code fence, 'SQLQuery:' prefix, or raw)."""
    fence = re.search(r"```(?:sql)?\s*(.*?)```", response, re.S | re.I)
    if fence:
        return fence.group(1).strip()
    m = re.search(r"SQLQuery:\s*(.*?)(?:SQLResult:|$)", response, re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b(WITH|SELECT)\b.*", response, re.S | re.I)
    return (m.group(0) if m else response).strip()


class NL2SQLEngine:
    def __init__(
        self,
        sql_database: SQLDatabase,
        llm: LLM,
        embed_model: BaseEmbedding | None = None,
        *,
        use_table_retrieval: bool = True,
        table_top_k: int = 4,
        validate: bool = True,
        max_retries: int = 2,
        row_limit: int = 500,
        descriptions_path=None,
    ):
        self.llm = llm
        self.catalog = SchemaCatalog(sql_database, descriptions_path)
        self.validator = SQLValidator(self.catalog)
        self.validate = validate
        self.max_retries = max_retries if validate else 0
        self.row_limit = row_limit
        self.table_retriever = None
        self.retrieval_warning = ""
        if use_table_retrieval and embed_model is not None and len(self.catalog.tables) > table_top_k:
            try:
                self.table_retriever = TableRetriever(self.catalog, embed_model, table_top_k)
            except Exception as e:  # noqa: BLE001 - embeddings unavailable: fall back to full schema
                self.retrieval_warning = (
                    "Schema retrieval is off (embedding model unavailable: "
                    f"{str(e).splitlines()[0][:160]}). Using the full schema instead.")

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_settings(cls, settings: Settings | None = None, **overrides) -> "NL2SQLEngine":
        from .providers import build_embed_model, build_llm

        s = settings or Settings()
        db = SQLDatabase(create_engine(s.database_url))
        kwargs = dict(
            llm=build_llm(s),
            embed_model=build_embed_model(s),
            table_top_k=s.table_top_k,
            max_retries=s.max_retries,
            row_limit=s.row_limit,
            descriptions_path=s.descriptions_path,
        )
        kwargs.update(overrides)
        return cls(db, **kwargs)

    # ------------------------------------------------------------------ steps
    def select_tables(self, question: str) -> list[str]:
        if self.table_retriever is None:
            return list(self.catalog.tables)
        try:
            return self.table_retriever.retrieve(question)
        except Exception:  # noqa: BLE001 - e.g. embedding rate limit: fall back to full schema
            return list(self.catalog.tables)

    def _generate(self, question: str, schema_text: str, feedback: str = "") -> str:
        prompt = TEXT_TO_SQL_PROMPT.format(
            dialect=self.catalog.dialect.upper(), schema=schema_text,
            question=question, feedback=feedback,
        )
        return extract_sql(self.llm.complete(prompt).text)

    def generate_sql(self, question: str) -> QueryResult:
        """Generate (and, if enabled, validate + self-correct) SQL without executing it."""
        result = QueryResult(question=question)
        result.tables_used = self.select_tables(question)
        schema_text = self.catalog.describe(result.tables_used)

        feedback = ""
        for _ in range(self.max_retries + 1):
            sql = self._generate(question, schema_text, feedback)
            if not self.validate:
                result.attempts.append(Attempt(sql, []))
                result.sql, result.success = sql, True
                return result
            check = self.validator.validate(sql)
            result.attempts.append(Attempt(check.sql or sql, check.errors))
            if check.ok:
                result.sql, result.success = check.sql, True
                return result
            # If a column lives in a table we didn't show the model, widen the schema.
            if any("exists in:" in e or "Unknown table" in e for e in check.errors):
                schema_text = self.catalog.describe()
            feedback = FEEDBACK_TEMPLATE.format(
                sql=sql, errors="\n".join(f"- {e}" for e in check.errors))

        result.sql = result.attempts[-1].sql
        result.error = "; ".join(result.attempts[-1].errors) or "Could not produce valid SQL."
        return result

    def execute(self, sql: str) -> tuple[pd.DataFrame, bool]:
        with self.catalog.engine.connect() as conn:
            cursor = conn.execute(text(sql))
            rows = cursor.fetchmany(self.row_limit + 1)
            df = pd.DataFrame(rows[: self.row_limit], columns=list(cursor.keys()))
        return df, len(rows) > self.row_limit

    def summarize(self, result: QueryResult) -> str:
        rows = result.data.head(20).to_csv(index=False) if result.data is not None else ""
        prompt = ANSWER_PROMPT.format(question=result.question, sql=result.sql, rows=rows or "(empty)")
        return self.llm.complete(prompt).text.strip()

    # ------------------------------------------------------------------ main entry
    def query(self, question: str, summarize: bool = False) -> QueryResult:
        start = time.perf_counter()
        result = self.generate_sql(question)
        if result.success:
            try:
                result.data, result.truncated = self.execute(result.sql)
                if summarize:
                    result.answer = self.summarize(result)
            except Exception as e:  # noqa: BLE001
                result.success = False
                result.error = f"Execution failed: {str(getattr(e, 'orig', e)).splitlines()[0]}"
        result.seconds = round(time.perf_counter() - start, 2)
        return result
