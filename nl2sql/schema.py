"""Schema grounding.

Two jobs:
1. Describe the *real* database schema to the LLM (columns, types, keys,
   business descriptions, and the actual values of categorical columns), so
   generated SQL references columns and literals that exist.
2. Pick only the tables relevant to a question, using a LlamaIndex
   ObjectIndex (vector retrieval over table descriptions). Foreign-key
   neighbours are added so joins stay possible.
"""
from __future__ import annotations

import json
from pathlib import Path

from llama_index.core import SQLDatabase, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.objects import ObjectIndex, SQLTableNodeMapping, SQLTableSchema
from sqlalchemy import inspect, text

# Text columns with at most this many distinct values get their values listed.
CATEGORICAL_MAX_DISTINCT = 12


class SchemaCatalog:
    """Snapshot of the database schema used for prompting and validation."""

    def __init__(self, sql_database: SQLDatabase, descriptions_path: Path | None = None):
        self.sql_database = sql_database
        self.engine = sql_database.engine
        self.dialect = self.engine.dialect.name  # "sqlite" | "mysql" | ...
        self.descriptions = self._load_descriptions(descriptions_path)
        insp = inspect(self.engine)

        self.tables: list[str] = sorted(sql_database.get_usable_table_names())
        self.columns: dict[str, dict[str, str]] = {}
        self.primary_keys: dict[str, list[str]] = {}
        self.foreign_keys: dict[str, list[tuple[str, str, str]]] = {}
        for t in self.tables:
            self.columns[t] = {c["name"]: str(c["type"]) for c in insp.get_columns(t)}
            self.primary_keys[t] = insp.get_pk_constraint(t).get("constrained_columns", [])
            self.foreign_keys[t] = [
                (fk["constrained_columns"][0], fk["referred_table"], fk["referred_columns"][0])
                for fk in insp.get_foreign_keys(t)
            ]
        self.categorical_values = self._sample_categoricals()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _load_descriptions(path: Path | None) -> dict:
        if path and Path(path).exists():
            return json.loads(Path(path).read_text())
        return {}

    def _sample_categoricals(self) -> dict[tuple[str, str], list[str]]:
        found: dict[tuple[str, str], list[str]] = {}
        q = self.engine.dialect.identifier_preparer.quote
        with self.engine.connect() as conn:
            for t, cols in self.columns.items():
                for c, typ in cols.items():
                    if not any(k in typ.upper() for k in ("CHAR", "TEXT", "STRING")):
                        continue
                    sql = (f"SELECT DISTINCT {q(c)} FROM {q(t)} "
                           f"WHERE {q(c)} IS NOT NULL LIMIT {CATEGORICAL_MAX_DISTINCT + 1}")
                    vals = [r[0] for r in conn.execute(text(sql))]
                    if 0 < len(vals) <= CATEGORICAL_MAX_DISTINCT:
                        found[(t, c)] = sorted(map(str, vals))
        return found

    def table_description(self, table: str) -> str:
        return self.descriptions.get(table, {}).get("description", "")

    def all_columns(self) -> set[str]:
        return {c for cols in self.columns.values() for c in cols}

    def fk_neighbours(self, tables: list[str]) -> list[str]:
        """Tables directly linked to `tables` through a foreign key (either direction)."""
        linked = set()
        for t in tables:
            linked.update(ref for _, ref, _ in self.foreign_keys.get(t, []))
        for t, fks in self.foreign_keys.items():
            if any(ref in tables for _, ref, _ in fks):
                linked.add(t)
        return sorted(linked - set(tables))

    # ------------------------------------------------------------------ prompt text
    def describe(self, tables: list[str] | None = None) -> str:
        """Render a compact, LLM-friendly description of the chosen tables."""
        blocks = []
        for t in tables or self.tables:
            col_docs = self.descriptions.get(t, {}).get("columns", {})
            lines = [f"TABLE {t}" + (f"  -- {self.table_description(t)}" if self.table_description(t) else "")]
            for c, typ in self.columns[t].items():
                notes = []
                if c in self.primary_keys[t]:
                    notes.append("primary key")
                for col, ref_t, ref_c in self.foreign_keys[t]:
                    if col == c:
                        notes.append(f"references {ref_t}.{ref_c}")
                if c in col_docs:
                    notes.append(col_docs[c])
                if (t, c) in self.categorical_values:
                    vals = ", ".join(f"'{v}'" for v in self.categorical_values[(t, c)])
                    notes.append(f"values: {vals}")
                lines.append(f"  {c} {typ}" + (f"  -- {'; '.join(notes)}" if notes else ""))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


class TableRetriever:
    """Vector retrieval of relevant tables with LlamaIndex's ObjectIndex."""

    def __init__(self, catalog: SchemaCatalog, embed_model: BaseEmbedding, top_k: int = 4):
        self.catalog = catalog
        self.top_k = top_k
        schemas = [
            SQLTableSchema(
                table_name=t,
                context_str=f"{catalog.table_description(t)} Columns: {', '.join(catalog.columns[t])}",
            )
            for t in catalog.tables
        ]
        self.index = ObjectIndex.from_objects(
            schemas,
            SQLTableNodeMapping(catalog.sql_database),
            VectorStoreIndex,
            embed_model=embed_model,
        )
        self.retriever = self.index.as_retriever(similarity_top_k=top_k)

    def retrieve(self, question: str) -> list[str]:
        hits = [s.table_name for s in self.retriever.retrieve(question)]
        # Keep join paths intact: add tables one FK-hop away from the hits.
        return hits + self.catalog.fk_neighbours(hits)
