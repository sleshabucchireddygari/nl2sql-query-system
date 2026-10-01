"""Offline tests (no API key needed): a scripted fake LLM stands in for OpenAI."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from llama_index.core import SQLDatabase
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "eval"))

import build_database  # noqa: E402
from nl2sql import NL2SQLEngine, SchemaCatalog, SQLValidator, extract_sql  # noqa: E402
from run_eval import results_match  # noqa: E402


class ScriptedLLM(CustomLLM):
    """Returns canned responses in order and records the prompts it saw."""
    responses: list[str] = []
    prompts: list[str] = []

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="scripted")

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kw: Any) -> CompletionResponse:
        self.prompts.append(prompt)
        return CompletionResponse(text=self.responses.pop(0))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kw: Any):
        raise NotImplementedError


@pytest.fixture(scope="session")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "hospital.db"
    build_database.build(f"sqlite:///{path}")
    return SQLDatabase(create_engine(f"sqlite:///{path}"))


@pytest.fixture(scope="session")
def validator(db):
    return SQLValidator(SchemaCatalog(db, build_database.DESCRIPTIONS_FILE))


def make_engine(db, responses, **kw):
    llm = ScriptedLLM(responses=list(responses), prompts=[])
    eng = NL2SQLEngine(db, llm, MockEmbedding(embed_dim=8),
                       descriptions_path=build_database.DESCRIPTIONS_FILE, **kw)
    return eng, llm


# ------------------------------------------------------------------ validator
@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM patients",
    "SELECT d.name, AVG(a.length_of_stay) AS los FROM admissions a JOIN departments d "
    "ON a.department_id = d.department_id GROUP BY d.name ORDER BY los DESC",
    "WITH t AS (SELECT patient_id, COUNT(*) AS n FROM admissions GROUP BY patient_id) "
    "SELECT AVG(n) FROM t",
    "SELECT * FROM doctors;",
])
def test_valid_queries_pass(validator, sql):
    assert validator.validate(sql).ok


@pytest.mark.parametrize("sql, fragment", [
    ("SELECT * FROM patient", "Unknown table 'patient'"),
    ("SELECT diagnosis FROM admissions", "does not exist"),
    ("SELECT a.icd10_code FROM admissions a", "does not exist in table 'admissions'"),
    ("DELETE FROM patients", "read-only"),
    ("DROP TABLE patients", "read-only"),
    ("SELECT 1; SELECT 2", "exactly one"),
    ("SELEC name FROM departments", "error"),
])
def test_invalid_queries_fail(validator, sql, fragment):
    res = validator.validate(sql)
    assert not res.ok and any(fragment in e for e in res.errors), res.errors


def test_wrong_table_column_gets_join_hint(validator):
    res = validator.validate("SELECT drug_name FROM admissions")
    assert "It exists in: prescriptions" in res.errors[0]


# ------------------------------------------------------------------ engine
def test_extract_sql_variants():
    assert extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("SQLQuery: SELECT 2 SQLResult: x") == "SELECT 2"
    assert extract_sql("Sure! SELECT 3") == "SELECT 3"


def test_self_correction_fixes_bad_column(db):
    eng, llm = make_engine(db, [
        "```sql\nSELECT AVG(stay_length) FROM admissions\n```",       # wrong column
        "```sql\nSELECT AVG(length_of_stay) FROM admissions\n```",    # corrected
    ])
    r = eng.query("average length of stay?")
    assert r.success and r.retries == 1
    assert "stay_length" in llm.prompts[1] and "does not exist" in llm.prompts[1]
    assert r.data.iloc[0, 0] > 0


def test_gives_up_after_max_retries(db):
    eng, _ = make_engine(db, ["SELECT nope FROM admissions"] * 3, max_retries=2)
    r = eng.query("x")
    assert not r.success and len(r.attempts) == 3 and "does not exist" in r.error


def test_unsafe_sql_never_executes(db):
    eng, _ = make_engine(db, ["DELETE FROM patients"], max_retries=0)
    assert not eng.query("delete everyone").success
    with db.engine.connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM patients")).scalar() == 300


def test_prompt_grounds_categorical_values(db):
    eng, llm = make_engine(db, ["SELECT 1"], use_table_retrieval=False)
    eng.generate_sql("How many Medicare patients?")
    assert "'Medicare'" in llm.prompts[0] and "references departments.department_id" in llm.prompts[0]


def test_table_retrieval_includes_fk_neighbours(db):
    eng, _ = make_engine(db, [], table_top_k=1)
    tables = eng.select_tables("Which drugs were prescribed?")
    assert len(tables) >= 2  # one retrieved table + its foreign-key neighbours


def test_row_limit(db):
    eng, _ = make_engine(db, ["SELECT * FROM diagnoses"], row_limit=10)
    r = eng.query("all diagnoses")
    assert len(r.data) == 10 and r.truncated


# ------------------------------------------------------------------ evaluation set
def test_every_gold_query_is_valid_and_runs(db, validator):
    questions = json.loads((ROOT / "eval" / "questions.json").read_text())
    for q in questions:
        assert validator.validate(q["gold_sql"]).ok, q["id"]
        with db.engine.connect() as c:
            assert c.execute(text(q["gold_sql"])).fetchall(), f"gold #{q['id']} returned no rows"


def test_results_match_rules():
    assert results_match([(1.004,)], [(1.0,)], False)
    assert results_match([("Oncology", 7)], [("Oncology",)], False)          # extra column OK
    assert results_match([("b",), ("a",)], [("a",), ("b",)], False)           # order ignored
    assert not results_match([("b",), ("a",)], [("a",), ("b",)], True)        # unless ordered
    assert not results_match([("a",)], [("a",), ("b",)], False)
