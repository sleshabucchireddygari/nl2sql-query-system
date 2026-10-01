"""Evaluate the NL->SQL engine against hand-written gold SQL.

Metrics (per configuration):
  * Valid SQL rate      - generated query passed validation and executed
  * Execution accuracy  - result set matches the gold query's result set
  * Schema errors       - attempts that referenced a non-existent table/column
  * Avg retries         - self-correction rounds used per question

Configurations (ablation):
  baseline   : full schema in prompt, no validation / retries
  grounded   : + LlamaIndex table retrieval and validation with self-correction

Usage:
    python eval/run_eval.py                      # both configs
    python eval/run_eval.py --configs grounded   # one config
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

from sqlalchemy import text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nl2sql import NL2SQLEngine, Settings  # noqa: E402

CONFIGS = {
    "baseline": dict(use_table_retrieval=False, validate=False),
    "grounded": dict(use_table_retrieval=True, validate=True),
}


def _norm(v):
    if isinstance(v, (int, float, Decimal)) and not isinstance(v, bool):
        return round(float(v), 2)
    return None if v is None else str(v)


def results_match(pred, gold, ordered: bool) -> bool:
    """Execution match on values (column names/order ignored).

    Exact match, or a lenient match where the prediction has the same rows but
    extra columns (e.g. it shows the count next to the department name).
    """
    p = [Counter(_norm(v) for v in r) for r in pred]
    g = [Counter(_norm(v) for v in r) for r in gold]
    if len(p) != len(g):
        return False
    if not ordered:
        key = lambda c: repr(sorted(c.items(), key=repr))  # noqa: E731
        p, g = sorted(p, key=key), sorted(g, key=key)
        if p == g:
            return True
        # lenient path: try a greedy multiset match of gold rows inside predicted rows
        remaining = list(p)
        for grow in g:
            hit = next((i for i, prow in enumerate(remaining) if not (grow - prow)), None)
            if hit is None:
                return False
            remaining.pop(hit)
        return True
    return all(not (grow - prow) for prow, grow in zip(p, g))


def run_sql(engine, sql):
    with engine.connect() as conn:
        return conn.execute(text(sql)).fetchall()


TRANSIENT = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded", "high demand", "rate")


def with_retry(fn, *args, attempts: int = 6, first_wait: float = 15.0):
    """Call fn, waiting and retrying when the free-tier API is busy (503/429)."""
    wait = first_wait
    for i in range(attempts):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001
            if i == attempts - 1 or not any(t.lower() in str(e).lower() for t in TRANSIENT):
                raise
            print(f"      API busy ({str(e).splitlines()[0][:60]}...). Waiting {wait:.0f}s and retrying.")
            time.sleep(wait)
            wait = min(wait * 2, 120)


def evaluate(engine: NL2SQLEngine, questions: list[dict], pause: float = 0.0) -> dict:
    db = engine.catalog.engine
    records = []
    for i, q in enumerate(questions):
        if i and pause:
            time.sleep(pause)  # stay under free-tier requests-per-minute limits
        try:
            res = with_retry(engine.generate_sql, q["question"])
        except Exception as e:  # noqa: BLE001 - API kept failing: record and move on
            msg = str(e).splitlines()[0][:200]
            records.append(dict(id=q["id"], difficulty=q["difficulty"], question=q["question"],
                                sql="", executed=False, correct=False, retries=0,
                                schema_errors=0, error=f"API error: {msg}", skipped=True))
            print(f"  [SKIP] #{q['id']:>2} API unavailable, skipped")
            continue
        schema_errs = sum(
            any(k in e for e in a.errors for k in ("does not exist", "Unknown table", "unknown table"))
            for a in res.attempts
        )
        correct, exec_ok, err = False, False, res.error
        try:
            pred = run_sql(db, res.sql)
            exec_ok = True
            gold = run_sql(db, q["gold_sql"])
            correct = results_match(pred, gold, q.get("ordered", False))
        except Exception as e:  # noqa: BLE001
            err = str(getattr(e, "orig", e)).splitlines()[0]
        records.append(dict(id=q["id"], difficulty=q["difficulty"], question=q["question"],
                            sql=res.sql, executed=exec_ok, correct=correct,
                            retries=res.retries, schema_errors=schema_errs, error=err, skipped=False))
        mark = "OK " if correct else ("EXE" if exec_ok else "ERR")
        print(f"  [{mark}] #{q['id']:>2} ({q['difficulty']}) retries={res.retries}  {q['question'][:70]}")

    done = [r for r in records if not r["skipped"]]  # score only questions the API answered
    n = max(1, len(done))
    summary = {
        "questions": len(records),
        "answered": len(done),
        "skipped_api_errors": len(records) - len(done),
        "valid_sql_rate": round(sum(r["executed"] for r in done) / n, 3),
        "execution_accuracy": round(sum(r["correct"] for r in done) / n, 3),
        "schema_error_attempts": sum(r["schema_errors"] for r in done),
        "avg_retries": round(sum(r["retries"] for r in done) / n, 2),
        "by_difficulty": {
            d: round(sum(r["correct"] for r in done if r["difficulty"] == d)
                     / max(1, sum(r["difficulty"] == d for r in done)), 3)
            for d in ("easy", "medium", "hard")
        },
    }
    return {"summary": summary, "records": records}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--questions", default=str(ROOT / "eval" / "questions.json"))
    p.add_argument("--pause", type=float, default=None,
                   help="Seconds to wait between questions (default: 7 for gemini, else 0)")
    p.add_argument("--out", default=str(ROOT / "eval" / "results.json"))
    args = p.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    settings = Settings()
    pause = args.pause if args.pause is not None else (7.0 if settings.llm_provider == "gemini" else 0.0)
    results = {}
    for name in args.configs:
        print(f"\n=== {name} ===")
        engine = with_retry(lambda: NL2SQLEngine.from_settings(settings, **CONFIGS[name]))
        results[name] = evaluate(engine, questions, pause)

    print("\n{:<10} {:>9} {:>10} {:>10} {:>14} {:>8}".format(
        "config", "answered", "valid SQL", "exec acc", "schema errors", "retries"))
    for name, r in results.items():
        s = r["summary"]
        print("{:<10} {:>9} {:>10.1%} {:>10.1%} {:>14} {:>8}".format(
            name, f"{s['answered']}/{s['questions']}", s["valid_sql_rate"], s["execution_accuracy"],
            s["schema_error_attempts"], s["avg_retries"]))
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nDetailed results written to {args.out}")


if __name__ == "__main__":
    main()
