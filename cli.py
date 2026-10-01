"""Command-line interface.

    python cli.py "How many patients have Medicare?"
    python cli.py            # interactive mode
"""
import sys

import pandas as pd

from nl2sql import NL2SQLEngine


def show(engine: NL2SQLEngine, question: str) -> None:
    r = engine.query(question, summarize=True)
    print(f"\nSQL ({r.retries} self-corrections, {r.seconds}s):\n  {r.sql}\n")
    if r.success:
        with pd.option_context("display.max_rows", 30, "display.width", 120):
            print(r.data.to_string(index=False))
        print(f"\n{r.answer}\n")
    else:
        print(f"FAILED: {r.error}\n")


def main() -> None:
    engine = NL2SQLEngine.from_settings()
    if len(sys.argv) > 1:
        show(engine, " ".join(sys.argv[1:]))
        return
    print("Ask a question about the database (blank line to quit).")
    while q := input("> ").strip():
        show(engine, q)


if __name__ == "__main__":
    main()
