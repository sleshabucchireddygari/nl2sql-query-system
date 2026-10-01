# Natural Language to SQL Query System

Ask questions about a relational database in plain English and get back validated, executable SQL and the results.
Built with **LlamaIndex**, **MySQL** (with a zero-setup SQLite fallback), **Python**, and a **Streamlit** UI.
Runs **free** on Google Gemini's free tier (default) or a local Ollama model; OpenAI is supported too.

```
"Which department had the highest mortality rate?"
        │
        ▼
┌─────────────────────┐   LlamaIndex ObjectIndex retrieves the relevant tables
│ 1. Schema grounding │   (+ foreign-key neighbours) and renders the real schema:
└─────────────────────┘   columns, types, keys, descriptions, categorical values
        │
        ▼
┌─────────────────────┐
│ 2. SQL generation   │   LlamaIndex LLM (Gemini / Ollama / OpenAI) + prompt
└─────────────────────┘
        │
        ▼
┌─────────────────────┐   sqlglot: single read-only SELECT, tables & columns exist
│ 3. Validation       │   EXPLAIN dry run on the database
└─────────────────────┘
        │ errors? ──► 4. Self-correction: errors fed back to the LLM (up to N retries)
        ▼
┌─────────────────────┐
│ 5. Execution        │   row-limited → pandas DataFrame → plain-English answer
└─────────────────────┘
```

## Features

- **Schema grounding with LlamaIndex** – `ObjectIndex` + `SQLTableNodeMapping` retrieve only relevant tables; the prompt lists real column names, PK/FK relationships, business descriptions and the actual values of categorical columns (e.g. `insurance_type: 'Medicaid', 'Medicare', 'Private', 'Uninsured'`). This cuts invalid table/column references and wrong literal values.
- **SQL validation** – every query is parsed with `sqlglot` for the target dialect; non-SELECT statements are blocked; every table and column is checked against the live schema; then an `EXPLAIN` dry run catches anything else.
- **Self-correction loop** – validation errors are written to be LLM-actionable (*"Column 'drug_name' does not exist in admissions. It exists in: prescriptions (join that table)"*) and fed back for a retry.
- **Safe execution** – read-only, row-limited, never runs unvalidated SQL.
- **Evaluation harness** – 25 hand-written questions (easy/medium/hard) with gold SQL; reports valid-SQL rate, execution accuracy, schema-error count and retries, with an ablation of baseline vs. grounded+validated.
- **Streamlit app** – shows generated SQL, tables used, validation history, results table, auto bar chart, CSV download.

## Demo database

A synthetic hospital database (deterministic, seed 42), 6 related tables:

| table | rows | description |
|---|---|---|
| departments | 8 | clinical departments, floor, bed capacity |
| doctors | 25 | physicians, specialty, department |
| patients | 300 | demographics, city/state, insurance type |
| admissions | 900 | stays: dates, length of stay, type, outcome, charges, 30-day readmission |
| diagnoses | ~1,800 | ICD-10 codes per admission (primary/secondary) |
| prescriptions | ~2,250 | drugs, dose, days supply, cost |

## Choosing an AI provider (free options)

| `LLM_PROVIDER` | cost | what you need |
|---|---|---|
| `gemini` (default) | free tier, no credit card | a key from https://aistudio.google.com/apikey → `GOOGLE_API_KEY` |
| `ollama` | free, runs on your computer | install https://ollama.com, then `ollama pull qwen2.5-coder:7b` and `ollama pull nomic-embed-text` |
| `openai` | paid | `OPENAI_API_KEY` with credit |

The Gemini free tier allows roughly 10 requests per minute, so the evaluation script pauses 7 s between questions. If schema retrieval embeddings are unavailable, the engine automatically falls back to the full schema.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # then put your free GOOGLE_API_KEY in .env
python data/build_database.py                            # creates data/hospital.db (SQLite)
```

### Using MySQL (as in production)

```bash
mysql -u root -p -e "CREATE DATABASE hospital;"
python data/build_database.py --url "mysql+pymysql://root:PASSWORD@localhost:3306/hospital"
# then in .env:
DATABASE_URL=mysql+pymysql://root:PASSWORD@localhost:3306/hospital
```

Point `DATABASE_URL` at any other MySQL database to query your own data (the descriptions file is optional).

## Run

```bash
streamlit run app.py                                    # web UI at http://localhost:8501
python cli.py "What are the top 5 most prescribed drugs?"
python cli.py                                           # interactive
python eval/run_eval.py                                 # evaluation + ablation
pytest -q                                               # offline tests, no API key needed
```

## Use it from Python

```python
from nl2sql import NL2SQLEngine

engine = NL2SQLEngine.from_settings()
r = engine.query("What is the 30-day readmission rate by admission type?", summarize=True)
print(r.sql)          # validated SQL
print(r.data)         # pandas DataFrame
print(r.answer)       # plain-English answer
print(r.retries)      # self-corrections used
```

## Project structure

```
nl2sql/
  config.py      settings from .env
  schema.py      SchemaCatalog (schema description) + TableRetriever (LlamaIndex ObjectIndex)
  validator.py   sqlglot static checks + EXPLAIN dry run
  engine.py      generate → validate → self-correct → execute → summarise
data/
  build_database.py          synthetic hospital DB for SQLite or MySQL
  schema_descriptions.json   table/column business descriptions
eval/
  questions.json   25 NL questions with gold SQL
  run_eval.py      execution-accuracy evaluation + ablation
tests/             pytest suite using a scripted fake LLM
app.py             Streamlit UI
cli.py             command-line interface
```

## Evaluation

`python eval/run_eval.py` compares two configurations on the 25-question set:

- **baseline** – full schema in the prompt, no validation or retries
- **grounded** – LlamaIndex table retrieval + validation + self-correction

It prints valid-SQL rate, execution accuracy (result-set match against gold SQL, column order ignored), number of attempts that referenced a non-existent table/column, and average retries, and writes per-question details to `eval/results.json`. Record your numbers here after running it.

| config | valid SQL | execution accuracy | schema-error attempts | avg retries |
|---|---|---|---|---|
| baseline | – | – | – | – |
| grounded | – | – | – | – |
