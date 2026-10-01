"""Static + dry-run validation of LLM-generated SQL.

Checks, in order:
1. Parses as exactly one statement in the target dialect (sqlglot).
2. Read-only: a SELECT / UNION / CTE, with no INSERT/UPDATE/DELETE/DDL anywhere.
3. Every table exists in the database.
4. Every column exists (in the table its qualifier points to, or in some
   referenced table when unqualified). Select-list aliases and CTE/subquery
   columns are allowed.
5. The database accepts the query plan (EXPLAIN), catching anything left.

Returned error messages are written to be fed straight back to the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlalchemy import text

from .schema import SchemaCatalog

FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
             exp.Merge, exp.Command, exp.TruncateTable)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    sql: str = ""  # normalised SQL (single statement, no trailing semicolon)


class SQLValidator:
    def __init__(self, catalog: SchemaCatalog):
        self.catalog = catalog
        self.dialect = "mysql" if catalog.dialect.startswith("mysql") else catalog.dialect

    def validate(self, sql: str, dry_run: bool = True) -> ValidationResult:
        sql = sql.strip().rstrip(";").strip()
        if not sql:
            return ValidationResult(False, ["The model returned no SQL."])

        # 1. parse ------------------------------------------------------------
        try:
            statements = [s for s in sqlglot.parse(sql, read=self.dialect) if s is not None]
        except sqlglot.errors.ParseError as e:
            return ValidationResult(False, [f"SQL syntax error: {str(e).splitlines()[0]}"], sql)
        if len(statements) != 1:
            return ValidationResult(False, ["Return exactly one SQL statement."], sql)
        tree = statements[0]

        # 2. read-only ---------------------------------------------------------
        if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)) or \
                any(tree.find_all(*FORBIDDEN)):
            return ValidationResult(False, ["Only read-only SELECT queries are allowed."], sql)

        errors = self._check_tables_and_columns(tree)
        if errors:
            return ValidationResult(False, errors, sql)

        # 5. dry run -----------------------------------------------------------
        if dry_run:
            try:
                with self.catalog.engine.connect() as conn:
                    conn.execute(text(f"EXPLAIN {sql}"))
            except Exception as e:  # noqa: BLE001 - surface any DB error to the LLM
                msg = str(getattr(e, "orig", e)).splitlines()[0]
                return ValidationResult(False, [f"Database rejected the query: {msg}"], sql)
        return ValidationResult(True, [], sql)

    # ------------------------------------------------------------------------
    def _check_tables_and_columns(self, tree: exp.Expression) -> list[str]:
        cat = self.catalog
        known_tables = {t.lower(): t for t in cat.tables}
        cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
        derived_aliases = {s.alias.lower() for s in tree.find_all(exp.Subquery) if s.alias}
        select_aliases = {a.alias.lower() for a in tree.find_all(exp.Alias)}

        errors: list[str] = []
        alias_to_table: dict[str, str] = {}
        referenced: set[str] = set()

        for tbl in tree.find_all(exp.Table):
            name = tbl.name.lower()
            if name in cte_names:
                continue
            if name not in known_tables:
                errors.append(f"Unknown table '{tbl.name}'. Available tables: {', '.join(cat.tables)}.")
                continue
            real = known_tables[name]
            referenced.add(real)
            alias_to_table[tbl.alias_or_name.lower()] = real

        for col in tree.find_all(exp.Column):
            if isinstance(col.this, exp.Star):
                continue
            cname, qual = col.name.lower(), col.table.lower()
            if qual:
                if qual in cte_names or qual in derived_aliases:
                    continue
                table = alias_to_table.get(qual)
                if table is None:
                    errors.append(f"Column '{col.sql()}' uses unknown table or alias '{col.table}'.")
                elif cname not in {c.lower() for c in cat.columns[table]}:
                    errors.append(
                        f"Column '{col.name}' does not exist in table '{table}'. "
                        f"Its columns are: {', '.join(cat.columns[table])}.")
            else:
                pool = referenced or set(cat.tables)
                if any(cname in {c.lower() for c in cat.columns[t]} for t in pool):
                    continue
                if cname in select_aliases:
                    continue  # reference to a select-list alias (e.g. ORDER BY total)
                if cte_names or derived_aliases:
                    continue  # may be a column produced by a CTE/subquery; EXPLAIN will catch it
                owners = [t for t in cat.tables if cname in {c.lower() for c in cat.columns[t]}]
                hint = f" It exists in: {', '.join(owners)} (join that table)." if owners else ""
                errors.append(f"Column '{col.name}' does not exist in the referenced tables.{hint}")

        return list(dict.fromkeys(errors))  # de-duplicate, keep order
