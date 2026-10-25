"""Query every dataset in the project with SQL (DuckDB, no server needed).

  python -m sei.sql                      interactive SQL prompt
  python -m sei.sql "SELECT ..."         run one query
  python -m sei.sql --tables             list tables and their columns
  python -m sei.sql --file queries/examples.sql   run a .sql file (queries separated by ';')

In Python / notebooks:
  from sei.sql import connect, q
  q("SELECT arm, COUNT(*) FROM users GROUP BY arm")      # -> pandas DataFrame
"""
from __future__ import annotations

import sys

import duckdb
import pandas as pd

from . import config as C

_CON: duckdb.DuckDBPyConnection | None = None


def connect(refresh: bool = False) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with one view per project file (event logs, user tables, result tables, Power BI tables).

    Views read the files lazily, so rebuilt data is picked up automatically; pass refresh=True to also pick up
    files that did not exist when the connection was first created.
    """
    global _CON
    if _CON is not None and not refresh:
        return _CON
    con = duckdb.connect()
    sources = [(p.stem, f"read_parquet('{p}')") for d in (C.RAW, C.PROCESSED) for p in sorted(d.glob("*.parquet"))]
    sources += [(p.stem, f"read_csv_auto('{p}')") for p in sorted(C.TABLES.glob("*.csv"))]
    sources += [(f"pbi_{p.stem}", f"read_csv_auto('{p}')") for p in sorted((C.POWERBI / "data").glob("*.csv"))]
    for name, reader in sources:
        con.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM {reader}')
    _CON = con
    return con


def q(sql: str, refresh: bool = False) -> pd.DataFrame:
    return connect(refresh).execute(sql).df()


def tables() -> pd.DataFrame:
    return q("SELECT table_name, COUNT(*) AS n_columns, string_agg(column_name, ', ') AS columns "
             "FROM information_schema.columns GROUP BY table_name ORDER BY table_name")


def _show(df: pd.DataFrame):
    with pd.option_context("display.max_rows", 200, "display.max_columns", 30, "display.width", 200,
                           "display.float_format", lambda x: f"{x:,.4f}".rstrip("0").rstrip(".")):
        print(df if len(df) > 200 else df.to_string(index=False))


def main():
    args = sys.argv[1:]
    if args[:1] == ["--tables"]:
        _show(tables()[["table_name", "n_columns"]])
        return
    if args[:1] == ["--file"]:
        title, body = "", []
        for line in open(args[1]).read().splitlines() + [";"]:
            if line.strip().startswith("--"):
                title = line.strip("- ").strip() or title
                continue
            body.append(line)
            if line.rstrip().endswith(";"):
                stmt = "\n".join(body).strip().rstrip(";").strip()
                if stmt:
                    print(f"\n>>> {title}")
                    _show(q(stmt))
                title, body = "", []
        return
    if args:
        _show(q(" ".join(args)))
        return
    print("SQL prompt (DuckDB). End queries with ';'.  Commands: .tables  .quit")
    buf = []
    while True:
        try:
            line = input("sql> " if not buf else "...> ")
        except EOFError:
            break
        if line.strip() in (".quit", ".exit"):
            break
        if line.strip() == ".tables":
            _show(tables()[["table_name", "n_columns"]])
            continue
        buf.append(line)
        if line.rstrip().endswith(";"):
            try:
                _show(q("\n".join(buf).rstrip().rstrip(";")))
            except duckdb.Error as e:
                print(f"error: {e}")
            buf = []


if __name__ == "__main__":
    main()
