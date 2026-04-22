import os
from sqlalchemy import create_engine, text
import pandas as pd

def get_engine():
    server   = os.getenv("DB_SERVER", "localhost")
    port     = os.getenv("DB_PORT", "1433")
    name     = os.getenv("DB_NAME", "gnnbind")
    user     = os.getenv("DB_USER", "sa")
    password = os.getenv("DB_PASSWORD", "GnnBind0ptimizer!")
    conn_str = (
        f"mssql+pyodbc://{user}:{password}@{server}:{port}/{name}"
        "?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
    )
    return create_engine(conn_str, fast_executemany=True)

def query_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})
