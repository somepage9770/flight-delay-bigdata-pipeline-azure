"""
Phase 6 — Load curated data into PostgreSQL (Azure version)

Same logic as the original SQLTablesCreation.py, but:
  1. Points at Azure Database for PostgreSQL Flexible Server instead
     of a local instance.
  2. Reads connection details from environment variables instead of
     hardcoding them — a hardcoded password was fine for a local
     academic project, but never for anything pointed at the cloud.

Set these before running:
  AZURE_PG_HOST      e.g. flight-delay-db.postgres.database.azure.com
  AZURE_PG_USER       e.g. pgadmin
  AZURE_PG_PASSWORD
  AZURE_PG_DB         e.g. flight_project_db   (defaults to this if unset)
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text

DB_HOST = os.environ["AZURE_PG_HOST"]
DB_USER = os.environ["AZURE_PG_USER"]
DB_PASSWORD = os.environ["AZURE_PG_PASSWORD"]
DB_PORT = os.environ.get("AZURE_PG_PORT", "5432")
DB_NAME = os.environ.get("AZURE_PG_DB", "flight_project_db")

# Azure Database for PostgreSQL Flexible Server requires SSL by default
SSL_MODE = "require"

# 1. Create the database if it doesn't exist
print("Connecting to Azure Database for PostgreSQL...\n")
engine_default = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres?sslmode={SSL_MODE}"
)

with engine_default.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
    result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'"))
    if not result.fetchone():
        conn.execute(text(f"CREATE DATABASE {DB_NAME}"))
        print(f" Database '{DB_NAME}' created!\n")
    else:
        print(f" Database '{DB_NAME}' already exists.\n")

engine_default.dispose()

# 2. Connect to the project database
engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode={SSL_MODE}"
)

# 3. Curated files downloaded from ADLS Gen2 -> table name mapping
archivos_curated = {
    "curated_month_risk_profile_part-00000-23215731-fb95-4fd7-b71c-e24ee7c40cc4-c000.csv": "month_risk_profile",
    "curated_hour_risk_profile_part-00000-da147d9d-c1ad-4758-a683-53352c39fcc7-c000.csv": "hour_risk_profile",
    "curated_route_risk_profile_part-00000-811a7917-f911-4bf4-9dbb-4acda56dee62-c000.csv": "route_risk_profile",
    "curated_airline_risk_profile_part-00000-9d8fec7f-bfad-4d09-aefd-4914f2191f07-c000.csv": "airline_risk_profile",
}

print("Starting bulk load into PostgreSQL...\n")

# 4. Read each file and load it as a table
for archivo, nombre_tabla in archivos_curated.items():
    print(f"Reading data from: {nombre_tabla}...")
    df = pd.read_csv(archivo)
    df.to_sql(nombre_tabla, engine, if_exists="replace", index=False)
    print(f" Table '{nombre_tabla}' created successfully!\n")

# 5. Run sample queries
print("=" * 60)
print("SQL QUERIES")
print("=" * 60)

with engine.connect() as conn:
    print("\n Top 10 airlines with highest delay rate:\n")
    result = pd.read_sql(
        text("SELECT * FROM airline_risk_profile ORDER BY airline_delay_rate DESC LIMIT 10"),
        conn,
    )
    print(result.to_string(index=False))

    print("\n\n Top 10 hours with highest delay rate:\n")
    result = pd.read_sql(
        text("SELECT * FROM hour_risk_profile ORDER BY hour_delay_rate DESC LIMIT 10"),
        conn,
    )
    print(result.to_string(index=False))

engine.dispose()
print("\n\nDone! All your tables are in the database.")
