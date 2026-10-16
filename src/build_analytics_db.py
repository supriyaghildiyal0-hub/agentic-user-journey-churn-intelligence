
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
DB_DIR = ROOT / "data" / "database"

DB_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DB_DIR / "customer_intelligence.duckdb"

events_path = (DATA_DIR / "events_clean.csv.gz").as_posix()
visitors_path = (DATA_DIR / "visitor_features.csv.gz").as_posix()
sessions_path = (DATA_DIR / "session_features.csv.gz").as_posix()

con = duckdb.connect(str(DB_PATH))

# Load cleaned events
con.execute(f"""
CREATE OR REPLACE TABLE events AS
SELECT *
FROM read_csv_auto('{events_path}')
""")

# Load visitor-level features
con.execute(f"""
CREATE OR REPLACE TABLE visitor_features AS
SELECT *
FROM read_csv_auto('{visitors_path}')
""")

# Load session-level features
con.execute(f"""
CREATE OR REPLACE TABLE session_features AS
SELECT *
FROM read_csv_auto('{sessions_path}')
""")

print("\n========== DATABASE SUMMARY ==========")

for table in ["events", "visitor_features", "session_features"]:
    count = con.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

    print(f"{table}: {count:,} rows")

print("\n========== EVENT DISTRIBUTION ==========")

result = con.execute("""
SELECT
    event,
    COUNT(*) AS event_count,
    COUNT(DISTINCT visitorid) AS unique_visitors
FROM events
GROUP BY event
ORDER BY event_count DESC
""").df()

print(result.to_string(index=False))

print("\n========== DATABASE COMPLETE ==========")
print("Database saved to:", DB_PATH)

con.close()
