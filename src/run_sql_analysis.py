
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "database" / "customer_intelligence.duckdb"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(DB_PATH), read_only=True)

queries = {
    "overall_customer_behavior": """
        SELECT
            COUNT(DISTINCT visitorid) AS total_visitors,
            COUNT(DISTINCT CASE WHEN event = 'view'
                THEN visitorid END) AS viewers,
            COUNT(DISTINCT CASE WHEN event = 'addtocart'
                THEN visitorid END) AS cart_users,
            COUNT(DISTINCT CASE WHEN event = 'transaction'
                THEN visitorid END) AS purchasers
        FROM events
    """,

    "viewers_without_cart": """
        WITH behavior AS (
            SELECT
                visitorid,
                BOOL_OR(event = 'view') AS viewed,
                BOOL_OR(event = 'addtocart') AS carted
            FROM events
            GROUP BY visitorid
        )
        SELECT COUNT(*) AS viewers_without_cart
        FROM behavior
        WHERE viewed AND NOT carted
    """,

    "cart_users_without_purchase": """
        WITH behavior AS (
            SELECT
                visitorid,
                BOOL_OR(event = 'addtocart') AS carted,
                BOOL_OR(event = 'transaction') AS purchased
            FROM events
            GROUP BY visitorid
        )
        SELECT COUNT(*) AS cart_users_without_purchase
        FROM behavior
        WHERE carted AND NOT purchased
    """,

    "visitor_session_segments": """
        WITH sessions AS (
            SELECT visitorid, COUNT(DISTINCT session_id) AS session_count
            FROM session_features
            GROUP BY visitorid
        )
        SELECT
            CASE WHEN session_count = 1
                 THEN 'Single-session visitor'
                 ELSE 'Returning visitor'
            END AS visitor_segment,
            COUNT(*) AS visitors
        FROM sessions
        GROUP BY visitor_segment
        ORDER BY visitors DESC
    """,

    "repeat_transaction_visitors": """
        SELECT COUNT(*) AS repeat_transaction_visitors
        FROM (
            SELECT visitorid
            FROM events
            WHERE event = 'transaction'
              AND transactionid IS NOT NULL
            GROUP BY visitorid
            HAVING COUNT(DISTINCT transactionid) > 1
        ) AS repeat_buyers
    """
}

for name, query in queries.items():
    print(f"\n========== {name.upper()} ==========")

    result = con.execute(query).df()
    print(result.to_string(index=False))

    output_path = OUTPUT_DIR / f"{name}.csv"
    result.to_csv(output_path, index=False)
    print("Saved:", output_path)

con.close()
print("\nSQL analysis completed successfully.")
