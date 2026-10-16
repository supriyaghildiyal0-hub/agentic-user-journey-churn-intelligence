
-- Query 1: Overall customer behavior
SELECT
    COUNT(DISTINCT visitorid) AS total_visitors,
    COUNT(DISTINCT CASE WHEN event = 'view'
        THEN visitorid END) AS viewers,
    COUNT(DISTINCT CASE WHEN event = 'addtocart'
        THEN visitorid END) AS cart_users,
    COUNT(DISTINCT CASE WHEN event = 'transaction'
        THEN visitorid END) AS purchasers
FROM events;


-- Query 2: Visitors who viewed products but never added to cart
WITH visitor_behavior AS (
    SELECT
        visitorid,
        BOOL_OR(event = 'view') AS viewed,
        BOOL_OR(event = 'addtocart') AS added_to_cart,
        BOOL_OR(event = 'transaction') AS purchased
    FROM events
    GROUP BY visitorid
)
SELECT
    COUNT(*) AS viewers_without_cart
FROM visitor_behavior
WHERE viewed
  AND NOT added_to_cart;


-- Query 3: Visitors who added to cart but never purchased
WITH visitor_behavior AS (
    SELECT
        visitorid,
        BOOL_OR(event = 'addtocart') AS added_to_cart,
        BOOL_OR(event = 'transaction') AS purchased
    FROM events
    GROUP BY visitorid
)
SELECT
    COUNT(*) AS cart_users_without_purchase
FROM visitor_behavior
WHERE added_to_cart
  AND NOT purchased;


-- Query 4: Visitors with multiple recorded sessions
SELECT
    CASE
        WHEN session_count = 1 THEN 'Single-session visitor'
        ELSE 'Returning visitor'
    END AS visitor_segment,
    COUNT(*) AS visitors
FROM (
    SELECT
        visitorid,
        COUNT(DISTINCT session_id) AS session_count
    FROM session_features
    GROUP BY visitorid
) AS visitor_sessions
GROUP BY visitor_segment
ORDER BY visitors DESC;


-- Query 5: Visitors with multiple distinct transactions
WITH visitor_transactions AS (
    SELECT
        visitorid,
        COUNT(DISTINCT transactionid) AS transaction_count
    FROM events
    WHERE event = 'transaction'
      AND transactionid IS NOT NULL
    GROUP BY visitorid
)
SELECT
    COUNT(*) AS visitors_with_multiple_transactions
FROM visitor_transactions
WHERE transaction_count > 1;


-- Query 6: Event activity by day
SELECT
    CAST(event_datetime AS DATE) AS event_date,
    event,
    COUNT(*) AS total_events,
    COUNT(DISTINCT visitorid) AS unique_visitors
FROM events
GROUP BY event_date, event
ORDER BY event_date, event;
