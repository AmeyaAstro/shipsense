-- Order-level feature table for late-delivery risk.
--
-- The raw DataCo file has one row per ORDER ITEM. The label (Late_delivery_risk),
-- shipping mode and payment type are identical for every item in an order, so we
-- model at the ORDER level: one row per order. That also prevents the same order
-- landing in both train and test (a silent form of leakage).
--
-- Only columns known AT ORDER TIME are kept. These are deliberately excluded
-- because they are only known after shipping, or they define the label:
--   days_for_shipping_real, delivery_status, shipping_date, order_status,
--   benefit/profit columns (computed after fulfilment).
--
-- Canceled and suspected-fraud orders never ship (always label 0), so they are
-- removed: the model answers "will this order arrive late, given it ships?".

DROP TABLE IF EXISTS order_features;

CREATE TABLE order_features AS
SELECT
    order_id,
    MIN(order_ts)                                    AS order_ts,
    MIN(shipping_mode)                               AS shipping_mode,
    MIN(days_scheduled)                              AS days_scheduled,
    MIN(payment_type)                                AS payment_type,
    MIN(market)                                      AS market,
    MIN(order_region)                                AS order_region,
    MIN(order_country)                               AS order_country,
    MIN(customer_segment)                            AS customer_segment,
    COUNT(*)                                         AS n_items,
    SUM(quantity)                                    AS total_quantity,
    SUM(sales)                                       AS total_sales,
    AVG(discount_rate)                               AS avg_discount_rate,
    COUNT(DISTINCT department_name)                  AS n_departments,
    -- department of the most expensive line item (ties broken alphabetically)
    (SELECT i2.department_name FROM order_items i2
      WHERE i2.order_id = i.order_id
      ORDER BY i2.sales DESC, i2.department_name LIMIT 1) AS main_department,
    MAX(late_delivery_risk)                          AS late
FROM order_items i
WHERE order_status NOT IN ('CANCELED', 'SUSPECTED_FRAUD')
GROUP BY order_id;
