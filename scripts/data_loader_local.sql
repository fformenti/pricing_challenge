-- ----------------------
-- CREATE SCHEMA
-- ----------------------

CREATE SCHEMA b2w_schema;

-- ----------------------
-- DROP TABLES
-- ----------------------

DROP TABLE IF EXISTS b2w_schema.sales;
DROP TABLE IF EXISTS b2w_schema.comp_prices;
DROP TABLE IF EXISTS b2w_schema.sales_agg;

-- ----------------------
-- CREATING AND IMPORTING RAW TABLES
-- ----------------------

create table b2w_schema.sales (
           PROD_ID char(2)
           , DATE_ORDER_str char(10)
           , QTY_ORDER float
           , REVENUE float) ;


create table b2w_schema.comp_prices (
       PROD_ID char(2)
       , DATE_EXTRACTION_str varchar(20)
       , COMPETITOR char(2)
       , COMPETITOR_PRICE float
       , PAY_TYPE int) ;

commit;

-- ----------------------
-- Copying data from the folder to the db tables
-- ----------------------

COPY b2w_schema.sales FROM
          '/Users/felipeformentiferreira/Documents/github_portfolio/pricing_challenge/data/input/sales.csv'
                  DELIMITER ',' CSV HEADER;


COPY b2w_schema.comp_prices FROM
          '/Users/felipeformentiferreira/Documents/github_portfolio/pricing_challenge/data/input/comp_prices.csv'
                  DELIMITER ',' CSV HEADER;


-- ----------------------
-- Fixing which schema to look
-- ----------------------
set search_path = 'b2w_schema';

-- ----------------------
-- CREATING NEW TABLES
-- ----------------------

CREATE TABLE sales_agg AS
  SELECT PROD_ID, DATE_ORDER_str,
          sum(QTY_ORDER) as qty_order,
          sum(REVENUE) as revenue,
          sum(REVENUE)/sum(QTY_ORDER) as price
  from sales
  GROUP BY PROD_ID,DATE_ORDER_str;

-- ----------------------
-- Feature Engineering
-- ----------------------

-- sales
ALTER TABLE sales ADD COLUMN date_order DATE;
UPDATE sales SET date_order = TO_DATE(DATE_ORDER_str, 'YYYY/MM/DD');

-- sales_agg
ALTER TABLE sales_agg ADD COLUMN date_order DATE;
ALTER TABLE sales_agg ADD COLUMN day_week varchar(10); -- fix this (maybe put in the generating query)
ALTER TABLE sales_agg ADD COLUMN month int;
UPDATE sales_agg SET date_order = TO_DATE(DATE_ORDER_str, 'YYYY/MM/DD');
UPDATE sales_agg SET day_week = to_char(date_order, 'day');
UPDATE sales_agg SET month = EXTRACT(MONTH FROM "date_order");

-- comp_prices
ALTER TABLE comp_prices ADD COLUMN DATE_EXTRACTION DATE;
ALTER TABLE comp_prices ADD COLUMN date_order DATE;
UPDATE comp_prices SET DATE_EXTRACTION = TO_DATE(DATE_EXTRACTION_str, 'YYYY/MM/DD HH24:MI:SS');
UPDATE comp_prices SET date_order = TO_DATE(DATE_EXTRACTION_str, 'YYYY/MM/DD');

-- dropping columns
ALTER TABLE sales DROP COLUMN date_order_str RESTRICT;
ALTER TABLE sales_agg DROP COLUMN date_order_str RESTRICT;
ALTER TABLE comp_prices DROP COLUMN DATE_EXTRACTION_str RESTRICT;



-- Queries for the models
select PROD_ID,date_order,competitor, min(COMPETITOR_PRICE) as price from comp_prices
  GROUP BY PROD_ID,date_order,COMPETITOR,PAY_TYPE
  HAVING PAY_TYPE = 2;


-- Queries por presentation
select prod_id, date_order, qty_order, revenue from sales
order by prod_id, date_order
limit 5;

select prod_id, date_order, qty_order, revenue from sales_agg
order by prod_id, date_order
limit 5;

select * from comp_prices
order by prod_id, date_order, COMPETITOR
limit 5;

select prod_id, date_order, COMPETITOR, PAY_TYPE, count(*) as cnt from comp_prices
GROUP BY prod_id, date_order, COMPETITOR, PAY_TYPE
order by cnt DESC
limit 5;

select prod_id, date_order, COMPETITOR, count(*) as cnt from comp_prices
GROUP BY prod_id, date_order, COMPETITOR
order by cnt ASC
limit 5;