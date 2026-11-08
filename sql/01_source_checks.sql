-- 원천 데이터 점검: 정제 전 원본 전체(10개 매장)를 대상으로 한다.
-- severity: error = 키·날짜 무결성 문제, review = 자동 오류가 아니라 검토 후보, info = 기술 통계.

CREATE OR REPLACE TABLE src AS
SELECT
    *,
    strptime(Date, '%m/%d/%Y')::DATE AS date_str,
    make_date(Year, Month, Day)      AS date_ymd
FROM raw_sales;

-- 1) 날짜 문자열과 연·월·일 컬럼 불일치
CREATE OR REPLACE TABLE chk_date_mismatch AS
SELECT Store_num, date_str, date_ymd, count(*) AS rows
FROM src
WHERE date_str <> date_ymd
GROUP BY ALL
ORDER BY Store_num, date_str;

-- 2) 일요일이 아닌 날짜(문자열 기준·연월일 기준 각각)
CREATE OR REPLACE TABLE chk_non_sunday AS
SELECT 'date_str' AS basis, Store_num, date_str AS d, count(*) AS rows
FROM src WHERE dayofweek(date_str) <> 0 GROUP BY ALL
UNION ALL
SELECT 'date_ymd', Store_num, date_ymd, count(*)
FROM src WHERE dayofweek(date_ymd) <> 0 GROUP BY ALL
ORDER BY basis, Store_num, d;

-- 3) 매장×상품코드×주 키 중복(Date 문자열 기준)
CREATE OR REPLACE TABLE chk_duplicate_keys AS
SELECT Store_num, INV_NUMBER, date_str, count(*) AS rows
FROM src
GROUP BY ALL
HAVING count(*) > 1
ORDER BY Store_num, INV_NUMBER, date_str;

-- 4) 매장별 주 달력 누락: 전체 일요일 달력 대비 기록이 없는 주
CREATE OR REPLACE TABLE chk_store_week_gaps AS
WITH cal AS (
    SELECT week::DATE AS week
    FROM generate_series(DATE '2012-04-01', DATE '2013-03-24', INTERVAL '7 days') AS t(week)
), stores AS (SELECT DISTINCT Store_num FROM src),
have AS (SELECT DISTINCT Store_num, date_str AS week FROM src)
SELECT s.Store_num, c.week
FROM stores s CROSS JOIN cal c
ANTI JOIN have h ON h.Store_num = s.Store_num AND h.week = c.week
ORDER BY 1, 2;

-- 5) 음수 판매수량(이유 미확인 → 분석 타깃에서는 결측)
CREATE OR REPLACE TABLE chk_negative_sold AS
SELECT Store_num, INV_NUMBER, Description, date_str, Price, Sold, Sales
FROM src WHERE Sold < 0
ORDER BY Store_num, date_str, INV_NUMBER;

-- 6) 가격 0 이하: 할인·리워드·조정 코드일 수 있으므로 검토 후보
CREATE OR REPLACE TABLE chk_price_nonpositive AS
SELECT INV_NUMBER, any_value(Description) AS description,
       count(*) AS rows, sum(Sold) AS sold_sum, sum(Sales) AS sales_sum,
       min(Price) AS price_min, max(Price) AS price_max
FROM src WHERE Price <= 0
GROUP BY INV_NUMBER
ORDER BY rows DESC;

-- 7) 매출 부호 불일치: 가격·수량이 양수인데 매출이 음수, 또는 그 반대(검토 후보)
--    가격×수량과 매출의 크기 차이는 원천 집계 방식이 확인되지 않아 오류로 보지 않는다.
CREATE OR REPLACE TABLE chk_sales_sign AS
SELECT Store_num, INV_NUMBER, Description, date_str, Price, Sold, Sales
FROM src
WHERE (Price > 0 AND Sold > 0 AND Sales < 0)
   OR (Price < 0 AND Sold > 0 AND Sales > 0)
ORDER BY Store_num, date_str, INV_NUMBER;

-- 8) 매출 ÷ (가격×수량) 비율 분포(기술 통계)
CREATE OR REPLACE TABLE info_sales_ratio AS
SELECT
    count(*)                                   AS rows_used,
    quantile_cont(Sales / (Price * Sold), 0.05) AS p05,
    quantile_cont(Sales / (Price * Sold), 0.50) AS p50,
    quantile_cont(Sales / (Price * Sold), 0.95) AS p95
FROM src
WHERE Price > 0 AND Sold > 0;

-- 요약
CREATE OR REPLACE TABLE source_check_summary AS
SELECT * FROM (VALUES
    ('date_mismatch_rows',    'error',  (SELECT coalesce(sum(rows),0) FROM chk_date_mismatch)),
    ('date_mismatch_groups',  'error',  (SELECT count(*) FROM chk_date_mismatch)),
    ('non_sunday_str_rows',   'error',  (SELECT coalesce(sum(rows),0) FROM chk_non_sunday WHERE basis='date_str')),
    ('non_sunday_ymd_rows',   'error',  (SELECT coalesce(sum(rows),0) FROM chk_non_sunday WHERE basis='date_ymd')),
    ('duplicate_key_groups',  'error',  (SELECT count(*) FROM chk_duplicate_keys)),
    ('store_week_gaps',       'error',  (SELECT count(*) FROM chk_store_week_gaps)),
    ('negative_sold_rows',    'review', (SELECT count(*) FROM chk_negative_sold)),
    ('price_nonpositive_codes','review',(SELECT count(*) FROM chk_price_nonpositive)),
    ('price_nonpositive_rows','review', (SELECT coalesce(sum(rows),0) FROM chk_price_nonpositive)),
    ('sales_sign_rows',       'review', (SELECT count(*) FROM chk_sales_sign)),
    ('total_rows',            'info',   (SELECT count(*) FROM src)),
    ('stores',                'info',   (SELECT count(DISTINCT Store_num) FROM src)),
    ('weeks_date_str',        'info',   (SELECT count(DISTINCT date_str) FROM src))
) AS t(check_name, severity, value_raw);

CREATE OR REPLACE TABLE source_check_summary AS
SELECT check_name, severity, value_raw::BIGINT AS value FROM source_check_summary
