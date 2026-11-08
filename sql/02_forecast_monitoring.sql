-- 예측 점검: 최종 12주 고정 평가 키 기준. 성능 지표는 0 제한 후 prediction 사용.
-- 지표 정의
--   MAE  = 절대오차 합 / 평가 건수
--   WAPE = 절대오차 합 / 실제 합          (집계 단위 안의 여러 건을 합산한 값)
--   Bias = (예측 합 - 실제 합) / 실제 합   (+ 과대, - 과소)
-- 평가 건수 0이면 세 지표 모두 NULL, 실제 합 0이면 WAPE·Bias NULL.

-- 0) 입력 무결성
CREATE OR REPLACE TABLE pred_input_checks AS
SELECT * FROM (VALUES
    ('frozen_keys',               (SELECT count(*) FROM frozen_keys)),
    ('frozen_keys_duplicate',     (SELECT count(*) FROM (SELECT Store_num, INV_NUMBER, week FROM frozen_keys GROUP BY ALL HAVING count(*) > 1))),
    ('pred_rows',                 (SELECT count(*) FROM common_pred)),
    ('pred_duplicate_keys',       (SELECT count(*) FROM (SELECT Store_num, INV_NUMBER, week FROM common_pred GROUP BY ALL HAVING count(*) > 1))),
    ('keys_missing_prediction',   (SELECT count(*) FROM frozen_keys k ANTI JOIN common_pred p USING (Store_num, INV_NUMBER, week))),
    ('predictions_not_in_keys',   (SELECT count(*) FROM common_pred p ANTI JOIN frozen_keys k USING (Store_num, INV_NUMBER, week))),
    ('null_actual_or_forecast',   (SELECT count(*) FROM common_pred WHERE actual IS NULL OR last_week IS NULL OR mean_4weeks IS NULL OR prediction IS NULL)),
    ('null_or_nonfinite_forecast',(SELECT count(*) FROM common_pred WHERE prediction_raw IS NULL OR NOT isfinite(prediction_raw) OR NOT isfinite(prediction) OR NOT isfinite(actual) OR NOT isfinite(last_week) OR NOT isfinite(mean_4weeks))),
    ('prediction_not_clipped_raw',(SELECT count(*) FROM common_pred WHERE abs(prediction - greatest(prediction_raw, 0)) > 1e-12)),
    ('scheduled_keys',            (SELECT count(*) FROM schedule)),
    ('scheduled_pairs',           (SELECT count(*) FROM (SELECT DISTINCT Store_num, INV_NUMBER FROM schedule)))
) AS t(check_name, value);

-- 1) 방법별 세로형
CREATE OR REPLACE TABLE pred_long AS
SELECT p.Store_num, p.INV_NUMBER, p.week, p.category, p.volume_band, p.actual, m.method, m.forecast
FROM common_pred p
SEMI JOIN frozen_keys k USING (Store_num, INV_NUMBER, week)
CROSS JOIN LATERAL (VALUES
    ('last_week',   p.last_week),
    ('mean_4weeks', p.mean_4weeks),
    ('prediction',  p.prediction)
) AS m(method, forecast);

CREATE OR REPLACE MACRO agg_metrics(n, actual_sum, forecast_sum, abs_sum) AS STRUCT_PACK(
    MAE  := CASE WHEN n > 0 THEN abs_sum / n END,
    WAPE := CASE WHEN n > 0 AND actual_sum <> 0 THEN abs_sum / actual_sum END,
    Bias := CASE WHEN n > 0 AND actual_sum <> 0 THEN (forecast_sum - actual_sum) / actual_sum END
);

-- 2) 전체·유형·판매량 구간·주 단위 집계
CREATE OR REPLACE TABLE group_metrics AS
WITH g AS (
    SELECT 'overall' AS dimension, 'all' AS grp, * FROM pred_long
    UNION ALL SELECT 'category', category, * FROM pred_long
    UNION ALL SELECT 'volume_band', volume_band, * FROM pred_long
    UNION ALL SELECT 'week', strftime(week, '%Y-%m-%d'), * FROM pred_long
), s AS (
    SELECT dimension, grp, method,
           count(*)                    AS n,
           sum(actual)                 AS actual_sum,
           sum(forecast)               AS forecast_sum,
           sum(abs(forecast - actual)) AS absolute_error_sum
    FROM g GROUP BY ALL
)
SELECT dimension, grp AS "group", method, n, actual_sum, absolute_error_sum,
       agg_metrics(n, actual_sum, forecast_sum, absolute_error_sum).MAE  AS MAE,
       agg_metrics(n, actual_sum, forecast_sum, absolute_error_sum).WAPE AS WAPE,
       agg_metrics(n, actual_sum, forecast_sum, absolute_error_sum).Bias AS Bias
FROM s
ORDER BY dimension, "group", method;

-- 3) 조합별 12주 점검(172조합 × 3방법). 평가 건수가 0인 조합도 일정표에서 남긴다.
--    규칙: n < 10 → insufficient_data
--          과대 주 >= 8 이고 Bias >= 0.10 → overprediction_review
--          과소 주 >= 8 이고 Bias <= -0.10 → underprediction_review
--          그 외 → not_flagged.  예측 = 실제인 주는 어느 방향에도 세지 않음. 연속 조건 아님.
CREATE OR REPLACE TABLE pair_flags AS
WITH pairs AS (
    SELECT DISTINCT s.Store_num, s.INV_NUMBER, m.method
    FROM schedule s
    CROSS JOIN (VALUES ('last_week'), ('mean_4weeks'), ('prediction')) AS m(method)
), s AS (
    SELECT p.Store_num, p.INV_NUMBER, p.method,
           count(l.actual)                                  AS n,
           coalesce(sum(CASE WHEN l.forecast > l.actual THEN 1 ELSE 0 END), 0) AS over_weeks,
           coalesce(sum(CASE WHEN l.forecast < l.actual THEN 1 ELSE 0 END), 0) AS under_weeks,
           sum(l.actual)                                    AS actual_sum,
           sum(l.forecast)                                  AS forecast_sum,
           sum(abs(l.forecast - l.actual))                  AS absolute_error_sum
    FROM pairs p
    LEFT JOIN pred_long l USING (Store_num, INV_NUMBER, method)
    GROUP BY ALL
), m AS (
    SELECT *,
           agg_metrics(n, actual_sum, forecast_sum, absolute_error_sum).MAE  AS MAE,
           agg_metrics(n, actual_sum, forecast_sum, absolute_error_sum).WAPE AS WAPE,
           agg_metrics(n, actual_sum, forecast_sum, absolute_error_sum).Bias AS Bias
    FROM s
)
SELECT Store_num, INV_NUMBER, method, over_weeks, under_weeks,
       CASE
           WHEN n < 10 THEN 'insufficient_data'
           WHEN over_weeks  >= 8 AND Bias >=  0.10 THEN 'overprediction_review'
           WHEN under_weeks >= 8 AND Bias <= -0.10 THEN 'underprediction_review'
           ELSE 'not_flagged'
       END AS status,
       n, actual_sum, absolute_error_sum, MAE, WAPE, Bias
FROM m
ORDER BY Store_num, INV_NUMBER, method;

CREATE OR REPLACE TABLE review_counts_sql AS
PIVOT (SELECT method, status FROM pair_flags)
ON status IN ('insufficient_data', 'not_flagged', 'overprediction_review', 'underprediction_review')
USING count(*)
GROUP BY method
ORDER BY method;
