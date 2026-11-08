"""12 · SQL 운영 점검 (DuckDB)

이미 고정된 최종 12주 결과를 SQL로 다시 계산하고, Python 산출물과 대조한다.
새 성능 실험이 아니며 모델·예측·규칙을 바꾸지 않는다.

입력
  raw/WEEKLY_SALES_10STORES.csv                    원천 점검용(정제 전 전체)
  final_evaluation/final_common_predictions.csv    고정 평가 집합의 예측
  final_evaluation/final_common_evaluation_keys.csv 고정 평가 키 2,027건
  final_evaluation/final_predictions_all.csv       예정 일정표(172조합 × 12주)
  final_evaluation/final_metrics.csv               Python 집계(대조 기준)
  final_evaluation/pair_review_flags.csv           Python 조합 점검(대조 기준)
  final_evaluation/review_counts.csv               Python 점검 대상 수(대조 기준)

출력: sql_monitoring/ (이미 있으면 덮어쓰지 않음)
종료 코드: 대조 전부 일치 0, 하나라도 불일치 1
필요: duckdb, pandas
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SQL_DIR = ROOT / "sql"
FE = ROOT / "final_evaluation"
KEY = ["Store_num", "INV_NUMBER", "week"]
PAIR = ["Store_num", "INV_NUMBER", "method"]
METRIC_COLS = ["actual_sum", "absolute_error_sum", "MAE", "WAPE", "Bias"]
ABS_TOL = 1e-9          # MAE·비율 지표의 절대 허용오차
REL_TOL = 1e-12         # 합계 지표의 상대 허용오차(합산 순서 차이)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_sql(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    for stmt in path.read_text(encoding="utf-8").split(";\n"):
        body = "\n".join(l for l in stmt.splitlines() if not l.strip().startswith("--")).strip()
        if body:
            con.execute(body)


def lit(p: Path) -> str:
    """CREATE VIEW는 매개변수를 받을 수 없어 경로를 SQL 문자열 리터럴로 안전하게 넣는다."""
    return "'" + str(p).replace("'", "''") + "'"


def register_inputs(con: duckdb.DuckDBPyConnection, raw: Path) -> None:
    con.execute(
        f"CREATE VIEW raw_sales AS SELECT * FROM read_csv({lit(raw)}, header=true, types={{'Date':'VARCHAR'}})"
    )
    con.execute(
        f"""CREATE VIEW common_pred AS
           SELECT Store_num::INTEGER AS Store_num, INV_NUMBER::INTEGER AS INV_NUMBER, week::DATE AS week,
                  category, volume_band,
                  actual::DOUBLE AS actual, last_week::DOUBLE AS last_week,
                  mean_4weeks::DOUBLE AS mean_4weeks,
                  prediction_raw::DOUBLE AS prediction_raw, prediction::DOUBLE AS prediction
           FROM read_csv({lit(FE / "final_common_predictions.csv")}, header=true, all_varchar=true)"""
    )
    con.execute(
        f"""CREATE VIEW frozen_keys AS
           SELECT Store_num::INTEGER AS Store_num, INV_NUMBER::INTEGER AS INV_NUMBER, week::DATE AS week
           FROM read_csv({lit(FE / "final_common_evaluation_keys.csv")}, header=true, all_varchar=true)"""
    )
    con.execute(
        f"""CREATE VIEW schedule AS
           SELECT DISTINCT Store_num::INTEGER AS Store_num, INV_NUMBER::INTEGER AS INV_NUMBER, week::DATE AS week
           FROM read_csv({lit(FE / "final_predictions_all.csv")}, header=true, all_varchar=true)"""
    )


def compare_numeric(a: pd.Series, b: pd.Series, col: str) -> tuple[bool, float]:
    both_na = a.isna() & b.isna()
    one_na = a.isna() ^ b.isna()
    diff = (a - b).abs().where(~both_na, 0.0)
    if col in ("actual_sum", "absolute_error_sum"):
        tol = np.maximum(ABS_TOL, REL_TOL * b.abs().fillna(0))
    else:
        tol = ABS_TOL
    ok = (not one_na.any()) and bool((diff.fillna(0) <= tol).all())
    return ok, float(diff.fillna(0).max()) if len(diff) else 0.0


def reconcile(con: duckdb.DuckDBPyConnection) -> dict:
    out: dict = {}

    # 1) 그룹 지표
    sql_m = con.sql("SELECT * FROM group_metrics").df()
    py_m = pd.read_csv(FE / "final_metrics.csv")
    # Python 산출물의 주 그룹은 "2013-01-06 00:00:00" 형식 → 날짜 부분만 비교
    wk = py_m["dimension"].eq("week")
    py_m.loc[wk, "group"] = py_m.loc[wk, "group"].astype(str).str[:10]
    k = ["dimension", "group", "method"]
    m = sql_m.merge(py_m, on=k, how="outer", suffixes=("_sql", "_py"), indicator=True, validate="one_to_one")
    res = {"rows_sql": len(sql_m), "rows_py": len(py_m),
           "keys_one_to_one": bool((m["_merge"] == "both").all())}
    both = m[m["_merge"] == "both"]
    res["n_equal"] = bool((both["n_sql"] == both["n_py"]).all())
    for c in METRIC_COLS:
        ok, mx = compare_numeric(both[f"{c}_sql"], both[f"{c}_py"], c)
        res[f"{c}_match"], res[f"{c}_max_abs_diff"] = ok, mx
    out["group_metrics"] = res

    # 2) 조합별 점검
    sql_f = con.sql("SELECT * FROM pair_flags").df()
    py_f = pd.read_csv(FE / "pair_review_flags.csv")
    f = sql_f.merge(py_f, on=PAIR, how="outer", suffixes=("_sql", "_py"), indicator=True, validate="one_to_one")
    res = {"rows_sql": len(sql_f), "rows_py": len(py_f),
           "keys_one_to_one": bool((f["_merge"] == "both").all())}
    both = f[f["_merge"] == "both"]
    for c in ["n", "over_weeks", "under_weeks", "status"]:
        res[f"{c}_equal"] = bool((both[f"{c}_sql"] == both[f"{c}_py"]).all())
    for c in METRIC_COLS:
        ok, mx = compare_numeric(both[f"{c}_sql"], both[f"{c}_py"], c)
        res[f"{c}_match"], res[f"{c}_max_abs_diff"] = ok, mx
    mism = both[both["status_sql"] != both["status_py"]][PAIR + ["status_sql", "status_py"]]
    res["status_mismatches"] = mism.to_dict(orient="records")
    out["pair_flags"] = res

    # 3) 점검 대상 수
    sql_c = con.sql("SELECT * FROM review_counts_sql").df().set_index("method").sort_index()
    py_c = pd.read_csv(FE / "review_counts.csv").set_index("method").sort_index()
    cols = sorted(set(sql_c.columns) | set(py_c.columns))
    sql_c, py_c = sql_c.reindex(columns=cols, fill_value=0), py_c.reindex(columns=cols, fill_value=0)
    out["review_counts"] = {"equal": bool(sql_c.astype(int).equals(py_c.astype(int))),
                            "sql": sql_c.astype(int).to_dict(orient="index")}

    # 4) 입력 무결성(기대값)
    chk = dict(con.sql("SELECT check_name, value FROM pred_input_checks").fetchall())
    expect_zero = ["frozen_keys_duplicate", "pred_duplicate_keys", "keys_missing_prediction",
                   "predictions_not_in_keys", "null_actual_or_forecast", "prediction_not_clipped_raw", "null_or_nonfinite_forecast"]
    out["input_checks"] = {"values": chk,
                           "all_expected_zero": all(chk[c] == 0 for c in expect_zero),
                           "fixed_counts_equal": all(chk[k] == v for k, v in {
                               "frozen_keys": 2027, "pred_rows": 2027,
                               "scheduled_keys": 2064, "scheduled_pairs": 172}.items())}
    return out


def flatten_ok(rec: dict) -> bool:
    keys_ok = [v for k, v in rec.items()
               if isinstance(v, bool) and (k.endswith("_match") or k.endswith("_equal")
                                           or k in ("keys_one_to_one", "equal", "all_expected_zero"))]
    return all(keys_ok)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "raw" / "WEEKLY_SALES_10STORES.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "sql_monitoring")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} 가 이미 있습니다. 덮어쓰지 않습니다.")

    con = duckdb.connect()
    register_inputs(con, a.raw)
    run_sql(con, SQL_DIR / "01_source_checks.sql")
    run_sql(con, SQL_DIR / "02_forecast_monitoring.sql")
    recon = reconcile(con)
    passed = all(flatten_ok(v) for v in recon.values())

    a.out.mkdir(parents=True)
    tables = ["source_check_summary", "chk_date_mismatch", "chk_non_sunday", "chk_duplicate_keys",
              "chk_store_week_gaps", "chk_negative_sold", "chk_price_nonpositive", "chk_sales_sign",
              "info_sales_ratio", "pred_input_checks", "group_metrics", "pair_flags", "review_counts_sql"]
    for t in tables:
        con.sql(f"SELECT * FROM {t}").df().to_csv(a.out / f"{t}.csv", index=False, encoding="utf-8-sig")

    inputs = [a.raw] + [FE / n for n in ["final_common_predictions.csv", "final_common_evaluation_keys.csv",
                                         "final_predictions_all.csv", "final_metrics.csv",
                                         "pair_review_flags.csv", "review_counts.csv"]]
    manifest = {
        "purpose": "고정된 최종 결과의 SQL 재계산 및 Python 산출물 대조. 새 성능 실험 아님.",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": passed,
        "tolerance": {"ratio_abs": ABS_TOL, "sum_rel": REL_TOL},
        "reconciliation": recon,
        "source_check_summary": con.sql("SELECT * FROM source_check_summary").df().to_dict(orient="records"),
        "input_sha256": {p.name: sha(p) for p in inputs},
        "script_sha256": {"12_sql_monitoring.py": sha(Path(__file__)),
                          **{p.name: sha(p) for p in sorted(SQL_DIR.glob("*.sql"))}},
        "versions": {"python": platform.python_version(), "duckdb": duckdb.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    (a.out / "reconciliation.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
                                               encoding="utf-8")

    print(json.dumps({"passed": passed,
                      "group_metrics": {k: v for k, v in recon["group_metrics"].items() if "max_abs" in k or k == "keys_one_to_one"},
                      "pair_flags_status_equal": recon["pair_flags"]["status_equal"],
                      "review_counts_equal": recon["review_counts"]["equal"],
                      "input_checks_ok": recon["input_checks"]["all_expected_zero"]},
                     ensure_ascii=False, indent=2))
    print(con.sql("SELECT * FROM source_check_summary").df().to_string(index=False))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
