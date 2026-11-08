# 데이터 출처와 이용 조건

## 출처

Trent Spaulding, Edgar Hassler, Charles Edwards, Joseph Cazier (2020).
*Sandwich Analytics: A Dataset Comprising One Year’s Weekly Sales Data Correlated with Crime, Demographics, and Weather.* Mendeley Data, Version 2.

- DOI: https://doi.org/10.17632/6htjnfs78b.2
- 데이터 페이지: https://data.mendeley.com/datasets/6htjnfs78b/2
- 관련 논문: https://doi.org/10.1016/j.dib.2019.104252

관련 논문은 미국 퀵서비스 매장의 자료라고 설명합니다. 이 프로젝트는 제공받은 10개 매장 파일에서 날짜 충돌이 있는 24번 매장을 보류하고, 9개 매장을 분석했습니다.

## 라이선스

원데이터의 라이선스는 **CC BY 4.0**입니다.

https://creativecommons.org/licenses/by/4.0/

공유·가공 시 적절한 출처 표시, 라이선스 링크, 변경 사항 표시가 필요합니다. 제공자가 이 프로젝트를 보증하거나 후원한다는 뜻은 아닙니다.

이 저장소는 공개 범위를 코드·문서·요약 지표로 제한하며, 원본 데이터와 파생 데이터 파일은 포함하지 않습니다. 원데이터의 라이선스는 이 저장소의 코드에 자동으로 적용되지 않습니다.

## 변경 사항

다음 처리를 수행했으며 원본 파일은 덮어쓰지 않았습니다.

- 날짜 충돌이 있는 매장 보류
- 상품명 후보 분류와 분석 대상 선별
- 음수 판매와 미관측 타깃을 결측으로 처리
- 시간순 분할과 과거 정보 기반 피처 생성
- 예측, 평가, 진단 집계

Version 2 설명에는 두 매장에서 한 주씩 날짜를 주 시작일 기준에 맞춰 보정했다고 명시되어 있습니다. 다만 대상 매장 번호는 명시되지 않았으므로, 발견한 7번·24번 매장과 같은 사례라고 확정하지 않았습니다. 24번 매장의 키 충돌은 보류했습니다.

## 최초 실험에 사용한 원본 파일의 SHA256

| 파일 | SHA256 |
|---|---|
| `WEEKLY_SALES_10STORES.csv` | `179640001216367c3ffa25080984228ca106606d777d48e5dce0e50bf5f7ffec` |
| `AppWichStoreAttributes.csv` | `11e5f84e646c6fd61629a9cba03d137d1095af875f7ff3331b1312b3d38909ff` |
| `WEEKLY_WEATHER.csv` | `f88ccbcbdaa99c709bfde31e1a9e286f49810502fd5f7d93b8507ce3fd646214` |

다른 버전의 데이터를 사용한다면 먼저 파일 해시, 행 수, 날짜 범위를 비교해야 합니다.
