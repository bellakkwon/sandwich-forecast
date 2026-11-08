# AWS S3에서 한 주의 예측 재현

## 목적과 확인 결과

2013년 3월 24일 주의 저장 모델과 저장 입력을 S3에서 읽어 예측하고, 기존 결과와 같은지 확인했습니다. 새로운 성능 실험이 아니라 **이미 평가한 예측의 재현 확인**입니다.

| 항목 | 실제 실행 결과 |
|---|---|
| 실행 리전 | **`ap-southeast-2` (시드니)** |
| 입력·참조·대응 행 수 | **172 / 172 / 172** |
| 원예측 최대 절대 차이 | `2.842170943040401e-14` |
| 0 제한 후 예측 최대 절대 차이 | `2.842170943040401e-14` |
| 허용오차 | `1e-9` |
| 검사 및 종료 상태 | 모든 검사 통과, `passed: true`, 종료 코드 `0` |
| Python / LightGBM | `3.12.14` / `4.6.0` |
| run_id | `20261108T181122368834Z_f8084b3b` |

**퍼블릭 액세스 차단 설정은 콘솔에서 확인했으며, 실행 로그와는 별도 근거입니다.** 전용 IAM 사용자는 지정한 버킷 접두사에 필요한 권한만 사용했습니다.

코드의 기본 리전은 `ap-northeast-2`(서울)입니다. 실제 실행에서는 `--region ap-southeast-2`를 명시했습니다. 아래 예시도 실제 실행 리전을 따릅니다. 다른 리전을 사용할 때는 버킷의 리전에 맞게 지정해야 합니다.

## 공개 저장소에 포함하지 않은 파일

데이터·모델 묶음과 실행 로그는 이 저장소에 포함하지 않습니다. 앞 단계의 로컬 분석 산출물로 `package` 명령을 실행해 묶음을 만듭니다.

필요한 파일은 다음과 같습니다.

- 3월 24일 예측에 실제 사용한 저장 모델
- 같은 주의 저장 입력 172행
- 같은 주의 기존 예측 **전체 172행**

공통 평가 부분집합을 참조 예측으로 사용하면 안 됩니다. 키는 `Store_num`, `INV_NUMBER`, `week`입니다. 참조 예측에는 `prediction_raw`와 `prediction`이 모두 필요합니다. 실제 판매 타깃값은 재현 확인에 필요하지 않습니다.

## 저장값을 보존하는 이유

피처가 트리 분할 경계에 가까우면 작은 부동소수점 차이도 예측을 바꿀 수 있습니다. 따라서 저장 입력을 다시 계산하거나 임계값에 맞춰 수정하지 않습니다.

`package`는 입력·모델·참조 예측의 **원본 바이트를 그대로 복사**하고, CSV는 `float_precision='round_trip'`으로 읽습니다. 저장 모델로 직접 예측해 허용오차 이내인지 검사한 뒤 묶음을 만듭니다.

범주 수준과 피처 순서는 모델 파일에서 추출합니다. 패키지 파일의 SHA256을 매니페스트에 기록하고, 매니페스트 자체의 해시는 로컬 `trusted_manifest.sha256`에 보관합니다.

## 환경 준비

LightGBM은 실행 당시 버전인 `4.6.0`으로 고정합니다. 다른 라이브러리의 실행 버전도 로그에 남깁니다. 저장소 최상위 폴더에서 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

macOS에서 OpenMP 라이브러리가 필요한 경우 로컬 환경에 맞게 준비합니다. 라이브러리 버전이 같더라도 결과를 가정하지 않고 실제 차이를 검사해야 합니다.

## AWS 설정

1. 계정의 요금과 무료 사용 조건을 확인합니다. 무료 사용을 보장하지 않으며, 예산 알림은 과금을 자동으로 차단하지 않습니다.
2. 전용 비공개 버킷을 만들고 퍼블릭 액세스 차단 네 가지 설정을 모두 유지합니다. 이번 예시는 기본 암호화 SSE-S3를 사용합니다.
3. 프로필 또는 Boto3의 표준 자격증명 체인을 사용합니다. 액세스 키를 코드·문서·노트북·커밋에 넣지 않습니다.
4. 실행용 권한과 리소스 관리·정리용 권한을 구분합니다.

### 실행용 IAM 정책 예시

`<BUCKET>`을 자신의 버킷 이름으로 바꿉니다. 예시에는 버킷 생성·설정·삭제 권한이 포함되지 않습니다.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": [
        "arn:aws:s3:::<BUCKET>/sandwich-repro/2013-03-24/input/*",
        "arn:aws:s3:::<BUCKET>/sandwich-repro/2013-03-24/models/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": "s3:PutObject",
      "Resource": [
        "arn:aws:s3:::<BUCKET>/sandwich-repro/2013-03-24/input/*",
        "arn:aws:s3:::<BUCKET>/sandwich-repro/2013-03-24/models/*",
        "arn:aws:s3:::<BUCKET>/sandwich-repro/2013-03-24/output/*",
        "arn:aws:s3:::<BUCKET>/sandwich-repro/2013-03-24/logs/*"
      ]
    }
  ]
}
```

실행 스크립트는 `ListBucket`이나 `DeleteObject`를 사용하지 않습니다. 정리는 전용 접두사의 조회·삭제와 버킷 삭제 권한이 있는 관리 프로필에서 수행합니다.

## 실행 순서

### 1. 묶음 생성

아래 경로는 실제 로컬 파일로 바꿉니다. 출력 폴더가 이미 있으면 덮어쓰지 않습니다.

```bash
python 11_s3_reproduce.py package --week 2013-03-24 \
  --model <저장모델.txt> \
  --features <저장입력.csv> \
  --expected <전체172행기존예측.csv> \
  --out package_2013-03-24
```

### 2. 업로드

```bash
python 11_s3_reproduce.py upload \
  --profile sandwich-repro --region ap-southeast-2 \
  --bucket <BUCKET> --prefix sandwich-repro/2013-03-24 \
  --package package_2013-03-24
```

### 3. S3에서 읽어 예측하고 대조

```bash
python 11_s3_reproduce.py run \
  --profile sandwich-repro --region ap-southeast-2 \
  --bucket <BUCKET> --prefix sandwich-repro/2013-03-24 \
  --manifest-sha256 "$(cat package_2013-03-24/trusted_manifest.sha256)" \
  --local-copy s3_run_copies
```

로컬에서 신뢰하는 매니페스트 해시를 전달합니다. S3의 파일과 매니페스트가 함께 변경되는 경우를 잡기 위해, 기준 해시까지 같은 S3 위치에서 새로 받아 사용하지 않습니다.

성공은 종료 코드 `0`, 실패는 `1`입니다. 종료 코드와 `passed`를 함께 확인합니다. 허용오차는 묶음에 고정되어 있으며, 실패한 결과를 통과시키려고 완화하지 않습니다.

## 저장 구조와 검사

| 접두사 | 내용 |
|---|---|
| `input/` | 입력, 참조 예측, 범주 수준, 피처 순서, 매니페스트 |
| `models/` | 저장 모델 |
| `output/<run_id>/` | 재현 예측 |
| `logs/<run_id>/` | 실행 로그 |

S3의 폴더는 객체 키의 접두사입니다.

해시가 다르면 예측 전에 중단합니다. 행 수, 단일 주, 키의 일대일 대응, 범주 수준, 피처 순서, 예측값의 유한성도 검사합니다. 원예측과 0 제한 후 예측을 모두 대조합니다.

로그에는 UTC 실행 시각, 상태, 파일 해시, 행 수, 최대 절대 차이, 허용오차, 검사 결과, 라이브러리 버전을 기록합니다. S3에 로그를 쓰지 못한 경우에는 가능한 로컬 기록을 보존하고 성공으로 표시하지 않습니다.

## 종료 후 정리

먼저 결과와 로그의 로컬 사본을 확인합니다. 전용 실습 버킷인지 확인한 뒤, 정리 권한이 있는 프로필로 실행합니다. 공유 버킷은 삭제하지 않습니다.

```bash
aws s3 ls s3://<BUCKET>/sandwich-repro/2013-03-24/ \
  --recursive --profile <정리프로필>

aws s3 rm s3://<BUCKET>/sandwich-repro/2013-03-24/ \
  --recursive --profile <정리프로필>

aws s3 rb s3://<BUCKET> --profile <정리프로필>
```

버전 관리가 켜져 있으면 과거 버전과 삭제 마커도 정리해야 합니다. 실습용으로 만든 장기 액세스 키는 종료 후 비활성화하거나 삭제합니다. 기존 공용 키를 임의로 삭제하지 않습니다.

## 검증 범위

실제 AWS 실행 전에 로컬 모의 S3에서 정상 재현과 여섯 가지 실패 조건을 시험했습니다. 파일 변조, 참조 예측 변경, 중복 키, 모델 누락, 참조값 NaN, 매니페스트 기준 해시 불일치가 포함됩니다.

모의 시험은 실제 IAM·과금·계정 정책을 증명하지 않습니다. 실제 AWS 성공 로그와 콘솔의 접근 설정은 별도의 근거로 구분합니다. 테스트 입력·모델·원본 로그는 공개 저장소에 포함하지 않습니다.

이 실습은 S3 입출력을 이용한 저장 예측 재현입니다. SageMaker 배포, 자동 재학습, 현업 운영 파이프라인 구축으로 표현하지 않습니다.

## 공식 참고

- [S3 퍼블릭 액세스 차단](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-control-block-public-access.html)
- [Boto3 자격증명](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html)
