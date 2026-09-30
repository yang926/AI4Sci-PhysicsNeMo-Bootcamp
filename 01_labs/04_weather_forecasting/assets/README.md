# 기상 실습 자료의 출처 · 한국어판

이 실습은 사전 학습된 모델로 추론하며 새 모델을 학습하지 않습니다.
고정된 사례의 시작 시각은 2022년 9월 1일 00:00 UTC이며, 이후 48시간을 예측합니다.
그 이후 시각의 ERA5 장은 예측 입력과 분리된 검증 데이터입니다.

## 모델과 구현

- **NVIDIA FourCastNet1 / AFNO**: [nvidia/fourcastnet1](https://huggingface.co/nvidia/fourcastnet1)의
  리비전 `c67a63995f6c8e0e557eb3d791f32f437e9b02d5`에 있는 공식 26채널 체크포인트입니다.
  모델 카드에 명시된 라이선스는 Apache-2.0입니다. 체크포인트와 정규화 파일은
  해당 리비전에서 내려받으며, 불러오기 전에 전체 SHA-256 해시를 확인합니다.
  모델 파일은 이 저장소에 포함되어 있지 않습니다.
- **NVIDIA Earth2Studio**: 정규화와 6시간 간격의 반복 예측은
  [공식 FCN 래퍼](https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/models/px/fcn.py)를 따릅니다.
  모델은 상태의 증분이 아니라 정규화된 다음 상태 전체를 예측합니다.
- **상대습도 전처리**: IFS 혼합상 공식은
  [Earth2Studio DerivedRH 구현](https://github.com/NVIDIA/earth2studio/blob/486c5daa98841b0ce93cb78cb70d0dee7a8a56e6/earth2studio/models/dx/derived.py)을 바탕으로 적용했으며,
  리비전은 `486c5daa98841b0ce93cb78cb70d0dee7a8a56e6`으로 고정했습니다.
  Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. Apache-2.0.
  저장소의 [라이선스](../../../LICENSE)를 참고하세요. 강의용 구현은 NumPy를 사용하며,
  ERA5의 온도(K), 비습(kg/kg), 기압(hPa)을 입력받아 계산한 상대습도를
  FP32 백분율로 저장합니다. 공식 구현의 혼합상 혼합 방식과 최종 상대습도 범위
  0–100%를 유지합니다. 원본 비습을 바꾸거나 모델 예측값의 범위를 잘라내지 않습니다.
  진단 기록에는 상대습도 범위 제한의 영향을 받은 전처리 값의 개수와 범위를 저장합니다.

## ERA5 재분석

ERA5는 ECMWF의 Copernicus Climate Change Service에서 제작하며,
[Google ARCO-ERA5](https://github.com/google-research/arco-era5)를 통해 접근합니다.
[ERA5 데이터 라이선스](https://cds.climate.copernicus.eu/licences/cc-by)와
[ARCO 인용 안내](https://github.com/google-research/arco-era5#how-to-cite-this-work)를 참고하세요.
이 자료는 재분석이며, 직접 관측 자료나 실시간 현업 자료가 아닙니다.

`era5-source-plan.json`은 정확한 GCS 객체 세대와 원본 체크섬을 고정합니다.
`era5-source-metadata.json`은 이에 대응하는 원본 차원과 단위를 기록합니다.
이 작은 메타데이터 문서에는 기상 배열이 들어 있지 않습니다. `prepare.py`는
이 파일들의 SHA-256 해시를 확인합니다. 원본 청크는 GCS의 MD5와 크기로,
준비된 파일은 기록된 SHA-256 해시로 확인합니다.
내려받은 원본 메타데이터의 SHA와 로컬에서 서식을 정리한 메타데이터의 SHA를
별도로 기록합니다. JSON 서식은 기반 데이터셋을 바꾸지 않습니다.

원본 장에 적용하는 변환은 명확합니다. 필요한 채널과 기압면을 선택하고,
공식 720×1440 격자에 맞게 남극점 행을 제거하며, 위 공식으로 상대습도를 계산합니다.
보간, 예측 진폭 보정, 미래 상태 피드백, 공간 정렬은 사용하지 않습니다.
전체 출처와 변환 진단 기록은 외부 캐시의
`data-20220901/provenance.json`에 저장됩니다.

## 지리적 윤곽선

`coastline.geojson`은 퍼블릭 도메인으로 배포되는 Natural Earth 1:110m 해안선 자료입니다.
[원본 파일](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_coastline.geojson)과
[Natural Earth 이용 조건](https://www.naturalearthdata.com/about/terms-of-use/)을 참고하세요.
SHA-256: `851f581ff5ffb844deed8ae1a9ce22e3c4bb3d74fa342cadb5d8e39b41ae7c3c`.
생성된 지도에는 출처로 Natural Earth가 표시됩니다.

## 검증 범위

`weather-acceptance.json`에는 이 사례를 평가하기 전에 정한 기준이 들어 있습니다.
예측 점수는 물리 단위와 위도 코사인에 따른 면적 가중치를 사용합니다.
초기 프레임은 예측 오차 지표에서 제외합니다. 초기 상태를 유지하는 지속성 예측을
기준으로 삼아, 0보다 큰 예측 선행 시간 여덟 개를 모두 보고합니다.
과거 사례 하나가 기준을 통과했다고 해서 현업 기상 예측 성능이나
다른 GPU에서의 실행 시간이 입증되는 것은 아닙니다.
