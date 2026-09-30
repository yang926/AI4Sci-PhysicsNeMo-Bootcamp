<a id="teaching-diagrams"></a>
# 수업용 도식

이 로컬 SVG 17개는 현재 사용하는 실습 및 도전 과제 노트북 여덟 개에 삽입되어 있습니다.
실행 전에 영역/데이터의 구조와 실제 프로그램 연결 관계를 보여 줍니다.
그림은 개략도이며 학습된 결과나 수렴의 근거가 아닙니다.

SVG 소스를 직접 수정하세요. 그림을 보는 데 이미지 생성기, 외부 CDN, 브라우저 확장,
GPU 작업, 노트북 실행은 필요하지 않습니다. 도식은 아래 소스 함수와 일치하게 유지하세요.
구현 흐름을 보여 주는 설명을 완성된 도전 과제 정답으로 바꾸지 마세요.

| 자료 | 형상과 데이터 흐름의 출처 |
|---|---|
| `lab1-domain.svg`, `lab1-flow.svg` | [pinn_basics.py](../../../01_labs/01_pinn/source_code/pinn_basics.py), [labs.py](../../runtime/labs.py): `BasicPINN`, `loss_terms`, `optimize_lab` |
| `lab2-domain.svg`, `lab2-flow.svg` | [projectile.py](../../../01_labs/02_projectile/source_code/projectile.py): `ProjectileModel`, `loss_terms`, `optimize_projectile` |
| `lab3-domain.svg`, `lab3-flow.svg` | [diffusion_bar.py](../../../01_labs/03_heat_conduction/source_code/diffusion_bar.py), [labs.py](../../runtime/labs.py): 두 재료의 샘플링, 계면 잔차, 저장된 모델 추론 |
| `lab4-domain.svg`, `lab4-workflow.svg` | [run_forecast.py](../../../01_labs/04_weather_forecasting/source_code/run_forecast.py), [evaluate_weather.py](../../../01_labs/04_weather_forecasting/source_code/evaluate_weather.py): 전 지구 AFNO 추론과 독립적 검증 |
| `challenge1-domain.svg`, `challenge1-circle.svg`, `challenge1-flow.svg` | [wave_l1.py](../../../02_challenges/01_wave/wave_l1.py), [wave_l2.py](../../../02_challenges/01_wave/wave_l2.py), [wave_l3.py](../../../02_challenges/01_wave/wave_l3.py), [pinn.py](../../runtime/pinn.py): 정사각형/원판 샘플링, 별도의 PDE/신경망 객체, 네 가지 손실 |
| `challenge2-domain.svg`, `challenge2-flow.svg` | [chip_2d_l1.py](../../../02_challenges/02_fluid/chip_2d_l1.py), [chip_2d_l2.py](../../../02_challenges/02_fluid/chip_2d_l2.py), [chip_2d_l3.py](../../../02_challenges/02_fluid/chip_2d_l3.py): 마스크를 적용한 채널, 형상/조건/PDE 경로 |
| `challenge3-domain.svg`, `challenge3-flow.svg` | [기후 스크립트](../../../02_challenges/03_climate): 정사각형/시간 샘플링, 하나 또는 두 개의 장, 별도의 비교 경로 |
| `challenge4-domain.svg`, `challenge4-flow.svg` | [연산자 스크립트](../../../02_challenges/04_neural_operators): 주기 격자, 생성 함수의 규약, 데이터 손실과 PINO 스펙트럴 잔차 분기 |

설명 레이블은 한국어이며, 로컬 노트북 이미지 링크와 SVG 제목/설명 대체 텍스트를 제공합니다.
수정 후에는 레이블과 화살표 경로를 포함하여 렌더링된 그림을 검토하세요.
`ETC/tests/test_teaching_visuals.py`는 이미지 삽입, 벡터 안전성, 도식 기준 버전 대비
실행 가능한 노트북 셀의 보존 여부를 검사합니다.
