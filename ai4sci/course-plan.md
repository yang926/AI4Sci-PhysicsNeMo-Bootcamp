# AI4Sci · PhysicsNeMo 강의 운영안

이 행사는 원본 GitHub의 전체 Lab·Challenge를 유지합니다. 학습 순서는 [원본 시작 노트북](../Start_Here.ipynb)을 기준으로 안내하고, 실제 파일과 실행 방법은 [한국어 참가자 안내](README.md)에서 확인합니다. 이전의 두 실습 중심 축소안은 철회했습니다. PPT는 사용자가 이 교재에 맞춰 수정합니다.

## 전체 학습 순서

원본 `Start_Here.ipynb`의 순서와 제목입니다. 각 Challenge의 모든 Level을 포함합니다.

| 순서 | 원본 과정 | 연결 자료 | 포함 내용 |
|---|---|---|---|
| 소개 | Getting started with PhysicsNeMo | [PhysicsNeMo 소개](../tutorial/introduction/Getting_Started_PhysicsNeMo.ipynb) | Physics-informed·data-driven 접근 |
| Lab 1 | Introduction to Physics-Informed Neural Networks (PINNs) | [PINN 소개](../tutorial/introduction/Introductory_Notebook.ipynb) | 신경망 해법, 매개변수 문제, 역문제 |
| Lab 2 | Solving ODEs with PhysicsNeMo | [투사체 운동](../tutorial/projectile/Getting_Started_Projectile.ipynb) | 투사체 ODE, 결과 시각화, ParaView |
| Lab 3 | From ODEs to PDEs - Diffusion Problems | [1D 확산](../tutorial/diffusion_1d/Diffusion_Problem_Notebook.ipynb) | 복합 막대의 정상 확산, 매개변수 확산 |
| Lab 4 | Advanced PDE Systems | [Navier–Stokes 기상 예제](../tutorial/navier_stokes/Weather-forecasting-navier-stokes.ipynb) | 해수면 기상 예측 예제, 데이터·학습·시각화 |
| Challenge 1 | Advanced Wave Dynamics | [파동](../challenge/wave/Advanced_Wave_Dynamics.ipynb) | Level 1 기본 2D 파동 → Level 2 가변 파동 속도 → Level 3 복잡한 경계·원형 영역 |
| Challenge 2 | Fluid-Structure Interaction | [유동](../challenge/fuild/Fluid_Structure_Interaction.ipynb) | Level 1 2D 유동 → Level 2 복수 블록 → Level 3 시간 의존 유동 |
| Challenge 3 | Multi-Physics Climate Modeling | [기후](../challenge/climate/Multi-Physics_Climate_Modeling.ipynb) | Level 1 단순 대기 → Level 2 대기·해양 결합 |
| Challenge 4 | Advanced Neural Operators | [신경 연산자](../challenge/neural_operator/Advanced_Neural_Operators.ipynb) | Level 1 FNO → Level 2 AFNO → Level 3 PINO |

원본 README의 `Bootcamp contents`에는 Challenge 2–4가 각각 Darcy/AFNO, FourCastNet, MHD/PINO로 기재되어 있습니다. 이 제목과 위 노트북 제목은 원본에서부터 다릅니다. 해당 이름을 그대로 옮긴 공유 강의안이 잘못 복사된 것은 아닙니다. 두 목록의 관계는 아래 강사 확인 사항으로 남기며, 다른 주제로 대체하거나 같은 구현이라고 임의 연결하지 않습니다.

## 7시간 시간표 · 공유 강의안 표기

아래는 [공유 강의안](https://docs.google.com/spreadsheets/d/1KS1z-Bmop8Jcn-PKLxuaCopveItRkmx2ImscphaLZfo/edit)의 2026-09-14 확인본에 있는 시각·제목을 유지한 표입니다. 표에서 비어 있던 10분 간격은 휴식으로 표시했습니다. 사전 등록은 09:30–10:00이며, 행사 전체 7시간은 점심·휴식을 포함한 10:00–17:00입니다.

| 시각 | 분 | 구분 | 공유 강의안 표기 |
|---|---:|---|---|
| 10:00–10:50 | 50 | 소개 | Introduction to NVIDIA PhysicsNeMo |
| 10:50–11:00 | 10 | 휴식 | 강의안의 빈 시간 |
| 11:00–12:00 | 60 | Lab | Training Labs: Fron PINN to PDE system problems |
| 12:00–13:00 | 60 | 점심 | Lunch |
| 13:00–13:50 | 50 | Challenge | Challenge 1: Advanced Wave Dynamics |
| 13:50–14:00 | 10 | 휴식 | 강의안의 빈 시간 |
| 14:00–14:50 | 50 | Challenge | Challenge 2: Solving the Darcy-Flow problem using AFNO |
| 14:50–15:00 | 10 | 휴식 | 강의안의 빈 시간 |
| 15:00–15:50 | 50 | Challenge | Challenge 3: Forecasting weather using FourCastNet |
| 15:50–16:00 | 10 | 휴식 | 강의안의 빈 시간 |
| 16:00–16:50 | 50 | Challenge | Challenge 4: Modeling Magnetohydrodynamics with Physics Informed Neural Operators |
| 16:50–17:00 | 10 | 정리 | Wrap-up |
| **합계** | **420** | **교육 320 / 점심 60 / 휴식 40** | **7시간** |

`Fron`은 공유 강의안에 있는 오탈자이며 `From`으로 고칠 수 있습니다. 공유 시트는 이번 작업에서 수정하지 않았습니다.

Lab·Challenge 배정은 총 260분입니다. 원본 README가 안내하는 Lab 120분 + Challenge 240분과는 시간 배정이 다릅니다. 이는 실습을 없애야 한다는 결론이 아니라, 전체 교재를 유지하면서 강사별 진행 분량과 실제 소요 시간을 확인할 항목입니다. 아직 측정하지 않은 실행 시간을 근거로 각 단계의 완주를 보장하지 않습니다.

## 6시간 운영

점심·휴식을 포함한 전체 6시간으로 진행할 경우의 시간 배정은 **확인 필요**입니다. 위 전체 Lab·Challenge와 11개 Level을 유지하며, 이 문서에서 일부를 생략하거나 시연으로 전환하는 시간표를 새로 만들지 않습니다.

## 강사 확인 사항

| 항목 | 확인할 내용 |
|---|---|
| README와 시작 노트북의 제목 | Challenge 2–4의 명칭과 연결 파일을 원본 관리 기준에 맞춰 확인. 확인 전에는 Darcy=Fluid, FourCastNet=Climate, MHD=Neural Operators라고 대응시키지 않음 |
| 두 강사의 역할 | Mingyu Yang·유영건의 실제 담당 구간과 실습 지원 역할은 협의 후 확정 |
| 전체 과정 리허설 | 소개·Lab 4개·Challenge 11개 Level의 설명, 파일 편집, 실행, 결과 확인에 걸리는 시간 측정 |
| PPT 연결 | 사용자가 GitHub의 실제 노트북·수식·실습 전환 순서에 맞춰 수정. 이 작업에서는 PPT를 편집하지 않음 |

수식·데이터 관련 확인 항목은 [강사 준비 안내](INSTRUCTOR.md)에 별도로 기록합니다. 이번 정비는 안내·오탈자·링크를 다루며, 원본 학습 Python 파일·YAML·노트북 코드 셀의 문제 정의와 구현은 유지합니다. 기존에 추가한 `ai4sci/01_Wave_PINN.ipynb`와 `ai4sci/wave/`는 개발 참고 자료로 보존하며 원본 파동 Challenge를 대체하지 않습니다.
