# AI4Sci · PhysicsNeMo 전체 실습 안내

PhysicsNeMo 소개부터 Training Lab 1–4, Challenge 1–4까지 차례로 진행합니다. 실습 순서는 원본 [시작 노트북](../Start_Here.ipynb)에 연결된 파일을 기준으로 합니다.

**처음에는 [환경 확인 노트북](00_환경확인.ipynb)을 엽니다.** JupyterLab의 파일 목록에서 이 README를 우클릭하고 **Open With → Markdown Preview**를 선택하면 안내와 링크를 읽기 편합니다.

## 전체 진행 순서

| 순서 | 원본 실습 | 열 노트북 | 학습 목표와 확인할 결과 |
|---|---|---|---|
| 소개 | Getting Started with PhysicsNeMo | [PhysicsNeMo 소개](../tutorial/introduction/Getting_Started_PhysicsNeMo.ipynb) | 물리 기반 학습과 데이터 기반 학습의 차이 이해 |
| Lab 1 | Introduction to Physics-Informed Neural Networks | [PINN 기초](../tutorial/introduction/Introductory_Notebook.ipynb) | 신경망의 입력·출력, PDE 잔차, 매개변수 문제와 역문제 구분 |
| Lab 2 | Solving ODEs with PhysicsNeMo | [투사체 운동](../tutorial/projectile/Getting_Started_Projectile.ipynb) | 초기조건과 운동 방정식을 코드에 연결하고 예측·해석해 비교, ParaView 사용 |
| Lab 3 | From ODEs to PDEs — Diffusion Problems | [1차원 확산](../tutorial/diffusion_1d/Diffusion_Problem_Notebook.ipynb) | 복합 막대의 경계·접합 조건과 매개변수화된 문제 실행 |
| Lab 4 | Advanced PDE Systems | [Navier–Stokes 예제](../tutorial/navier_stokes/Weather-forecasting-navier-stokes.ipynb) | 데이터 준비, 유동 방정식, 학습·시각화 연결 |
| Challenge 1 | Advanced Wave Dynamics | [파동 챌린지](../challenge/wave/Advanced_Wave_Dynamics.ipynb) | Level 1–3: 기본 파동, 가변 속도, 복잡한 경계 |
| Challenge 2 | Fluid-Structure Interaction | [유동 챌린지](../challenge/fuild/Fluid_Structure_Interaction.ipynb) | Level 1–3: 기본 유동, 복수 블록, 시간에 따라 변하는 유동 |
| Challenge 3 | Multi-Physics Climate Modeling | [기후 챌린지](../challenge/climate/Multi-Physics_Climate_Modeling.ipynb) | Level 1–2: 단순 대기와 대기·해양 결합 모형 |
| Challenge 4 | Advanced Neural Operators | [신경 연산자 챌린지](../challenge/neural_operator/Advanced_Neural_Operators.ipynb) | Level 1–3: FNO, AFNO, PINO의 데이터·모델·물리 손실 |

위 표는 실제 노트북의 순서와 제목입니다. [저장소 첫 화면](../README.md)의 Bootcamp contents에 실린 Darcy·FourCastNet·MHD 주제명도 유지되어 있으며, 강의 준비 시 확인할 명칭·파일 대응은 [강사 안내](INSTRUCTOR.md)에 정리되어 있습니다.

## 노트북과 Python 파일의 역할

| 파일 | 역할 | 실습할 때 하는 일 |
|---|---|---|
| `.ipynb` | 문제 설명, 수식, 코드 예시, 실행 명령, 그래프 | 위에서 아래로 읽고 지정한 실행 셀을 실행 |
| `.py` | 실제 방정식·모델·조건·학습·평가 프로그램 | 기초 Lab에서는 코드 읽기, Challenge에서는 `FIXME` 완성 후 저장 |
| `conf/*.yaml` | 신경망·학습 반복 횟수·배치 등 설정 | 본문과 강사가 안내한 설정 항목 확인 |

노트북 본문에 표시된 코드 블록은 설명용 텍스트일 수 있습니다. **설명용 코드 블록을 편집해도 옆의 `.py` 파일에는 반영되지 않습니다.** 실행 셀에 `!python wave_l1.py`라고 적혀 있으면 실제로 실행되는 파일은 `wave_l1.py`입니다. `!`는 노트북에서 터미널 명령을 실행한다는 뜻입니다.

이 교재는 긴 학습 프로그램을 `.py`에 두고 설명·그래프를 노트북에 두는 방식으로 구성되어 있습니다. JupyterLab 안에서 두 파일을 나란히 열면 편집과 실행을 한 화면에서 할 수 있습니다.

## 한 Level을 진행하는 방법

1. **노트북 설명을 읽습니다.** 이번 Level의 방정식, 정의역, 초기조건·경계조건과 구현할 항목을 확인합니다.
2. **실제 Python 파일을 엽니다.** 아래 표의 링크 또는 JupyterLab 파일 목록에서 해당 `.py`를 열고 노트북 옆에 둡니다. GitHub 웹 화면에서 수정하는 것이 아니라 행사 JupyterLab의 작업 파일을 수정합니다.
3. **과제를 완성하고 저장합니다.** Challenge는 파일 안에서 `FIXME`를 찾아 노트북의 힌트에 맞게 채웁니다. **Ctrl+S 또는 ⌘S**로 저장합니다. 기초 Lab은 준비된 코드를 먼저 읽고 실행합니다.
4. **노트북으로 돌아와 실행합니다.** 현재 Level의 `!python ...` 셀을 **Shift+Enter**로 실행합니다. 학습 로그가 출력되는지 확인합니다.
5. **출력을 확인합니다.** 실행이 끝나면 다음 평가·그래프 셀을 진행합니다. Challenge는 표시된 지표와 `challenge/leaderboard_metrics.csv`의 새 기록을 확인합니다. 이 CSV는 현재 작업 환경에 저장되는 결과 파일입니다.
6. **다음 Level로 이동합니다.** 오류가 있으면 해당 셀의 마지막 오류를 해결하고 다시 실행합니다. 그 뒤 같은 노트북의 다음 Level 또는 맨 아래의 다음 실습 링크로 이동합니다.

노트북 작업 폴더는 해당 노트북이 있는 폴더입니다. 실행 파일을 찾지 못하면 먼저 `%pwd`로 확인합니다. 강사 안내 없이 패키지 설치나 데이터 덮어쓰기를 반복하기보다, 현재 파일·폴더·오류 문장을 함께 확인합니다.

## Training Lab 실행 파일

| Lab | 읽고 실행할 파일 | 노트북에서 확인할 결과 |
|---|---|---|
| Lab 1 | [PINN 기초 노트북](../tutorial/introduction/Introductory_Notebook.ipynb)의 설명 | 순방향·매개변수·역문제의 입력과 학습 대상 구분 |
| Lab 2 | [projectile.py](../tutorial/projectile/source_code/projectile.py), [projectile_eqn.py](../tutorial/projectile/source_code/projectile_eqn.py) | 위치 예측과 해석해 그래프, ParaView 출력 |
| Lab 3 | [diffusion_bar.py](../tutorial/diffusion_1d/source_code/diffusion_bar.py), [diffusion_bar_parameterized.py](../tutorial/diffusion_1d/source_code/diffusion_bar_parameterized.py) | 두 구간의 해와 접합 조건, 매개변수화 결과 |
| Lab 4 | [navier_stokes.py](../tutorial/navier_stokes/source_code/navier_stokes.py) | 본문의 데이터 준비 결과와 유동 시각화 |

## Challenge별 편집 파일

각 행은 별도의 Level입니다. 실제 실행 명령은 해당 노트북의 기존 셀을 사용합니다.

| Challenge | Level | 편집할 Python 파일 | 설정 파일 |
|---|---|---|---|
| Wave | 1 | [wave_l1.py](../challenge/wave/wave_l1.py) | [config_wave.yaml](../challenge/wave/conf/config_wave.yaml) |
| Wave | 2 | [wave_l2.py](../challenge/wave/wave_l2.py) | [config_wave.yaml](../challenge/wave/conf/config_wave.yaml) |
| Wave | 3 | [wave_l3.py](../challenge/wave/wave_l3.py) | [config_wave.yaml](../challenge/wave/conf/config_wave.yaml) |
| Fluid | 1 | [chip_2d_l1.py](../challenge/fuild/chip_2d_l1.py) | [config_chip_2d.yaml](../challenge/fuild/conf/config_chip_2d.yaml) |
| Fluid | 2 | [chip_2d_l2.py](../challenge/fuild/chip_2d_l2.py) | [config_chip_2d.yaml](../challenge/fuild/conf/config_chip_2d.yaml) |
| Fluid | 3 | [chip_2d_l3.py](../challenge/fuild/chip_2d_l3.py) | [config_chip_2d.yaml](../challenge/fuild/conf/config_chip_2d.yaml) |
| Climate | 1 | [climate_l1.py](../challenge/climate/climate_l1.py) | [config_atmos.yaml](../challenge/climate/conf/config_atmos.yaml) |
| Climate | 2 | [climate_l2.py](../challenge/climate/climate_l2.py) | [config_coupled.yaml](../challenge/climate/conf/config_coupled.yaml) |
| Neural Operators | 1 · FNO | [fno_physicsnemo_l1.py](../challenge/neural_operator/fno_physicsnemo_l1.py) | [config_FNO.yaml](../challenge/neural_operator/conf/config_FNO.yaml) |
| Neural Operators | 2 · AFNO | [fno_physicsnemo_l2.py](../challenge/neural_operator/fno_physicsnemo_l2.py) | [config_AFNO.yaml](../challenge/neural_operator/conf/config_AFNO.yaml) |
| Neural Operators | 3 · PINO | [fno_physicsnemo_l3.py](../challenge/neural_operator/fno_physicsnemo_l3.py) | [config_PINO.yaml](../challenge/neural_operator/conf/config_PINO.yaml) |

Neural Operators 노트북에는 모델 학습 전 [generate_data.py](../challenge/neural_operator/generate_data.py)를 실행하는 단계도 있습니다. 데이터 준비 순서는 노트북의 **Step 0: Data Generation**을 따릅니다.

## 막혔을 때

| 증상 | 확인할 항목 |
|---|---|
| `NameError: name 'FIXME' is not defined` | 실행한 `.py`에 과제 빈칸이 남았는지 확인 |
| 코드를 고쳤는데 같은 오류가 남음 | 노트북 설명용 코드만 바꾸었는지, 실제 `.py`를 저장했는지 확인 |
| `can't open file` 또는 상대 경로 오류 | `%pwd`로 노트북 작업 폴더와 파일 위치 확인 |
| 모듈 import 또는 CUDA 오류 | [환경 확인](00_환경확인.ipynb)의 결과와 행사 커널 확인 |
| 학습 오류 뒤 예전 그래프가 보임 | 이번 실행이 정상 종료했고 새 출력이 생성됐는지 확인 |
| 그림·데이터 파일이 없음 | 파일이 저장소에 포함된 자료인지, 앞 단계에서 생성·다운로드되는 파일인지 본문에서 확인 |

[처음 실습 시작](../tutorial/introduction/Getting_Started_PhysicsNeMo.ipynb) · [원래 강의 시간표](course-plan.md) · [강사 안내](INSTRUCTOR.md)

## 출처

[OpenHackathons AI-Powered-Physics-Bootcamp](https://github.com/openhackathons-org/AI-Powered-Physics-Bootcamp)의 전체 과정과 원본 저작권·라이선스 고지를 유지합니다. [라이선스](../LICENSE)
