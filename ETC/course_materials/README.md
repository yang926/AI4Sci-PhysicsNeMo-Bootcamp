<a id="instructor-course-and-code-index"></a>
# 강사용 강좌 및 코드 안내

이 안내에서 수업 프로그램을 찾고 진행 중 발생한 문제를 해결하세요. 학생은 [시작하기](../../Start_Here.ipynb)를 따라 진행하며, 수업 전에 이 안내를 읽을 필요는 없습니다.

PhysicsNeMo 2.2.2를 사용하고 JupyterLab에서 수업을 실행하세요. 연습 중에는 노트북과 연결된 Python 파일을 나란히 열어 두세요.

[설치](../environment/SETUP.md) · [평가 안내](ASSESSMENT.md) · [검증 기록](VALIDATION.md) · [시작하기](../../Start_Here.ipynb)

<a id="understand-the-physicsnemo-workflow"></a>
## PhysicsNeMo 작업 흐름 이해하기

[소개](../../01_Introduction.ipynb)부터 시작한 뒤 수업의 Python 파일을 읽으며 [보조 함수에서 API까지 따라가기](PHYSICSNEMO_WORKFLOW.md)를 참고하세요. 도전 과제 1–3에서 `create_model`, `create_informer`, `residuals`는 강좌 보조 함수입니다. 이들은 PhysicsNeMo 모델과 PDE 잔차 평가를 명시적인 PyTorch 학습 반복문에 연결합니다. 실습과 신경 연산자는 각자의 래퍼를 사용하며, 해당 노트북에 설명되어 있습니다.

PINN 연습 문제를 채우기 전에 모델의 입력과 출력, 요구되는 잔차, 조건 손실, 최적화기 갱신을 파악하세요. 올바른 방정식은 이 작업 흐름의 한 부분입니다. 보조 함수를 실행하는 것과 그 역할을 설명하는 것은 다릅니다. 실습 4는 별도의 추론 흐름입니다. 사전 학습된 AFNO 가중치를 불러오고 날씨 상태를 전진시키며, 가중치를 갱신하지 않고 재분석 자료와 비교합니다.

<a id="your-first-training-problem"></a>
## 첫 번째 학습 문제

[환경 확인](../../00_Setup.ipynb)을 실행하고 소개를 읽은 뒤 실습 1을 시작하세요.

첫 번째 학습 문제는 [실습 1.1: 순방향 PINN](../../01_labs/01_pinn/Lab_1_PINN_Fundamentals.ipynb#first-problem)입니다. $[0,1]$에서 $u''(x)=1$과 $u(0)=u(1)=0$을 만족하는 $u(x)$를 학습합니다.

[순방향 PINN 실행](../../01_labs/01_pinn/Lab_1_PINN_Fundamentals.ipynb#forward-pinn-execution) 아래에서 설정, 학습, 그래프 그리기를 순서대로 실행하세요. `analytical`과 `PINN` 곡선을 비교한 뒤 실습 1.2(매개변수화)와 실습 1.3(역문제)으로 진행하세요. 실습 프로그램은 완성되어 있으므로 채워야 할 함수가 없습니다.

<a id="complete-course-sequence"></a>
## 전체 강좌 순서

| 순서 | 주제 | 노트북 | 학습 목표 |
|---|---|---|---|
| 소개 | PhysicsNeMo 소개 | [노트북 열기](../../01_Introduction.ipynb) | 물리 정보 기반 학습과 데이터 기반 학습을 읽고 이해합니다. 실행할 코드 셀은 없습니다. |
| 실습 1 | PINN 기초 | [노트북 열기](../../01_labs/01_pinn/Lab_1_PINN_Fundamentals.ipynb) | 첫 실행 수업: 1.1 순방향, 1.2 매개변수화, 1.3 역문제 PINN. |
| 실습 2 | 포물체 운동 ODE | [노트북 열기](../../01_labs/02_projectile/Lab_2_Projectile_Motion.ipynb) | 초기 조건과 ODE 잔차를 연결하고 해석적 궤적과 비교합니다. |
| 실습 3 | 정상 열전도 | [노트북 열기](../../01_labs/03_heat_conduction/Lab_3_Heat_Conduction.ipynb) | 두 재료로 이루어진 막대 문제를 풀고 계면 온도와 열유속을 검증합니다. |
| 실습 4 | FourCastNet을 활용한 AI 일기 예보 | [노트북 열기](../../01_labs/04_weather_forecasting/Lab_4_Weather_Forecasting.ipynb) | 사전 학습된 AFNO로 48시간 예보를 생성하고 예보 선행 시간에 따라 ERA5 재분석 및 지속성 예측과 비교합니다. |
| 도전 과제 1 | 파동 역학 | [노트북 열기](../../02_challenges/01_wave/Challenge_1_Wave_Dynamics.ipynb) | 레벨 1–3: 일정한 속도, 가변 속도, 원형 Robin 경계. |
| 도전 과제 2 | 유체 흐름 | [노트북 열기](../../02_challenges/02_fluid/Challenge_2_Fluid_Flow.ipynb) | 레벨 1–3: 단일 고정 장애물, 다중 고정 장애물, 시간 의존 흐름. |
| 도전 과제 3 | 교육용 기후 PDE | [노트북 열기](../../02_challenges/03_climate/Challenge_3_Climate_Modeling.ipynb) | 레벨 1–2: 온도 수송과 대기–해양 방정식. 레벨 2의 기본값은 원래의 비결합 경우(`gamma0=0`)이며, 0이 아닌 교환은 별도의 로컬 실험입니다. |
| 도전 과제 4 | 신경 연산자 | [노트북 열기](../../02_challenges/04_neural_operators/Challenge_4_Neural_Operators.ipynb) | 레벨 1–3: 동일한 주기적 반응–확산 문제에 FNO, AFNO, PINO 적용. |

강좌는 실습 네 개와 도전 과제의 레벨 열한 개로 구성됩니다. 실습 4는 기존 Navier–Stokes 수업을 대체하도록 명시적으로 승인된 것이며, 원래 부트캠프에 이 날씨 작업 흐름이 포함되었다는 뜻은 아닙니다. 기존 [PINN 실습](../reference_labs/04_navier_stokes/Lab_4_Navier_Stokes.ipynb)은 참고 자료로 유지됩니다. 이전 제목에 관한 배경은 [이전 안내](MIGRATION.md)에 있습니다.

<a id="notebooks-python-files-and-configuration"></a>
## 노트북, Python 파일, 설정

| 파일 | 용도 | 할 일 |
|---|---|---|
| `.ipynb` | 문제 설명, 방정식, 예제, 실행, 그래프 | 순서대로 읽고 실행 셀을 실행합니다. |
| `.py` | 방정식, 모델, 조건, 학습, 추론, 평가 | 완성된 실습을 읽고 도전 과제의 표시된 연습 코드를 채웁니다. |
| `conf/*.yaml` | 신경망, 배치 크기, 학습률, 학습 횟수 | 수업에서 지정한 설정을 확인하거나 조정합니다. |

Markdown 셀의 코드 예제는 `.py` 프로그램을 바꾸지 않습니다. Python 파일 자체를 수정하세요. 실행 셀은 현재 커널의 인터프리터로 해당 파일을 실행합니다.

도전 과제 1–3은 각 레벨의 표시된 함수를 모두 구현해야 합니다. `student_equations`와 `student_conditions`에 더해, 파동의 `student_speed`, 유체의 `student_geometry`, 기후의 `student_parameters`와 `student_solution`이 필요합니다. [과제 및 제출 대응표](CHALLENGE_CONTRACTS.md)를 확인하세요. 도전 과제 4는 대신 `build_datasets`, `build_model`, 레벨 3의 `ReactionDiffusionPDE`를 사용합니다. 이곳에서 이름이 문자 그대로 `student_*`인 함수를 찾지 말고 연결된 파일과 `FIXME` 표시를 따라가세요.

<a id="complete-one-level"></a>
## 레벨 하나 완료하기

1. PDE, 영역, 초기 및 경계 조건, 구현 과제를 읽습니다.
2. 행사 JupyterLab 작업 공간에서 연결된 `.py` 파일을 엽니다.
3. 도전 과제에서는 연결된 Python 파일의 모든 **EDIT HERE** 블록을 완성하고 **Ctrl+S** 또는 **Cmd+S**로 저장합니다. 도전 과제는 자신의 구현만 실행합니다. 실습 1–3은 완성된 예제로 유지됩니다.
4. 학습 전에 마지막 제출 패널 셀을 실행하고, 완료한 레벨만 선택한 뒤 **Check saved code**를 누릅니다. 이 검사는 학습하거나 채점기에 접속하지 않고 문법과 완성 여부를 확인합니다. 정확성은 제출 후 채점기가 평가합니다. 현재 학습 셀로 돌아와 **Shift+Enter**로 실행합니다.
5. 평가 및 그래프 셀을 실행합니다. 전후 비교 표에는 저장된 진단 결과가 요약됩니다. 전체 JSON은 **Full metrics and run settings**를 펼쳐 확인하세요. 같은 문제와 평가 설정에서만 시도 간 결과를 비교하세요. 이 오차는 연습 피드백이며 공식 순위 점수가 아닙니다.
6. 다음 레벨이나 다음 노트북으로 넘어가기 전에 오류를 해결합니다.

노트북은 보통 자신이 있는 디렉터리에서 시작합니다. 파일을 찾을 수 없으면 `%pwd`를 사용하세요. 패키지를 다시 설치하거나 데이터를 다시 생성하기 전에 파일 경로, 수정 내용의 저장 여부, 오류 메시지를 다시 확인하세요.

<a id="lab-programs"></a>
## 실습 프로그램

| 실습 | 프로그램 | 확인할 출력 |
|---|---|---|
| 1 | [pinn_basics.py](../../01_labs/01_pinn/source_code/pinn_basics.py): `forward`, `parameterized`, `inverse` | 학습한 해, 매개변수 의존성, 추정한 소스항. |
| 2 | [projectile.py](../../01_labs/02_projectile/source_code/projectile.py), [projectile_eqn.py](../../01_labs/02_projectile/source_code/projectile_eqn.py) | 예측 및 해석적 궤적, ParaView용 VTP 내보내기. |
| 3 | [diffusion_bar.py](../../01_labs/03_heat_conduction/source_code/diffusion_bar.py), [diffusion_bar_parameterized.py](../../01_labs/03_heat_conduction/source_code/diffusion_bar_parameterized.py) | 구간별 해, 계면 조건, 매개변수화 예측. |
| 4 | [run_forecast.py](../../01_labs/04_weather_forecasting/source_code/run_forecast.py), [evaluate_weather.py](../../01_labs/04_weather_forecasting/source_code/evaluate_weather.py) | 바람, 기압, 온도의 예보, ERA5 재분석, 오차와 애니메이션 및 선행 시간별 오차 곡선. |

실습 4는 NVIDIA의 사전 학습된 26채널 FourCastNet1 AFNO 체크포인트를 사용합니다. 2022년 9월 1일 00 UTC의 ERA5에서 시작하여 720×1440 전 지구 격자에서 6시간 예보 단계를 여덟 번 적용합니다. 미래 ERA5는 평가에만 사용합니다. 이 실습에는 최적화기, 학습 횟수, 채점기 제출이 없습니다. 하나의 과거 사례로 예보와 검증을 설명하며, 일반적인 날씨 예측 성능을 입증하지는 않습니다. [날씨 검증 기록](LAB4_WEATHER_VALIDATION.md)을 참고하세요.

최초 준비 시 모델 가중치 약 301 MB와 압축된 원본 데이터 622 MB를 다운로드합니다. 파일은 Git 체크아웃 밖의 `~/.cache/ai4sci/weather`에 캐시되며 검증 후 재사용됩니다. 수업 전에 준비 과정을 리허설하세요. 한 번의 다운로드 성공만으로 110명 동시 다운로드 용량이 확인되지는 않습니다.

<a id="challenge-programs"></a>
## 도전 과제 프로그램

| 도전 과제 | 레벨 | 수정할 파일 | 설정 |
|---|---|---|---|
| 파동 역학 | 1 | [wave_l1.py](../../02_challenges/01_wave/wave_l1.py) | [config_wave_l1.yaml](../../02_challenges/01_wave/conf/config_wave_l1.yaml) |
| 파동 역학 | 2 | [wave_l2.py](../../02_challenges/01_wave/wave_l2.py) | [config_wave.yaml](../../02_challenges/01_wave/conf/config_wave.yaml) |
| 파동 역학 | 3 | [wave_l3.py](../../02_challenges/01_wave/wave_l3.py) | [config_wave.yaml](../../02_challenges/01_wave/conf/config_wave.yaml) |
| 유체 흐름 | 1 | [chip_2d_l1.py](../../02_challenges/02_fluid/chip_2d_l1.py) | [config_chip_2d.yaml](../../02_challenges/02_fluid/conf/config_chip_2d.yaml) |
| 유체 흐름 | 2 | [chip_2d_l2.py](../../02_challenges/02_fluid/chip_2d_l2.py) | [config_chip_2d.yaml](../../02_challenges/02_fluid/conf/config_chip_2d.yaml) |
| 유체 흐름 | 3 | [chip_2d_l3.py](../../02_challenges/02_fluid/chip_2d_l3.py) | [config_chip_2d.yaml](../../02_challenges/02_fluid/conf/config_chip_2d.yaml) |
| 기후 모델링 | 1 | [climate_l1.py](../../02_challenges/03_climate/climate_l1.py) | [config_atmos.yaml](../../02_challenges/03_climate/conf/config_atmos.yaml) |
| 기후 모델링 | 2 | [climate_l2.py](../../02_challenges/03_climate/climate_l2.py) | [config_coupled.yaml](../../02_challenges/03_climate/conf/config_coupled.yaml) |
| 신경 연산자 | 1 | [fno_physicsnemo_l1.py](../../02_challenges/04_neural_operators/fno_physicsnemo_l1.py) | [config_FNO.yaml](../../02_challenges/04_neural_operators/conf/config_FNO.yaml) |
| 신경 연산자 | 2 | [fno_physicsnemo_l2.py](../../02_challenges/04_neural_operators/fno_physicsnemo_l2.py) | [config_AFNO.yaml](../../02_challenges/04_neural_operators/conf/config_AFNO.yaml) |
| 신경 연산자 | 3 | [fno_physicsnemo_l3.py](../../02_challenges/04_neural_operators/fno_physicsnemo_l3.py) | [config_PINO.yaml](../../02_challenges/04_neural_operators/conf/config_PINO.yaml) |

신경 연산자는 학습 전에 노트북의 데이터 생성 단계에 따라 [generate_data.py](../../02_challenges/04_neural_operators/generate_data.py)를 사용하세요.

<a id="results-and-reruns"></a>
## 결과와 재실행

실습 1–3과 도전 과제의 학습이 성공하면 `metrics.json`(평가), `loss.csv`(학습 기록), `model.pt`(가중치), `predictions.npz`(예측값), `preview.png`(그래프)를 저장합니다. 실습 4는 대신 `forecast.npz`, `runtime.json`과 지표, 지도, 애니메이션이 담긴 별도의 평가 디렉터리를 저장합니다. 매번 새로운 출력 디렉터리를 사용하세요. 기존 결과는 덮어쓰지 않습니다.

공통 [노트북 보조 함수](../runtime/notebook.py)는 학습 결과를 보여 주고 해당 실행의 결과인지 확인하며, 모델을 학습하거나 점수를 부여하지는 않습니다. 방정식은 수업 프로그램에 있고 실행 명령은 각 노트북에서 볼 수 있습니다. 실습 4는 자체 예보/평가 경로를 사용합니다. 학습 결과 표에는 위젯이 필요하지 않습니다. 별도의 제출 패널은 환경에 설치된 `ipywidgets` 패키지를 사용합니다.

노트북 학습 셀은 그 셀만 다시 실행하는 경우에도 매번 새로운 결과 디렉터리를 선택합니다. 실행에 성공한 뒤 이어지는 그래프/평가 셀을 실행하세요. 실패한 실행을 이전의 성공한 결과와 혼동하면 안 됩니다. 명령줄을 사용한다면 직접 새로운 `--output-dir`을 선택하고 선택된 학습 설정과 횟수를 확인하세요. 현재 노트북 기본값은 간단한 실행 확인용이 아니라 전체 수업 분량입니다. 노트북의 `STEPS` 값은 `--steps`로 전달되어 YAML 학습 횟수보다 우선합니다. `AI4SCI_STEPS`가 설정되어 있으면 노트북 기본값보다 우선합니다.

모든 도전 과제 실행에는 PDE만이 아니라 해당 레벨의 표시된 모든 함수가 필요합니다. 도전 과제 1–3의 기본값은 레벨당 Adam 갱신 5,000회이며, FNO, AFNO, PINO는 3,000회를 유지합니다. 이는 수업을 위해 제한한 실행량이며 수렴을 보장하지 않습니다. 로컬 PINN 잔차는 자신의 구현을 사용하고, 정확성은 채점기가 확인합니다. 파동 1은 독립적인 해석해 비교를, 유체 1은 제공된 OpenFOAM 비교를 유지합니다. 기후 비교에는 학습자가 유도한 `student_solution`을 사용합니다. 실습 4의 날씨 검증은 그대로 유지됩니다.

학생용 배포본에는 도전 과제의 완성된 정답 모드가 없습니다. 노트북 명령은 Python `-u`를 사용하므로 프로세스가 실행되는 동안 학습 진행 상황이 표시됩니다.

<a id="troubleshooting"></a>
## 문제 해결

| 증상 | 확인 사항 |
|---|---|
| `Complete student_...` 또는 `NotImplementedError` | 실제 `.py` 파일에서 연습 문제를 완성하고 저장하세요. |
| 수정 후에도 같은 오류가 발생함 | Markdown 예제가 아니라 Python 파일을 수정하고 저장했는지 확인하세요. |
| 학생의 수정이 반영되지 않음 | Python 파일 경로를 확인하고 수정 내용을 저장한 뒤 학습과 결과 셀을 다시 실행하세요. |
| 저장된 설정이 현재 실행과 다름 | 의도한 설정으로 다시 학습한 뒤 결과 셀을 다시 실행하세요. 이전 지표의 이름만 바꾸지 마세요. |
| 직접 입력한 명령에서 출력 디렉터리가 이미 존재한다고 표시됨 | 이전 결과를 보존하고 새로운 `--output-dir`을 선택하세요. 노트북 실행 셀은 이를 자동으로 처리합니다. |
| `can't open file` | `%pwd`와 프로그램 경로를 확인하세요. |
| import 또는 CUDA 오류 | 환경 확인을 실행하고 의도한 가상 환경 커널을 선택하세요. |
| 실패 후에도 이전 그래프가 남아 있음 | 현재 실행이 완료되어 새로운 결과 파일을 만들었는지 확인하세요. |
| 데이터나 그림이 없음 | 파일이 저장소에 포함되어 있는지, 앞 단계에서 생성되는지 확인하세요. |
| Markdown이 소스 텍스트로 열림 | 파일을 마우스 오른쪽 버튼으로 누르고 **Open With → Markdown Preview**를 선택하세요. |

[강좌 시작](../../01_Introduction.ipynb) · [일정](course-plan.md) · [강사 안내](INSTRUCTOR.md)

<a id="attribution"></a>
## 출처

[OpenHackathons AI-Powered-Physics-Bootcamp](https://github.com/openhackathons-org/AI-Powered-Physics-Bootcamp)를 바탕으로 수정했습니다. 원본의 출처 표시와 라이선스를 유지합니다. [라이선스](../../LICENSE).
