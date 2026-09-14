# Tutorial 학습 순서

[전체 수업 안내](../ai4sci/README.md) · [Start Here](../Start_Here.ipynb)

아래 순서는 원본 `Start_Here.ipynb`의 입문 및 Training Labs 순서입니다. 각 노트북의 처음과 끝에서도 이전·다음 수업으로 이동할 수 있습니다.

| 순서 | 노트북 | 진행 방식 |
|---|---|---|
| 1 | [PhysicsNeMo 소개](introduction/Getting_Started_PhysicsNeMo.ipynb) | 개념과 구성 요소 읽기 |
| 2 | [PINN 기초](introduction/Introductory_Notebook.ipynb) | PINN 손실·매개변수화·역문제 읽기 |
| 3 | [Projectile](projectile/Getting_Started_Projectile.ipynb) | 기존 Python 예제 실행 → 궤적 비교 → ParaView |
| 4 | [Diffusion](diffusion_1d/Diffusion_Problem_Notebook.ipynb) | 기본 확산 → 매개변수 확산 → 결과 비교 |
| 5 | [Navier–Stokes](navier_stokes/Weather-forecasting-navier-stokes.ipynb) | 데이터 준비 → 기존 Python 예제 실행 → 시간별 결과 확인 |

이어서 Challenge는 [Wave](../challenge/wave/Advanced_Wave_Dynamics.ipynb) → [Fluid](../challenge/fuild/Fluid_Structure_Interaction.ipynb) → [Climate](../challenge/climate/Multi-Physics_Climate_Modeling.ipynb) → [Neural Operators](../challenge/neural_operator/Advanced_Neural_Operators.ipynb) 순서로 확인합니다. 각 Challenge 안에서는 Level 순서대로 진행합니다.

## Python 파일을 여는 이유

노트북에는 문제 설명과 코드 예시, 실행 셀, 결과 확인 방법이 들어 있습니다. `.py`는 실행 셀이 실제로 시작하는 학습 프로그램이고, `conf/*.yaml`은 학습 설정입니다.

Tutorial은 이미 작성된 프로그램을 읽고 실행하는 방식입니다. Challenge는 해당 `.py` 파일의 `FIXME` 등 빈칸을 작성하는 방식입니다. JupyterLab 파일 편집기에서 파일을 열고 → 필요한 부분을 편집하고 → 저장한 다음 → 노트북의 기존 실행 셀을 실행합니다. 노트북의 설명용 코드 블록만 수정해도 `.py` 파일에 자동으로 반영되지는 않습니다.
