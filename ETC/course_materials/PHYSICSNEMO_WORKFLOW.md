# 교재 보조 함수에서 PhysicsNeMo까지

## Lab 4: 사전학습 기상 모델 추론

현재 [기상 Lab](../../01_labs/04_weather_forecasting/Lab_4_Weather_Forecasting.ipynb)은 `physicsnemo.models.afno.AFNO.from_checkpoint`와 `torch.inference_mode()`를 사용합니다. `PhysicsInformer`를 만들거나 PDE 손실을 계산하거나 가중치를 업데이트하지 않습니다. 공식 26채널 초기 상태를 체크포인트의 고정 통계로 정규화하고, 6시간 뒤를 예측한 다음 물리 단위로 되돌려 다음 단계 입력으로 사용합니다. 8단계를 거치면 48시간 예보가 됩니다. 미래 ERA5는 평가 코드만 읽으며, 같은 격자에서 지속성 예보도 평가합니다. 아래 설명은 PINN과 Challenge의 학습 과정에 관한 것으로, 현재 Lab 4의 추론 프로그램 설명이 아닙니다. 이전 PINN 유동 코드는 `ETC/reference_labs/04_navier_stokes`에 보존되어 있습니다.

[도입](../../01_Introduction.ipynb) · [Lab 1](../../01_labs/01_pinn/Lab_1_PINN_Fundamentals.ipynb) · [파동 Challenge](../../02_challenges/01_wave/Challenge_1_Wave_Dynamics.ipynb) · [신경 연산자](../../02_challenges/04_neural_operators/Challenge_4_Neural_Operators.ipynb)

이 교재는 PhysicsNeMo 2.2.2를 사용합니다. 한 실습은 모델, 방정식, 샘플링한 입력, PyTorch 학습 루프를 결합합니다. `ETC/runtime`의 짧은 보조 함수들이 이들을 연결합니다. 함수를 따라 읽으면 교재 코드와 라이브러리가 각각 무엇을 담당하는지 알 수 있습니다. 이 가이드는 현재 구현된 흐름을 설명합니다. 물리 문제, 학습 설정, 평가에 관한 최종 기준은 각 실습의 소스와 설정 파일입니다.

## 각 이름은 어디에서 왔을까요?

| 이름 | 제공하는 곳 | 역할 |
|---|---|---|
| `symbols`, `Function`, `.diff()` | SymPy | 좌표, 미지의 장, 미분을 기호로 표현합니다. 신경망을 계산하지는 않습니다. |
| `PDE` | `physicsnemo.sym.eq.pde` | 방정식 정의의 기본 클래스입니다. 실습에서 만든 하위 클래스가 이름을 붙인 잔차 식을 `self.equations`에 저장합니다. |
| `FullyConnected` | `physicsnemo.models.mlp` | 입력 텐서를 예측한 장의 값으로 바꾸는 PyTorch 호환 신경망입니다. |
| `PhysicsInformer` | `physicsnemo.sym.eq.phy_informer` | 장의 텐서와 선택한 공간 미분 방법으로 요청한 잔차를 계산합니다. |
| `create_model`, `create_informer`, `residuals` | 교재의 [ETC/runtime/pinn.py](../runtime/pinn.py) | Challenge 모델과 informer를 구성하고, 장을 계산하며, informer에 필요한 텐서를 준비합니다. PhysicsNeMo의 공개 API가 아니라 교재 보조 함수입니다. |
| `torch.autograd.grad`, `backward`, `torch.optim` | PyTorch | 텐서 연산을 미분하고 학습 가능한 매개변수를 업데이트합니다. |

예를 들어 `create_model(3, 1, config, device)`는 아래 라이브러리 호출을 감쌉니다. 다음은 실습의 `config`와 `device`를 사용하는 코드 발췌이며, 단독 실행 스크립트가 아닙니다.

```python
from physicsnemo.models.mlp import FullyConnected

model = FullyConnected(
    in_features=3,
    out_features=1,
    layer_size=config["model"]["width"],
    num_layers=config["model"]["layers"],
    activation_fn="tanh",
).to(device)
```

Wave Challenge에서는 입력 세 개가 `x`, `y`, `t`이고 출력 하나가 `u`입니다. 이 생성자만으로 신경망이 파동 방정식을 알게 되는 것은 아닙니다. 예측과 방정식을 연결하는 것은 손실입니다.

Challenge 보조 함수 `create_informer(pde, device)`는 방정식 매핑을 확인하고, 외부에서 제공할 입력으로 선언해야 하는 1차·2차 순수 시간 미분을 찾습니다. 이어서 [ETC/runtime/labs.py](../runtime/labs.py)의 `informer`에 처리를 맡깁니다. 이 함수는 `required_outputs=list(pde.equations)`, `equations=pde`, `grad_method="autodiff"`, `device`를 설정합니다. 공간 미분만 있는 방정식은 `PhysicsInformer`를 직접 만듭니다. 선언된 시간 미분이 있으면 교재의 `_SuppliedDerivativeInformer` 하위 클래스를 사용합니다. 이 클래스는 해당 텐서가 제공되었는지 검사하고 잔차 계산을 PhysicsInformer에 전달합니다. 이 어댑터가 모델을 학습하거나 PDE의 해를 구하는 것은 아닙니다.

## Wave Level 1의 학습 한 단계를 따라가기

[wave_l1.py](../../02_challenges/01_wave/wave_l1.py)와 [ETC/runtime/pinn.py](../runtime/pinn.py)를 나란히 여세요. 아래 이름들은 실제 파일의 함수명입니다. 발췌 코드는 읽기 위한 것이며, 과제 작성과 실행은 Challenge 노트북을 통해 진행하세요.

1. **기호 방정식을 작성합니다.** `WaveEquation2D(PDE)`가 SymPy 기호와 장을 만든 뒤 수강생의 `student_equations`와 `student_speed`를 호출합니다. 제공된 `c` 인자를 유지하면서 지정된 `wave` 키에 제시된 파동 PDE를 구현하세요. `.diff()`는 샘플 좌표나 신경망 값이 생기기 전에 기호 미분을 구성합니다. 완성된 과제 구현은 제공하지 않습니다.

2. **모델과 잔차 계산기를 만듭니다.** `main`에서 `WaveEquation2D()`가 수강생이 작성한 물리식을 구성하고, `create_informer`가 수치 잔차 계산을 준비하며, `create_model(3, 1, config, args.device)`가 신경망을 만듭니다. 고정된 별도 검증점에서의 잔차 검사도 같은 수강생 방정식을 사용하므로 정답을 보증하지 않습니다. 별도 채점기가 제시된 수학적 규약을 검사합니다.

3. **입력을 샘플링하고 장을 예측합니다.** `loss_terms`는 내부 위치 `xy`를 `[N, 2]`, 시간을 `[N, 1]` 모양으로 샘플링하고 `residuals(model, informer, xy, time, FIELD_NAMES)`를 호출합니다. 이 함수는 기울기 추적을 켠 새 좌표·시간 텐서를 만듭니다. `evaluate_fields`가 이들을 `[N, 3]`으로 합쳐 모델을 호출하고, `[N, 1]` 출력에 `"u"`라는 이름을 붙입니다. `FIELD_NAMES = ["u"]`가 이 텐서와 기호장의 이름을 연결합니다.

4. **예측값에 방정식을 적용합니다.** `residuals`는 예측한 `u`, 공간 `coordinates`, `x`, `y`, `t`, 별도로 계산한 `u__t`, `u__t__t` 텐서를 `informer.forward(inputs)`에 전달합니다. PhysicsInformer는 필요한 공간 미분을 자동 미분으로 구하고 이름이 지정된 식을 계산합니다. 반환 딕셔너리의 `"wave"`에는 각 점의 다음 잔차 텐서가 들어 있습니다.

   $$r_\theta(x,y,t)=u_{\theta,tt}-c^2(u_{\theta,xx}+u_{\theta,yy}).$$

5. **잔차와 조건을 하나의 손실값으로 만듭니다.** `loss_terms`는 `pde["wave"].square().mean()`과 함께 초기 변위, 초기 속도, 경계 손실을 계산합니다. Level 1의 두 초기 목표값은 모두 `sin(x) * sin(y)`이고 정사각형 가장자리의 변위는 0입니다. `record_step`은 이 네 스칼라 손실을 검사하고 합쳐 `total`을 반환합니다. 해석해는 비교에 사용하며 이 학습 루프의 목표값은 아닙니다.

6. **신경망을 업데이트합니다.** `main`의 해당 순서는 다음과 같습니다.

   ```python
   optimizer.zero_grad(set_to_none=True)
   losses = loss_terms(model, informer, config, args.device, exercise)
   total = record_step(step, losses, history)
   total.backward()
   optimizer.step()
   ```

   `total.backward()`는 모델 매개변수에 대한 기울기를 계산합니다. 이어서 Adam이 그 매개변수를 업데이트합니다. 다음 단계에서는 새 학습점을 샘플링해 예측과 손실 계산을 반복합니다.

전체 경로는 `student_equations` → `WaveEquation2D.equations` → `create_informer` → `residuals` / `informer.forward` → `loss_terms` → `total.backward()` → `optimizer.step()`입니다. 신경망은 `create_model`과 `evaluate_fields`를 통해 이 경로에 연결됩니다.

## 좌표에 대한 미분과 매개변수에 대한 기울기

여기에는 두 가지 미분이 있습니다. PDE를 계산할 때는 `u_xx`, `u_tt`처럼 좌표에 대해 예측값을 미분합니다. 신경망을 학습할 때는 가중치와 편향에 대해 스칼라 손실을 미분합니다. 두 번째 계산은 첫 번째 미분 연산을 거슬러 연결되어 있어야 합니다.

PINN Challenge에서 PhysicsInformer는 `grad_method="autodiff"`로 `x`, `y` 공간 미분을 담당합니다. 교재의 `gradient` 함수는 `torch.autograd.grad(..., create_graph=True)`를 호출해 시간 미분을 별도로 계산합니다. 시간이 있으면 `residuals`는 예측한 각 장의 1차·2차 시간 미분을 구하고 `u__t`, `u__t__t` 같은 이름으로 제공합니다. `create_informer`는 방정식에서 실제로 사용하는 시간 미분만 선언합니다. 신경망 입력에서 시간을 `x`, `y`와 합쳤다고 해서 informer가 시간을 자동 처리하는 공간 축으로 인식하는 것은 아닙니다. 이는 [시간 미분에 관한 PhysicsInformer 공식 안내](https://docs.nvidia.com/physicsnemo/latest/user-guide/physics_addition.html#adding-pde-losses)를 따릅니다.

`create_graph=True`는 미분 계산 자체의 미분 가능한 그래프를 유지합니다. 따라서 `total.backward()`가 잔차 손실을 모델까지 역전파할 수 있습니다. 좌표 기울기를 켜는 것은 Adam에게 샘플 위치를 움직이라는 뜻이 아닙니다. optimizer에는 `model.parameters()`를 전달했기 때문입니다.

방정식 잔차와 해의 오차도 다릅니다. `u_tt - c²(u_xx + u_yy)`가 작다는 것은 검사한 점에서 예측장이 방정식을 대략 만족한다는 뜻입니다. 의도한 해와 모든 위치에서 가깝다는 뜻은 아닙니다. 예를 들어 `u = 0`은 이 동차 파동 방정식의 잔차가 0이지만, Level 1의 0이 아닌 초기 변위와 속도를 위반합니다. 조건도 확인하고, 알려진 해가 있으면 그 해와의 오차도 보세요. Challenge의 고정 검증점 검사는 로컬 피드백이며 노트북의 제출·채점 절차를 대신하지 않습니다.

## 학습 루프를 직접 작성하는 이유

`zero_grad`, 스칼라 손실, `backward`, `step`을 작성하는 것도 PhysicsNeMo 사용 방식의 일부입니다. [공식 물리 기반 학습 튜토리얼](https://docs.nvidia.com/physicsnemo/latest/user-guide/physics_addition.html)은 이런 조합을 보여 주고, [2.2.2 마이그레이션 가이드](https://github.com/NVIDIA/physicsnemo/blob/v2.2.2/v2.0-MIGRATION-GUIDE.md)는 이전 `Solver` / `Domain` 방식에서 직접 작성하는 PyTorch 예제로 안내합니다. PhysicsNeMo는 재사용 가능한 모델과 잔차 도구를 제공하며, 학습 스크립트는 샘플, 조건, 손실 가중치, optimizer, 평가 절차를 선택합니다.

Lab도 다른 코드로 같은 역할 분담을 보여 줍니다. `mlp(nin, nout, cfg)`, `informer(pde, device, supplied_derivatives=...)`는 [ETC/runtime/labs.py](../runtime/labs.py)에 있으며, 설정에는 `layer_size`, `num_layers` 같은 키를 사용합니다. 반면 PINN Challenge의 `create_model(inputs, outputs, config, device)`는 `config["model"]["width"]`, `config["model"]["layers"]`를 읽습니다. 함수 인자와 설정 구조는 서로 바꿔 쓸 수 없습니다. Lab 1은 신경망을 `BasicPINN`으로 감싸며, 역문제 모드에서는 실습의 `FourierMLP`를 사용합니다. 선택한 모드는 [pinn_basics.py](../../01_labs/01_pinn/source_code/pinn_basics.py)에서 확인하세요.

각 실습의 기존 optimizer 설정을 따르세요. Lab 1의 L-BFGS는 `optimizer.step(closure)`를 사용하고, closure는 고정 학습점에서 손실을 다시 계산하고 `backward`를 호출합니다. 위 Wave 발췌는 Adam을 보여 줍니다. 둘 다 PyTorch로 모델을 최적화하며, 루프 형태의 차이는 선택한 optimizer에 따른 것입니다.

## Challenge 4에서는 입력과 미분이 어떻게 달라질까요?

[Challenge 4](../../02_challenges/04_neural_operators/Challenge_4_Neural_Operators.ipynb)의 입력은 외력장 전체이고 출력은 해의 장 전체입니다. 모양은 각각 `[batch, 1, n, n]`입니다. Level 1은 `physicsnemo.models.fno.FNO`, Level 2는 `physicsnemo.models.afno.AFNO`를 만듭니다. Level 3은 FNO에 물리 손실을 더해 물리 정보 기반 신경 연산자(PINO) 학습을 합니다. FNO와 AFNO는 모델 구조의 이름이고, PINO는 학습 목적함수에 물리를 사용하는 방식을 가리킵니다.

[fno_physicsnemo_l3.py](../../02_challenges/04_neural_operators/fno_physicsnemo_l3.py)의 기호식 `ReactionDiffusionPDE`와 `build_physics`를 읽어 보세요. `[0, 1)²`의 주기 방정식 `u - Δu = f`에 대해 PhysicsInformer를 `grad_method="spectral"`, `bounds=[1.0, 1.0]`로 설정합니다. 여기서는 주기 격자의 푸리에 표현으로 공간 미분을 구합니다. 교재의 `BatchedPhysicsInformer` 어댑터가 각 샘플에 informer를 별도로 적용하면서 샘플별 autograd 그래프를 유지합니다. 이 경로는 좌표 기반 PINN의 `residuals` 함수를 사용하지 않습니다.

[operator_training.py](../../02_challenges/04_neural_operators/operator_training.py)의 공통 루프는 값의 스케일을 명시적으로 구분합니다.

| 수량 | 스케일과 용도 |
|---|---|
| 모델 입력과 출력 | 정규화한 외력과 예측 해입니다. 평균과 표준편차는 학습 분할에서 계산합니다. |
| `data_loss` | 정규화한 예측과 목표값의 평균 제곱 오차입니다. |
| PDE 입력 | `u - Δu - f`를 계산하기 전에 예측과 외력을 물리 스케일로 되돌립니다. |
| `physics_loss` | 물리 잔차를 학습 외력의 표준편차로 나눈 값의 평균 제곱: `(residual / stats["f_std"]).square().mean()`. |
| 총손실 | `data_loss + physics_weight * physics_loss`이며, `loss.backward()`로 미분합니다. |

독립적으로 정규화한 `u`와 `f`에 원래 PDE를 그대로 적용하면, 스케일 인자와 오프셋을 고려하지 않는 한 방정식이 바뀝니다. 교재는 먼저 장을 물리 스케일로 복원한 뒤 잔차 손실의 크기를 조절합니다. 스펙트럴 미분도 FNO 예측에 대해 미분 가능하므로, 물리항은 데이터항과 같은 모델 매개변수를 업데이트할 수 있습니다.

다른 실습을 읽을 때는 모델 생성자, `PDE.equations`, `PhysicsInformer.forward`에 전달하는 이름 붙은 텐서, 스칼라 손실, optimizer 업데이트의 다섯 곳을 찾으세요. 교재 보조 함수가 더 짧은 이름으로 감싸 놓았더라도, 이 다섯 곳을 보면 라이브러리가 연결되는 방식을 알 수 있습니다.
