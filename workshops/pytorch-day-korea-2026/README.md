# PyTorch Day Korea 2026 · PhysicsNeMo 실습

2026년 11월 21일, 2시간 강의용 한국어 교재입니다. PINN으로 하나의 방정식의 해를 구하고, FNO로 입력장과 해 사이의 관계를 학습합니다.

강사가 원리와 결과를 설명합니다. 참가자는 준비된 Brev Jupyter에서 코드를 실행하고, 평소 사용하는 Gemini 또는 ChatGPT에 짧은 코드를 요청해 직접 붙여 넣습니다. AI 계정이나 API를 Brev에 연결하지 않습니다.

## 시작하기

1. 강사가 안내한 Brev 환경의 Jupyter를 엽니다.
2. [00_Start_Here.ipynb](00_Start_Here.ipynb)에서 커널과 GPU를 확인합니다.
3. [01_PINN.ipynb](01_PINN.ipynb)를 실행하고, AI가 작성한 그래프 코드로 결과를 그립니다.
4. [02_FNO.ipynb](02_FNO.ipynb)의 두 함수를 AI와 함께 완성하고 학습 결과를 확인합니다.

각 노트북을 위에서 아래로 진행하세요. 빈 코드 셀에는 AI의 답변을 검토한 뒤 직접 붙여 넣습니다. 실행이 막히면 노트북 마지막의 선택용 강사 예제를 사용할 수 있습니다.

## 120분 구성

| 시간 | 내용 |
| --- | --- |
| 0–10분 | Brev와 Jupyter 준비 |
| 10–35분 | PINN 원리 |
| 35–55분 | PINN 실행과 그래프 |
| 55–75분 | Neural Operator와 FNO 원리 |
| 75–105분 | FNO 두 함수 구현과 실행 |
| 105–120분 | 결과 비교와 질문 |

## 실습의 두 문제

| | PINN | FNO |
| --- | --- | --- |
| 입력 → 출력 | 좌표 x → 해 u(x) | forcing field f → solution field u |
| 문제 | u″=1, u(0)=u(1)=0 | u−Δu=f, 주기적 단위 정사각형 |
| 학습 신호 | 방정식 잔차와 경계조건 | 정답장과 예측장의 데이터 오차 |
| 결과 확인 | 해석해와 오차, 방정식 잔차, 경계 오차 | 미사용 test 데이터의 상대 L2와 방정식 잔차 |
| AI에 요청할 코드 | 이미 계산한 배열의 그래프 | 데이터셋과 FNO 모델을 만드는 두 함수 |

PINN은 1,000회 L-BFGS 호출, FNO는 3,000회 Adam update를 사용합니다. FNO의 PDE residual은 평가 지표이며 이 Level 1의 학습 loss에 추가되지 않습니다. 학습 실행 완료와 정확도 검증은 구분합니다.

Python 기본 문법과 배열을 읽을 수 있으면 따라올 수 있습니다. PhysicsNeMo 선수 경험은 필요하지 않습니다.

운영 준비는 [instructor/README.md](instructor/README.md), 환경 기준은 [environment/README.md](environment/README.md), 원본과 변경 범위는 [NOTICE.md](NOTICE.md)에 있습니다.
