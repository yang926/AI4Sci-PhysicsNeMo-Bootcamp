# 강사 준비 안내

## 기준 환경과 검증 상태

이 초안의 기준 컨테이너는 `nvcr.io/nvidia/physicsnemo/physicsnemo:25.11`입니다. 기존 교재가 사용하는 `physicsnemo.sym`의 `Domain`, `Solver`, `Key`, geometry API를 유지합니다.

2026-09-14 확인한 [PhysicsNeMo 최신 GitHub 릴리스](https://github.com/NVIDIA/physicsnemo/releases/tag/v2.2.2)는 v2.2.2입니다. 25.11은 컨테이너 릴리스 표기이고 v2.2.2는 Python 프로젝트 버전이므로 숫자를 직접 비교하지 않습니다. [공식 이식 안내](https://github.com/NVIDIA/physicsnemo/blob/main/v2.0-MIGRATION-GUIDE.md)에는 기존 Sym 고수준 API의 큰 변경이 있습니다. 행사 직전에 `pip install -U`로 패키지를 바꾸면 기존 교재와 맞지 않을 수 있습니다.

| 확인 항목 | 상태 |
|---|---|
| 원본 코드·노트북·강의안 대조 | 검토 완료 |
| 새 파동 해석해의 초기·경계조건 및 PDE 수치 검증 | CPU 검사 통과 |
| 새 Python 구문, 노트북 구조, 링크 검사 | 저장소의 검증 결과 확인 |
| 컨테이너 실제 빌드 및 이미지 digest 고정 | 확인 필요 |
| 행사 GPU에서 투사체·새 파동 학습 및 재시작 | 확인 필요 |
| 기본 설정의 학습 시간·오차·최대 GPU 메모리 | 확인 필요 |
| Brev 조직 접근, Launchable, 학생 계정, 브라우저 연결 | 확인 필요 |
| 강사 대체 결과와 참가자 실습 리허설 | 확인 필요 |

## 컨테이너 준비

행사 운영용 GPU 머신에서 저장소를 준비한 후 실행합니다. Dockerfile은 교재를 `/workspace/ai4sci`에 포함하며, Jupyter 기본 토큰 인증을 유지합니다. 컨테이너 안에 별도로 원본을 clone할 필요가 없습니다.

```bash
docker build -t ai4sci-physicsnemo:rehearsal .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 127.0.0.1:8888:8888 -it ai4sci-physicsnemo:rehearsal
```

로그에 표시된 인증 링크로 접속합니다. 원격 머신에서는 해당 환경의 인증 프록시 또는 승인된 SSH 터널을 사용합니다. 강의 당일에는 운영자가 미리 배포한 Brev 접속 경로를 참가자에게 제공합니다. 이 저장소만으로 Launchable이나 GPU가 생성되지는 않습니다.

Jupyter 첫 화면은 `ai4sci/README.md`입니다. Markdown 원문 편집기로 열리면 해당 파일을 우클릭해 **Open With → Markdown Preview**를 선택하고 첫 링크인 환경 확인 노트북을 엽니다. 리허설에서는 브라우저에서 이 동작과 링크 이동을 실제 확인합니다.

위 간단한 명령은 리허설용입니다. 참가자 배포 전 영구 저장소를 연결해 컨테이너 교체 후에도 결과가 남도록 하고, 통과한 이미지 digest와 패키지 목록을 기록합니다. 기본 이미지 태그만으로 추가 패키지의 버전까지 고정되지는 않습니다.

## 파동 실습 리허설

환경 확인 노트북을 마친 뒤 컨테이너 안 터미널에서 실행합니다.

```bash
cd /workspace/ai4sci/ai4sci/wave
python test_reference.py
python wave_baseline.py custom.run_name=smoke_01 training.max_steps=10
python wave_baseline.py custom.run_name=rehearsal_01 training.max_steps=1000
```

10회 반복은 import·설정·학습·결과 저장 경로를 확인하는 시험이며 정확도를 보장하지 않습니다. 1,000회 설정도 미측정 초안입니다. 실제 학습 시간과 오차를 보고 수업용 반복 횟수를 정합니다. `runs/<실험명>/metrics.json`과 `prediction.npz`가 생성되고 노트북이 그림을 표시하는지 확인합니다.

모든 실행에 새 `custom.run_name`을 부여합니다. 기존 결과 폴더가 있으면 코드는 덮어쓰지 않고 중단합니다. GPU 모델, VRAM, 이미지 digest, 패키지 버전, 반복 횟수, 경과 시간, 검증 RMSE, PDE 잔차를 리허설 기록에 남깁니다. 실제 측정 전에는 시간·정확도 목표를 참가자에게 약속하지 않습니다.

## 투사체 실습 리허설

`tutorial/projectile/Getting_Started_Projectile.ipynb`와 `source_code/projectile.py`를 사용합니다. 원본 설정은 5,000회 반복이고 0~5초를 학습한 뒤 최대 8초까지 추론합니다. 5초 이후 구간은 외삽임을 명시하고 검증 범위와 구분합니다. ParaView·TensorBoard를 동시에 처음 사용하게 하기보다 노트북 결과 해석부터 진행합니다. 강사는 필요한 파일·출력을 미리 확보합니다.

## 심화 자료의 범위

- 원본 Wave L1의 명시된 `c=1`과 검증 해가 불일치하므로 필수 실습에는 새 `01_Wave_PINN.ipynb`를 사용합니다.
- Neural Operator 자료는 Poisson·reaction-diffusion 문제식, 생성 코드, 데이터 경로가 일치하지 않습니다. 현 버전에서는 코드와 개념 설명용이며 실행형 필수 실습으로 사용하지 않습니다.
- `challenge/climate`는 단순화한 기후 PINN 예제입니다. FourCastNet 예측 실습으로 이름 붙이지 않습니다.
- MHD 전용 실습은 이 원본에서 확인되지 않았습니다. 추가 구현과 검증 전에는 강의안에서 참고 주제 또는 후속 학습으로 구분합니다.
- 모든 원본 `FIXME`를 수강생이 같은 속도로 해결한다고 가정하지 않습니다. 완성된 기본 실습을 먼저 마치고 시간이 남는 사람만 원본 과제로 확장합니다.

## 진행과 자료 연결

권장 편성은 [6시간·7시간 강의안](course-plan.md)입니다. 두 강사는 설명과 접속·오류 지원을 교대할 수 있으며, 실제 담당 배정은 별도로 확정합니다. 현재 PPT는 1시간 PINN 중심 자료이므로 슬라이드의 실습 링크·수식·실습 차원을 교재와 맞춘 뒤 배포합니다. 특히 PPT의 파동 예제는 1차원, 새 기본 실습은 2차원임을 설명하는 연결 슬라이드가 필요합니다.

리허설에서 저장한 결과를 준비하면 학습이 지연된 참가자도 예측과 해석해, 두 종류의 오차를 함께 해석할 수 있습니다. 강사 대체 결과는 실측 파일만 사용하고 예시 점수를 만들어 넣지 않습니다.
