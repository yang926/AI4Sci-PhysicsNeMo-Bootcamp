# AI4Sci · 강사 준비 안내

[참가자 안내](README.md)와 [원본 시작 노트북](../Start_Here.ipynb)을 따라 전체 과정을 진행합니다. Lab 4개와 Challenge의 11개 Level을 유지하며, [강의 운영안](course-plan.md)에서 공유 강의안의 시간표와 아직 확인할 제목 연결을 구분합니다. PPT 편집은 사용자가 진행합니다.

## 노트북과 Python 파일의 역할

원본 교재는 노트북에서 개념·수식·단계별 코드를 설명하고, 별도 `.py` 파일로 학습을 실행하는 구조입니다. 투사체 노트북은 Hydra 설정을 사용하는 학습 코드를 별도 Python 프로세스로 실행하기 위한 구성을 설명합니다. `.py`는 이번에 새로 만든 노출본이 아니라 원본 실습의 실행 파일입니다.

노트북 Markdown 안의 코드 예시는 설명입니다. 예시를 수정해도 실행 파일은 바뀌지 않습니다. Challenge의 `FIXME`는 참가자가 채우는 의도된 과제이며, 오탈자로 보고 없애거나 정답으로 바꾸지 않습니다.

1. 해당 노트북에서 문제와 현재 Level의 설명을 읽습니다.
2. Jupyter 파일 목록에서 같은 폴더의 `.py` 파일을 엽니다. Lab은 각 노트북 폴더 아래 `source_code/`에 있습니다.
3. 노트북의 안내를 따라 해당 파일의 `FIXME`를 편집합니다. 기본 Lab은 준비된 코드를 읽고 실행하며, 설정 실험이 지시된 경우 해당 YAML을 확인합니다.
4. **파일을 저장한 뒤** 노트북으로 돌아옵니다.
5. 해당 Level의 실행 셀을 실행합니다. 예를 들어 Wave Level 1은 노트북이 있는 폴더에서 `!python wave_l1.py`를 호출합니다.
6. 로그와 노트북이 안내하는 결과를 확인한 뒤 다음 Level로 이동합니다. 실패하면 실행한 파일명·작업 폴더·저장 여부·남은 `FIXME`를 먼저 확인합니다.

터미널에서 실행할 때도 해당 노트북 폴더로 이동한 뒤 같은 스크립트를 실행합니다. 노트북의 코드 셀이 호출하는 파일과 참가자가 편집한 파일이 같은지 첫 실습에서 함께 확인합니다.

## 전체 노트북 확인표

원본 노트북 10개를 모두 확인합니다. 아래는 리허설 체크리스트이며 완료 기록이 아닙니다.

| 순서 | 원본 노트북 | 리허설 확인 |
|---|---|---|
| 시작 | [Start_Here.ipynb](../Start_Here.ipynb) | 전체 순서와 각 노트북 이동 |
| 소개 | [Getting_Started_PhysicsNeMo.ipynb](../tutorial/introduction/Getting_Started_PhysicsNeMo.ipynb) | 물리 기반·데이터 기반 접근 설명 |
| Lab 1 | [Introductory_Notebook.ipynb](../tutorial/introduction/Introductory_Notebook.ipynb) | PINN 해법, 매개변수 문제, 역문제 |
| Lab 2 | [Getting_Started_Projectile.ipynb](../tutorial/projectile/Getting_Started_Projectile.ipynb) | `source_code/projectile.py` 실행, 예측·기준값 비교, ParaView |
| Lab 3 | [Diffusion_Problem_Notebook.ipynb](../tutorial/diffusion_1d/Diffusion_Problem_Notebook.ipynb) | `source_code/diffusion_bar.py`와 `source_code/diffusion_bar_parameterized.py` 실행·결과 |
| Lab 4 | [Weather-forecasting-navier-stokes.ipynb](../tutorial/navier_stokes/Weather-forecasting-navier-stokes.ipynb) | 데이터 다운로드, `source_code/navier_stokes.py` 실행·결과 |
| Challenge 1 | [Advanced_Wave_Dynamics.ipynb](../challenge/wave/Advanced_Wave_Dynamics.ipynb) | Level 1–3의 편집·저장·실행·평가 |
| Challenge 2 | [Fluid_Structure_Interaction.ipynb](../challenge/fuild/Fluid_Structure_Interaction.ipynb) | Level 1–3의 편집·저장·실행·결과 |
| Challenge 3 | [Multi-Physics_Climate_Modeling.ipynb](../challenge/climate/Multi-Physics_Climate_Modeling.ipynb) | Level 1–2의 편집·저장·실행·평가 |
| Challenge 4 | [Advanced_Neural_Operators.ipynb](../challenge/neural_operator/Advanced_Neural_Operators.ipynb) | 데이터 생성과 Level 1–3의 편집·저장·실행·평가 |

`challenge/fuild/`는 원본 폴더명입니다. 폴더명을 일괄 변경하면 기존 링크와 경로에 영향을 주므로 그대로 사용합니다.

## Challenge 11개 Level 확인표

노트북과 각 실행 파일은 같은 폴더에 있습니다. 모든 Level을 과정에 포함합니다. 리허설에서는 참가자 입장에서 빈칸을 완성한 별도 작업 사본을 사용하고, 배포할 원본 과제의 빈칸을 정답으로 덮어쓰지 않습니다.

| 과정 | Level | 편집·저장할 원본 실행 파일 | 확인할 내용 |
|---|---:|---|---|
| Wave | 1 | [wave_l1.py](../challenge/wave/wave_l1.py) | 기본 2D 파동, 초기·경계조건, 검증 값 |
| Wave | 2 | [wave_l2.py](../challenge/wave/wave_l2.py) | 가변 파동 속도, 조건·PDE 평가 |
| Wave | 3 | [wave_l3.py](../challenge/wave/wave_l3.py) | 원형 영역·경계조건, 조건·PDE 평가 |
| Fluid | 1 | [chip_2d_l1.py](../challenge/fuild/chip_2d_l1.py) | 기본 2D 유동과 경계조건 |
| Fluid | 2 | [chip_2d_l2.py](../challenge/fuild/chip_2d_l2.py) | 복수 블록과 영역 구성 |
| Fluid | 3 | [chip_2d_l3.py](../challenge/fuild/chip_2d_l3.py) | 시간 입력과 초기조건 |
| Climate | 1 | [climate_l1.py](../challenge/climate/climate_l1.py) | 단순 대기 모델 |
| Climate | 2 | [climate_l2.py](../challenge/climate/climate_l2.py) | 대기·해양 결합 모델 |
| Neural Operators | 1 | [fno_physicsnemo_l1.py](../challenge/neural_operator/fno_physicsnemo_l1.py) | FNO 입력·출력·데이터 |
| Neural Operators | 2 | [fno_physicsnemo_l2.py](../challenge/neural_operator/fno_physicsnemo_l2.py) | AFNO 입력·출력·데이터 |
| Neural Operators | 3 | [fno_physicsnemo_l3.py](../challenge/neural_operator/fno_physicsnemo_l3.py) | PINO 데이터 손실·물리 손실 |

Neural Operators 노트북은 먼저 [generate_data.py](../challenge/neural_operator/generate_data.py)를 실행합니다. 생성 데이터의 위치와 각 Level이 읽는 경로를 리허설에서 확인합니다. 강사가 확인한 실행 명령·로그·경과 시간·GPU 메모리·생성 파일·평가 값을 각 Level별로 기록합니다.

## 원본 관리 관점에서 확인할 사항

아래 항목은 원본의 제목·수식·데이터를 맞춰 볼 구체적인 확인 사항입니다. 과정 삭제나 강의 주제 변경의 근거로 사용하지 않으며, 이번 안내 정비에서 수식·학습 코드를 변경하지 않습니다.

| 항목 | 원본에서 확인한 근거 | 후속 확인 |
|---|---|---|
| README와 시작 노트북의 제목 | README Challenge 2–4는 Darcy/AFNO·FourCastNet·MHD/PINO, `Start_Here.ipynb`는 Fluid·Climate·Advanced Neural Operators | 원본 관리 의도와 올바른 제목–파일 관계 확인. 공유 강의안은 README 내용을 그대로 반영했음 |
| Wave Level 1 기준 해 | [wave_l1.py](../challenge/wave/wave_l1.py)는 `c=1`과 `sin(x)sin(y)(sin(t)+cos(t))`를 기재. 이를 미분하면 `u_tt=-u`, `u_xx+u_yy=-2u`여서 기재한 PDE에 잔차 `u`가 남음 | 과제에서 의도한 파동 속도·초기조건·검증 해 확인. Level 2·3에 동일 오류가 있다고 확대하지 않음 |
| Neural Operators의 문제식·데이터 | [generate_data.py](../challenge/neural_operator/generate_data.py)는 `f=-Δu`를 설명하고, [PINO 코드](../challenge/neural_operator/fno_physicsnemo_l3.py)는 평가 잔차에 `u-Δu-f`를 사용 | 각 Level의 의도된 문제식과 실제 HDF5 데이터를 함께 확인. 번들 데이터 자체의 물리 잔차는 아직 미검증 |
| 데이터 생성 구현 | `generate_data.py`의 Fourier 합성에 `knm,lij`·`knm,lim` 인덱스가 사용됨. 생성 경로와 Level별 로드 경로도 대조 필요 | 공간 좌표별 합성과 데이터 경로가 의도대로인지 확인. 생성 코드의 점검 결과를 기존 데이터 전체의 오류로 단정하지 않음 |

## 환경과 검증 상태

현재 저장소의 컨테이너 기준은 `nvcr.io/nvidia/physicsnemo/physicsnemo:25.11`입니다. [배포 안내](../Deployment_Guide.MD)와 [환경 확인 노트북](00_환경확인.ipynb)을 사용해 행사 환경을 준비합니다. 이번 문서 정비에서는 새 API로 이식하거나 GPU·Brev 환경을 생성하지 않습니다.

행사 운영용 GPU 머신의 저장소 루트에서 다음 명령으로 컨테이너를 준비합니다. 기존 Dockerfile은 교재를 `/workspace/ai4sci`에 복사하고 `ai4sci/README.md`를 시작 화면으로 지정합니다.

```bash
docker build -t ai4sci-physicsnemo:rehearsal .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 127.0.0.1:8888:8888 -it ai4sci-physicsnemo:rehearsal
```

로그에 표시된 인증 링크로 접속합니다. 원격 머신에서는 행사 환경의 인증 프록시 또는 승인된 SSH 터널을 사용합니다. 배포 전 교재의 영구 저장 위치와 참가자 접속 경로를 확인하고, 실제 검증한 이미지 digest·패키지 목록을 기록합니다. 위 명령은 이 문서 작성 중 실행하지 않았습니다.

| 항목 | 상태 |
|---|---|
| 원본 학습 Python·YAML·노트북 코드 셀 | 이번 정비에서 유지 |
| 전체 노트북·Level의 참가자 이동 안내 | 위 확인표와 참가자 안내에 명시 |
| 컨테이너 실제 기동·이미지 digest·패키지 목록 | 확인 필요 |
| 원본 Lab과 Challenge 11개 Level의 행사 GPU 리허설 | 확인 필요 |
| 학습 시간·GPU 메모리·생성 결과·평가 값 | 확인 필요 |
| 데이터 다운로드·재접속·파일 저장·동시 접속 | 확인 필요 |
| Mingyu Yang·유영건의 실제 담당 배정 | 확인 필요 |

이전에 추가한 [01_Wave_PINN.ipynb](01_Wave_PINN.ipynb)와 `ai4sci/wave/`는 개발 참고 자료로 보존합니다. 원본 과정의 대체 실습이 아니며, 해당 보조 자료의 검사 결과로 원본 Challenge가 검증되었다고 표현하지 않습니다. 파일·문법·링크 검사와 실제 GPU 학습 성공도 구분해 기록합니다.
