# 준비된 환경 사용

## 기존 Brev에 이번 교재 가져오기

배포 경로: [PyTorch Day Korea 교재](https://github.com/yang926/AI4Sci-PhysicsNeMo-Bootcamp/tree/workshop/pytorch-day-korea-2026/workshops/pytorch-day-korea-2026)

운영자는 기존 참가자 작업 폴더 밖의 **새 디렉터리**에 전용 branch를 받습니다. 같은 이름의 폴더가 이미 있으면 새 이름을 정하고, 기존 폴더에 덮어쓰지 않습니다.

```bash
git clone --branch workshop/pytorch-day-korea-2026 --single-branch https://github.com/yang926/AI4Sci-PhysicsNeMo-Bootcamp.git AI4Sci-PyTorch-Day-2026
cd AI4Sci-PyTorch-Day-2026/workshops/pytorch-day-korea-2026
```

기존 Jupyter에서 위 폴더의 `00_Start_Here.ipynb`를 열고 이미 설치된 **AI4Sci PhysicsNeMo 2.2.2 (uv / CUDA)** kernel을 선택합니다. 새 Jupyter 서비스를 시작할 필요가 없습니다. 교재 branch가 공개되기 전에는 위 경로로 clone할 수 없으므로, 배포 담당자가 실제 링크를 확인한 뒤 참가자에게 안내합니다.

## 환경 기준

이 교재는 9월 AI4Sci PhysicsNeMo 교재의 Python 3.12 / PhysicsNeMo 2.2.2 / PyTorch 2.10.0+cu128 환경을 사용합니다. 수업 중에는 00_Start_Here에서 기존 커널을 확인하며 설치나 업데이트를 실행하지 않습니다.

`requirements.txt`는 이번 두 실습이 직접 사용하는 패키지의 원본 고정 버전입니다. `original-linux-cu128.lock.txt`는 원본 전체 CUDA 환경의 버전 기록이며 이번 교재가 Jupyter 서비스나 모든 원본 기능을 다시 설치한다는 뜻이 아닙니다. 운영자가 별도 환경을 만들 때에는 원본 lock과 기존 Brev 환경을 기준으로 리허설해야 합니다.

외부 AI 서비스는 참가자 자신의 브라우저에서 사용합니다. API key, agent CLI, 채점 서버, 닉네임 등록, 개인 인증정보가 필요하지 않습니다.

수업 전 FNO 데이터를 아래 명령으로 한 번 준비할 수 있습니다. 교재 폴더에서 이미 준비된 course Python을 사용합니다. 기본값은 원본과 같은 grid 64, train/validation/test 8000/1000/1000, seed 42입니다. 기존 데이터 파일은 덮어쓰지 않습니다.

```bash
python -m src.generate_data --seed 42 --output-dir data/reaction_diffusion
```

GPU 인스턴스 생성, 비용, Launchable 설정과 참가자의 실제 접근은 이 저장소가 자동으로 처리하지 않습니다.
