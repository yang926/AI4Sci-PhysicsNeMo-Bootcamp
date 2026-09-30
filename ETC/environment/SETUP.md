# uv로 설치하고 JupyterLab에서 강의 열기 · 한국어판

이 과정은 **Python 3.12**, `sym` 추가 기능이 통합된 **PhysicsNeMo 2.2.2**, 별도의 가상 환경을 사용합니다. 학생용 Brev의 대상 하드웨어는 **NVIDIA L4 한 대**입니다. 실제 GPU는 `nvidia-smi`로 확인하세요. 로컬 RTX 3080 측정값을 L4 결과로 제시하면 안 됩니다. 각 결과의 범위는 [검증 기록](../course_materials/VALIDATION.md)과 [기상 실습 측정 기록](../course_materials/LAB4_WEATHER_VALIDATION.md)을 참고하세요.

학생용 Brev Launchable에서는 [관리형 Jupyter 설정](../launchable/README.md)을 사용하세요. 아래 수동 서버 안내는 독립된 장비용이며, Brev가 관리하는 Jupyter 옆에 추가 서버를 실행하기 위한 안내가 아닙니다.

현재 상황에 맞는 절을 선택하세요.

- JupyterLab이 이미 열리는 경우: [시작 안내](../../Start_Here.ipynb)로 이동하세요.
- 정상 환경이 있지만 Jupyter가 멈춘 경우: [JupyterLab 시작](#start-jupyterlab-on-the-remote-machine)을 참고하세요.
- Mac 브라우저의 연결이 끊긴 경우: [터널 재연결](#connect-from-the-mac-and-reconnect-after-moving)을 참고하세요.
- 새 원격 장비에 Python 패키지가 필요한 경우: 아래를 계속 읽으세요.

<a id="keep-an-existing-workspace"></a>

## 기존 작업 공간 유지

저장소가 이미 열려 있다면 그 작업 사본을 사용하세요. `pwd`와 `git status --short`를 실행하고, 노트북과 Python 수정 사항을 저장하며, 기존 데이터셋과 출력을 보존하세요. 이 안내를 따르기 위해 폴더를 교체하거나 기존 환경을 다시 만들지 마세요.

작업 사본이 없는 새 장비에서는 사용하지 않는 디렉터리로 복제하세요.

```bash
git clone --branch ko https://github.com/yang926/AI4Sci-PhysicsNeMo-Bootcamp.git
cd AI4Sci-PhysicsNeMo-Bootcamp
```

Brev에서는 **원격 Linux 터미널**, Windows 데스크톱에서는 **WSL**에서 설치 명령을 실행하세요. Mac 터미널은 나중에 연결 터널을 만드는 데 사용합니다. `uv --version`을 실행할 수 없다면 [uv를 설치](https://docs.astral.sh/uv/getting-started/installation/)하세요. 아래 명령 옵션은 uv 0.8.17을 기준으로 확인했으며, 해당 인터프리터가 없으면 `uv venv --python 3.12.11`이 내려받을 수 있습니다. 이 명령은 장비를 준비할 때 실행하는 절차이며, 새로운 Brev 환경에 설치가 이미 끝났다는 근거가 아닙니다.

이 작업 디렉터리에 정상 작동하는 `.venv`가 이미 있으면 활성화한 뒤 **환경 확인**으로 이동하세요. 다른 장비의 디렉터리에 연결된 환경은 Git 복제로 함께 옮겨지지 않습니다.

<a id="new-linux-nvidia-gpu-environment"></a>

## 새 Linux NVIDIA GPU 환경

저장소 루트에서 먼저 GPU와 드라이버를 확인하세요.

```bash
nvidia-smi
uv --version
```

아래 경로에 새 환경을 만듭니다. 보호 조건은 기존 디렉터리나 심볼릭 링크를 보존합니다. 이름이 이미 사용 중이면 다른 경로를 선택하거나 기존 환경을 확인한 뒤 재사용하세요.

```bash
course_env="$HOME/.venvs/ai4sci-brev-cuda"
(
  set -eu
  if test -e "$course_env" || test -L "$course_env"; then
    printf 'Environment already exists: %s. Reuse it or choose a new path.\n' "$course_env"
    exit 1
  fi
  uv venv --python 3.12.11 "$course_env"
  uv pip install --python "$course_env/bin/python" \
    'torch==2.10.0+cu128' 'torchvision==0.25.0+cu128' \
    --default-index https://download.pytorch.org/whl/cu128
  uv pip install --python "$course_env/bin/python" -r requirements.txt \
    'torch==2.10.0+cu128' 'torchvision==0.25.0+cu128'
  uv pip check --python "$course_env/bin/python"
  "$course_env/bin/python" -m ipykernel install --prefix "$course_env" \
    --name ai4sci-physicsnemo-uv --display-name 'AI4Sci PhysicsNeMo 2.2.2 (uv / CUDA)'
)
```

설치가 성공하면 환경을 활성화하세요.

```bash
source "$HOME/.venvs/ai4sci-brev-cuda/bin/activate"
```

두 번째 설치는 PyPI에서 강의 의존성을 해결하면서 CUDA용 PyTorch 버전을 명시적으로 유지합니다. `nvidia-physicsnemo[sym]==2.2.2`는 이미 `requirements.txt`에 포함되어 있으므로, 예전의 별도 패키지인 `nvidia-physicsnemo.sym`을 추가하지 마세요. 이 절차는 핵심 프레임워크 버전을 고정하며 **모든 전이 의존성까지 고정하지는 않습니다**. 데스크톱의 `ETC/environment/requirements-wsl-cuda.lock.txt`는 로컬 스냅샷이며 새 GitHub 복제본에서 제공되는 파일이 아닙니다. 새 설치와 드라이버 호환성은 대상 GPU에서 검증해야 합니다. 인덱스 처리 방식은 공식 [uv PyTorch 안내](https://docs.astral.sh/uv/guides/integration/pytorch/)를 참고하세요.

<a id="new-linux-cpu-environment"></a>

## 새 Linux CPU 환경

CPU 작업에는 별도 환경을 사용하세요. 이 절차는 보존된 Linux CPU 패키지 스냅샷의 프레임워크 버전을 재현하며, 위 CUDA 설치 절차와는 다릅니다.

```bash
course_env="$HOME/.venvs/ai4sci-linux-cpu"
(
  set -eu
  if test -e "$course_env" || test -L "$course_env"; then
    printf 'Environment already exists: %s. Reuse it or choose a new path.\n' "$course_env"
    exit 1
  fi
  uv venv --python 3.12.11 "$course_env"
  uv pip install --python "$course_env/bin/python" \
    'torch==2.14.0+cpu' 'torchvision==0.29.0+cpu' \
    --default-index https://download.pytorch.org/whl/cpu
  uv pip install --python "$course_env/bin/python" -r ETC/environment/requirements-linux-cpu.lock.txt
  uv pip install --python "$course_env/bin/python" 'ipywidgets>=8.1,<9'
  uv pip check --python "$course_env/bin/python"
  "$course_env/bin/python" -m ipykernel install --prefix "$course_env" \
    --name ai4sci-physicsnemo-uv --display-name 'AI4Sci PhysicsNeMo 2.2.2 (uv CPU)'
)
```

설치가 성공하면 `source "$HOME/.venvs/ai4sci-linux-cpu/bin/activate"`로 활성화하고 아래에서 `AI4SCI_DEVICE=cpu`와 `--device cpu`를 사용하세요. Linux 스냅샷은 macOS 환경이 아닙니다. 과거의 [macOS 패키지 스냅샷](requirements-macos.lock.txt)은 PyPI의 CPU용 PyTorch를 사용했습니다. Apple MPS와 AMD GPU 실행은 검증되지 않았습니다. 브라우저와 SSH 클라이언트로만 사용하는 Mac에는 강의 Python 패키지가 필요하지 않습니다.

<a id="check-the-environment"></a>

## 환경 확인

현재 의존성 목록에는 노트북 진행 막대용 `ipywidgets`가 포함되어 있습니다. 과거 패키지 스냅샷 중 하나의 환경을 재사용한다면 `uv pip install --python "$VIRTUAL_ENV/bin/python" 'ipywidgets>=8.1,<9'`로 설치하세요. 브라우저로만 사용하는 Mac이 아니라 강의 환경에 설치해야 합니다. 실행 중인 세션에 추가했다면 먼저 작업을 저장한 뒤 노트북을 다시 열고 커널을 재시작한 후 확인하세요. [Jupyter 위젯 설치 안내](https://ipywidgets.readthedocs.io/en/stable/user_install.html)를 참고하세요.

활성화한 환경에서 저장소 루트를 기준으로 실행하세요.

```bash
python --version
uv pip check --python "$VIRTUAL_ENV/bin/python"
python -c 'import sys, torch; from importlib.metadata import version; print("Python:", sys.executable); print("PhysicsNeMo:", version("nvidia-physicsnemo")); print("PyTorch:", torch.__version__); print("PyTorch CUDA runtime:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
```

CUDA 환경에서는 `2.2.2`, `2.10.0+cu128`, CUDA 사용 가능 여부, 실제 GPU 모델을 확인하세요. `nvidia-smi`가 보고하는 드라이버와 PyTorch가 보고하는 CUDA 런타임은 설치의 서로 다른 부분입니다. 수업을 시작하기 전에 선택한 강의 커널에서 [환경 확인 노트북](../../00_Setup.ipynb)을 실행하세요. `AI4SCI_DEVICE=cuda`는 CUDA를 사용할 수 없을 때 오류를 발생시킵니다.

<a id="start-jupyterlab-on-the-remote-machine"></a>

## 원격 장비에서 JupyterLab 시작

원격 터미널에서 Jupyter를 계속 실행해 두세요. `tmux`를 사용할 수 있으면 먼저 `tmux new -s ai4sci`를 실행한 뒤, 그 세션 안에서 저장소 루트로 이동하고 사용할 환경을 활성화하세요. **Ctrl+B를 누른 다음 D**로 세션에서 빠져나오고 `tmux attach -t ai4sci`로 다시 연결합니다. 이 방식은 원격 프로세스를 SSH 터미널과 독립적으로 유지하지만, 장비 재시작이나 삭제 후까지 유지하지는 못합니다.

일반 CUDA 수업에서는 각 노트북의 전체 학습 횟수를 사용하세요.

```bash
env -u AI4SCI_STEPS AI4SCI_DEVICE=cuda AI4SCI_REFERENCE=0 \
  jupyter lab --no-browser --ip=127.0.0.1 --port=8888 \
  --ServerApp.port_retries=0 --ServerApp.root_dir="$PWD" \
  --LabApp.default_url=/lab/tree/Start_Here.ipynb
```

`env -u AI4SCI_STEPS`는 새 서버에 상속되는 짧은 실행용 재설정을 제거합니다. 생성된 토큰 인증을 유지하고 출력된 localhost URL을 보관하세요. 8888 포트를 이미 사용 중이면 원하는 기존 강의 서버를 재사용하거나 다른 포트를 선택하고 아래 포워딩도 맞추세요. 관련 없는 서버를 중지하지 마세요. 네 도전 과제는 모두 학생이 저장한 구현을 실행합니다. 학습 전에 표시된 각 연습을 완성하세요. 미완성 함수는 메시지를 표시하고 중단됩니다. 정답 모드나 참조 구현 전환 옵션은 없습니다.

강사용 짧은 CUDA 리허설:

필요하다면 새 서버를 시작하기 전에 위 명령의 `env -u AI4SCI_STEPS`를 `env AI4SCI_STEPS=20`으로 바꾸세요. 옵티마이저를 20번 호출하여 실행 여부를 확인하며 수업의 정확성을 검증하지는 않습니다. 수업에서는 위의 일반 명령을 사용하세요. 실습 4는 사전 학습 모델의 추론이므로 `AI4SCI_STEPS`를 무시합니다. 6시간 간격의 여덟 예측 단계는 옵티마이저 갱신이 아닙니다. 장치와 학습 횟수 환경 변수는 새 서버를 시작할 때 기본값을 선택하며, 이미 실행 중인 커널은 기존 환경을 유지합니다.

`Start_Here.ipynb`를 열고 알맞은 **AI4Sci** 커널에서 환경 확인을 실행한 뒤 소개와 실습을 순서대로 진행하세요. 소개는 읽기 자료이며 첫 학습 연습은 실습 1입니다. 위의 `ai4sci-physicsnemo-uv` 커널 명세는 강의 노트북과 일치합니다.

<a id="read-markdown-as-a-document"></a>

## Markdown을 문서로 읽기

JupyterLab에서 **Settings > Settings Editor > Document Manager(설정 > 설정 편집기 > 문서 관리자)**를 열고 `markdown`의 기본 뷰어를 `Markdown Preview`로 설정하세요. JSON 설정 편집기에서 이에 해당하는 설정은 다음과 같습니다.

```json
{
  "defaultViewers": {
    "markdown": "Markdown Preview"
  }
}
```

저장 후 브라우저 페이지를 새로고침하세요. 기존 편집기 탭은 원래 보기를 유지하므로 닫았다 다시 열거나 파일을 오른쪽 클릭하여 **Open With > Markdown Preview(다음으로 열기 > Markdown 미리 보기)**를 선택하세요. 저장하지 않은 수정 사항을 먼저 저장하세요. Markdown 원문을 편집하려면 **Open With > Editor(다음으로 열기 > 편집기)**를 사용하세요.

강의는 `Start_Here.ipynb`에서 시작합니다. 루트의 `README.md`는 GitHub용 간단한 소개이며 별도의 수업 목차가 아닙니다.

<a id="connect-from-the-mac-and-reconnect-after-moving"></a>

## Mac에서 연결하고 이동 후 다시 연결하기

**Mac 터미널**에서 `YOUR_INSTANCE`를 기존 Brev 인스턴스 이름으로 바꾸어 사용하세요.

```bash
brev login
brev refresh
brev port-forward YOUR_INSTANCE --port 8888:8888
```

포워딩 프로세스를 계속 실행한 상태로 원격 서버의 토큰이 포함된 URL을 로컬 `127.0.0.1:8888`에서 여세요. 로컬 포트를 사용 중이면 `--port 8889:8888`을 사용하고 브라우저 URL의 포트만 8889로 바꾸세요. 매핑 순서는 **로컬:원격**입니다. [Brev 연결 문서](https://docs.nvidia.com/brev/cli/connectivity)

네트워크 이동, Mac 잠자기, 포워딩 프로세스 종료로 브라우저 연결이 끊길 수 있습니다. Mac을 다시 연결하고 포트 포워딩을 재실행한 뒤 URL을 새로고침하세요. 인스턴스가 재시작되었다면 먼저 `brev refresh`를 실행하고 셸에 다시 연결하여 Jupyter를 재시작해야 하는지 확인하세요. 터널이 끊겼다는 사실만으로 원격 학습이 멈췄다고 판단할 수는 없습니다. 중복 작업을 실행하기 전에 원격 프로세스와 현재 실행의 결과물을 확인하세요. 토큰이 포함된 URL과 SSH 키를 GitHub나 공유 스크린샷에 노출하지 마세요.

<a id="data-edits-and-saved-runs"></a>

## 데이터, 수정 사항, 저장된 실행 결과

채점 연결 상태는 각 도전 과제의 마지막 제출 패널에서 확인하세요. 연결된 작업 공간에서는 닉네임을 등록하고 노트북 안에서 코드를 제출한 뒤 결과를 확인합니다. 연결되지 않은 경우에도 로컬 실습과 저장된 코드 확인은 사용할 수 있습니다. 연결 설정은 강사가 [노트북 채점기 연결](JUDGE_CONNECTION.md)에 따라 관리합니다. 별도 웹페이지는 순위표로만 사용합니다. 학생 노트북 컴퓨터에 다른 애플리케이션을 설치할 필요는 없습니다.

학생마다 쓰기 가능한 개인 작업 사본을 사용하세요. Brev 인스턴스나 컨테이너의 수명 주기 동안 어떤 디스크나 마운트된 작업 공간이 유지되는지 확인하세요. 브라우저 연결과 실행 환경만으로 저장소의 지속성이 보장되지는 않습니다. 작업 공간을 교체하거나 인스턴스를 삭제하기 전에 저장한 노트북, 수정한 `.py` 파일, 필요한 출력을 백업하세요.

신경 연산자 도전 과제는 학습/검증/테스트 표본 8,000/1,000/1,000개로 이루어진 64×64 반응–확산 데이터셋을 직접 생성합니다. 준비된 작업 공간마다 한 번 생성하여 검증한 뒤 세 단계에서 재사용하세요. 다른 준비된 데이터 위치를 사용하려면 노트북에 `AI4SCI_DATA_DIR`을 지정할 수 있습니다. 과거의 `Poisson_Fourier` 파일은 이 구현의 입력이 아닙니다.

현재 실습 4는 버전이 고정된 사전 학습 FourCastNet 모델과 실제 ERA5 자료를 사용합니다. Launchable 설치 프로그램은 이를 `~/.cache/ai4sci/weather`에 자동으로 준비합니다. 독립된 장비에서는 실습 4의 첫 준비 셀이 같은 공개 파일을 내려받고 확인합니다. CDS 계정은 필요하지 않습니다. 처음 준비할 때 모델 파일 약 301 MB와 압축된 데이터 청크 약 622 MB를 전송합니다. 캐시는 저장소 밖에 있으며 이후 실행에서는 확인된 파일을 재사용합니다. [기상 검증 기록](../course_materials/LAB4_WEATHER_VALIDATION.md)을 참고하세요.

기존 Navier–Stokes PINN 수업과 변경하지 않은 `data_lat.npy`는 `ETC/reference_labs/04_navier_stokes`에 보존되어 있습니다. 해당 학습 및 ParaView 절차는 선택적 참고 자료이며 현재 기상 예측이나 그 검증에 해당하지 않습니다.

학습 실행이 성공하면 `metrics.json`, `loss.csv`, `model.pt`, `predictions.npz`, `preview.png`를 저장합니다. 기존 출력 디렉터리는 보호됩니다. 노트북 학습 셀은 실행할 때마다 새 결과 디렉터리를 선택하므로 같은 단계를 반복하기 위해 환경 설정을 다시 실행할 필요는 없습니다. 명령줄 실행에서는 새 `--output-dir`와 명시적인 `--steps` 값을 직접 지정하세요. 이 스크립트는 최종 결과물을 저장하며, 종료된 학습 프로세스에 재개 가능한 중간 체크포인트가 남는다고 보장하지 않습니다. 생성 데이터와 출력의 기본 위치는 Git에서 제외하므로 강의 코드를 푸시해도 이 파일들은 백업되지 않습니다. 사용자 지정 경로를 선택했다면 공개하기 전에 `git status`로 생성 데이터나 비공개 결과가 실수로 포함되지 않았는지 확인하세요.

<a id="short-checks-and-lesson-rehearsal"></a>

## 짧은 점검과 수업 리허설

환경을 활성화한 뒤 검증 세션마다 새 상위 디렉터리를 만드세요.

```bash
mkdir -p ETC/validation-runs
validation_root="$(mktemp -d "$PWD/ETC/validation-runs/cuda-check-XXXXXX")"
python ETC/course_materials/run_validation.py --suite unit --device cpu --output-dir "$validation_root/unit"
python ETC/course_materials/run_validation.py --suite smoke --case pinn_forward --device cuda --steps 2 --output-dir "$validation_root/first-gpu-check"
```

첫 GPU 점검이 성공하면 더 넓은 실행 경로를 확인하세요.

```bash
python ETC/course_materials/run_validation.py --suite smoke --device cuda --steps 20 --output-dir "$validation_root/smoke"
python ETC/course_materials/run_notebooks.py --device cuda --steps 2 --output-dir "$validation_root/notebooks"
```

도전 과제 리허설에는 선택한 학생 함수가 완성된 별도의 작업 사본이 필요합니다. 배포된 빈칸 상태에서는 두 실행기 모두 중단되는 것이 정상이며, 어느 실행기도 정답을 제공하거나 모드를 바꾸지 않습니다. `--case`로 완성된 수업을 선택하거나 노트북 실행기에 `--course-root /path/to/completed-checkout`을 전달하세요. 노트북 실행기는 포함된 그래프를 확인하고 실행한 사본을 원본 노트북 밖에 저장합니다. 신경 연산자 노트북은 수업의 전체 크기 데이터셋을 생성하며, 스모크 점검 모음은 더 작은 벤치마크를 사용합니다. 이 점검은 실행과 결과물을 확인하며, 도전 과제 답안의 독립적인 정답 검증, 전체 학습, 수업 소요 시간을 확인하지는 않습니다.

공개할 깨끗한 작업 사본에서는 엄격한 자료 검사를 실행하세요.

```bash
python ETC/course_materials/validate_materials.py --output "$validation_root/materials-clean.json"
```

노트북 출력을 유지할 작업 사본에서는 명시적으로 완화된 검사를 사용하세요.

```bash
python ETC/course_materials/validate_materials.py --allow-executed-notebooks --output "$validation_root/materials-working-copy.json"
```

후자는 출력을 보존하고 출력 정리 검증을 건너뛰었다고 보고합니다. 깨끗한 공개본 검사는 아닙니다. 엄격한 검사를 통과하려고 학생의 결과를 지우지 마세요.

별도의 `--suite convergence --convergence-steps 500` 옵션은 선택한 사례에서 학습에 사용하지 않은 평가 데이터에 대한 성능 향상을 측정합니다. 더 긴 실험이며 완전한 수렴을 인증하지는 않습니다. 실제 수업 리허설에서는 대상 하드웨어에서 설명, 학생 수정 사항 저장, 기본 횟수 실행, 오류 복구, 결과 해석에 걸리는 시간도 측정해야 합니다. L40 실행 시간과 다중 사용자 자원 계획은 아직 측정되지 않았습니다. 로컬의 짧은 점검 시간을 확대 추정하여 그런 주장에 사용하지 마세요.

<a id="optional-docker-path"></a>

## 선택 사항: Docker 사용

[Dockerfile](Dockerfile)은 공식 PhysicsNeMo 26.08 이미지를 사용합니다. 기록된 uv/WSL 결과는 이 이미지의 빌드, 호스트 드라이버 호환성, 실행을 검증하지 않습니다. 이 별도 환경을 준비한다면 인증을 유지하고, Jupyter를 호스트의 localhost에만 공개하며, 전용 영구 작업 사본을 마운트하세요.

```bash
docker build -f ETC/environment/Dockerfile -t ai4sci-physicsnemo:2.2.2 .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 127.0.0.1:8888:8888 -v "$PWD:/workspace/ai4sci" \
  -it --rm ai4sci-physicsnemo:2.2.2
```

컨테이너의 Jupyter는 컨테이너 안에서 수신하고, 위 호스트 매핑은 포워딩을 위한 접근을 localhost로 제한합니다. 마운트된 작업 사본은 컨테이너 종료 후에도 파일을 유지합니다. 마운트하지 않은 다른 컨테이너 파일은 `--rm`으로 사라집니다. 여러 사람이 동시에 편집할 학생용 공유 작업 사본을 마운트하지 마세요.

[전체 과정](../../Start_Here.ipynb) · [강사 안내](../course_materials/INSTRUCTOR.md) · [uv 환경 선택](https://docs.astral.sh/uv/pip/environments/)
