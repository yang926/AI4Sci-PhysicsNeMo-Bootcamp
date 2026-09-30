# AI4Sci PhysicsNeMo Bootcamp · 한국어판

영문 `main`과 분리된 `ko` 브랜치입니다. 설명과 도식을 번역했으며 실습 코드·방정식·학습 설정은 유지했습니다. 채점 연결 상태는 각 도전 과제의 마지막 제출 패널에서 확인하세요. 연결된 작업 공간에서는 닉네임 등록과 코드 제출이 가능하며, 연결 없이도 로컬 실습과 저장된 코드 확인은 사용할 수 있습니다. [번역 범위와 검증](ETC/localization/README.md)을 참고하세요.

이 과정은 PhysicsNeMo 2.2.2와 Python 3.12를 사용하여 물리 정보 신경망, 신경 연산자, AI 기상 예측을 다룹니다.

권장 사전 지식은 기본적인 Python 함수와 배열, 1차·2차 미분, 초기 조건과 경계 조건입니다. PhysicsNeMo 사용 경험은 필요하지 않습니다.

[시작 안내](Start_Here.ipynb)에서 수업 순서를 확인하세요. GitHub에서 자료를 읽거나 JupyterLab에서 실행할 수 있습니다. 실습 1–3은 완성된 PINN 학습 예제를 제공합니다. 실습 4: FourCastNet을 활용한 AI 기상 예측에서는 사전 학습된 AFNO 모델로 48시간 예측을 생성하고 ERA5 재분석 및 지속성 기준 예측과 비교합니다. 모델을 학습하지는 않습니다. 도전 과제 네 개에서는 연결된 Python 파일의 **EDIT HERE(여기를 수정)** 블록을 채우고 저장한 뒤 노트북을 실행하세요. 도전 과제는 직접 작성한 구현만 실행하며 완성된 정답 실행 모드는 포함하지 않습니다. 기본 갱신 횟수는 단계마다 도전 과제 1–3이 5,000회, 도전 과제 4가 3,000회입니다. **Check saved code(저장된 코드 확인)**는 구문과 작성 완료 여부만 확인하며, 정답 여부는 채점기가 평가합니다.

각 구성 요소가 어떻게 연결되는지 익혀보세요. PhysicsNeMo는 모델과 PDE 잔차 계산을 제공하고, SymPy는 방정식을 표현하며, PyTorch는 모델 매개변수를 갱신합니다. 소개 노트북에서 이 흐름을 설명합니다. [코드 해설](ETC/course_materials/PHYSICSNEMO_WORKFLOW.md)에서 `create_model`, `create_informer`, `residuals`와 같은 강의 보조 함수의 내부를 살펴보세요. 이 이름들은 PhysicsNeMo의 공개 API가 아닙니다.

실행 환경이 필요하면 [uv 설치 안내](ETC/environment/SETUP.md)를 따르세요. JupyterLab이 이미 실행 중이면 다시 설치할 필요 없이 시작 안내를 이용하세요.

학생용 Brev Launchable은 [GitHub 기반 설정 및 갱신 안내](ETC/launchable/README.md)를 따르세요. 독립된 강의용 커널을 Brev에서 관리하는 Jupyter에 연결하고, 갱신할 때 기존 학생 작업을 보존합니다.

## 배포 이력

2026-10-01에 개발 커밋 이력을 정리하고 영문·한국어 배포본으로 새 이력을 시작했습니다.
그 전에 저장소를 받았다면 기존 폴더와 작업은 보관하고, 새 폴더에 다시 clone하세요.
기존 업데이트 기능은 이전 이력을 덮어쓰지 않고 중단합니다. 새로 받은 저장소는
이후 일반적인 업데이트 기능을 사용할 수 있습니다.

## 출처

[OpenHackathons AI-Powered-Physics-Bootcamp](https://github.com/openhackathons-org/AI-Powered-Physics-Bootcamp)를 바탕으로 구성했습니다.

호환성 및 문제 설정 변경 사항은 [마이그레이션 기록](ETC/course_materials/MIGRATION.md)에 정리되어 있습니다. 강사의 승인을 받은 기상 실습이 기존 Navier–Stokes 수업을 대체하며, 기존 [PINN 노트북과 원본 데이터](ETC/reference_labs/04_navier_stokes)는 참고용으로 보존되어 있습니다. 이 대체로 실습 1–3과 도전 과제의 문제는 바뀌지 않았습니다.

기상 실습은 [NVIDIA FourCastNet1](https://huggingface.co/nvidia/fourcastnet1)과 Google ARCO-ERA5에서 제공하는 Copernicus/ECMWF ERA5 재분석 자료를 사용합니다. 내려받은 모델과 데이터는 저장소 작업 디렉터리 밖에 캐시됩니다. 출처, 라이선스, 입력 전처리의 자세한 내용은 기상 실습을 참고하세요.

추가 학습과 커뮤니티 지원은 [Open Hackathons 자료](https://www.openhackathons.org/s/technical-resources)와 [OpenACC 및 Hackathons Slack 채널](https://www.openacc.org/community#slack)을 이용하세요.

## 라이선스

Copyright © 2026 OpenACC-Standard.org. This material is released by OpenACC-Standard.org, in collaboration with NVIDIA Corporation, under the Creative Commons Attribution 4.0 International (CC BY 4.0). These materials may include references to hardware and software developed by other entities; all applicable licensing and copyrights apply.
