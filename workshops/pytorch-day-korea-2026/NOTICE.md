# 원본과 변경 범위

원본: [yang926/AI4Sci-PhysicsNeMo-Bootcamp](https://github.com/yang926/AI4Sci-PhysicsNeMo-Bootcamp)  
확인한 커밋: [`42870e2d144ce4e176c06f1e37a94080a81c11b4`](https://github.com/yang926/AI4Sci-PhysicsNeMo-Bootcamp/tree/42870e2d144ce4e176c06f1e37a94080a81c11b4)  
원본 확인일: 2026-11-18

원래 교재는 [OpenHackathons AI-Powered-Physics-Bootcamp](https://github.com/openhackathons-org/AI-Powered-Physics-Bootcamp)를 바탕으로 PhysicsNeMo 2.2.2에 맞춰 개편한 자료입니다. 루트 LICENSE와 각 파일의 기존 NVIDIA 저작권 고지를 보존했습니다.

이번 교재는 2시간 한국어 워크숍을 위해 Lab 1 forward PINN과 Challenge 4 Level 1 FNO를 선택했습니다. 한국어 notebook 안내와 두 개의 짧은 AI 코드 요청, 명시적으로 선택하는 강사 예제를 새로 작성했습니다. 기존 행사 채점기·등록·Brev 서비스 자동 설정·다른 Labs/Challenges는 포함하지 않았습니다.

선택한 원본 training, data generation, loss, normalization, independent split 평가, 정확도 확인과 결과 저장 코드는 유지했습니다. 변경은 모듈 import와 repo-relative 기본 경로이며, notebook은 기존 함수와 CLI를 호출합니다. 지원 소스에는 원본의 공용 함수가 일부 남아 있지만 참가자 동선은 두 실습으로 한정됩니다.

정확한 원본 파일, SHA-256과 경로 변경은 [SOURCE_MAP.json](SOURCE_MAP.json)에 기록했습니다. 환경 확인과 새 실행 검증 범위는 강사 준비 기록과 함께 확인해야 합니다.
