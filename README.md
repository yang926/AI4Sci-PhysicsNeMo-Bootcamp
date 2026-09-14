# AI4Sci PhysicsNeMo Bootcamp

점심·휴식을 포함한 6~7시간 동안 PINN의 원리를 배우고, 투사체와 파동 문제에서 물리 조건과 코드의 관계를 확인하는 한국어 워크숍입니다.

**[참가자 수업 안내](ai4sci/README.md)** 에서 순서와 완료 기준을 확인하고 **[환경 확인 노트북](ai4sci/00_환경확인.ipynb)** 을 엽니다.

## 현재 버전

행사 준비용 개편 초안입니다. 기존 PhysicsNeMo 25.11 환경을 기준으로 구성했습니다. 파동 해석해의 CPU 수치 검증과 자료 구조 검사를 수행했으며, 컨테이너 빌드·행사 GPU 학습·Brev 접속 리허설은 아직 필요합니다.

| 자료 | 용도 |
|---|---|
| [한국어 수업 안내](ai4sci/README.md) | 실습 순서, 결과, 완료 기준 |
| [파동 PINN 실습](ai4sci/01_Wave_PINN.ipynb) | 완성된 기본 코드 실행과 조건 변경 |
| [6시간·7시간 강의안](ai4sci/course-plan.md) | 점심·휴식을 포함한 편성 초안 |
| [강사 준비 안내](ai4sci/INSTRUCTOR.md) | 환경, 검증 상태, 리허설 절차 |

`tutorial/`과 `challenge/`는 출처 추적을 위해 보존한 원본 자료입니다. 일부 심화 예제에는 의도적인 `FIXME` 과제와 별도의 수식·데이터 불일치가 있습니다. 필수 실습은 참가자 수업 안내의 경로를 따릅니다. Darcy·FourCastNet·MHD를 모두 구현한 과정으로 안내하지 않습니다.

## 원본과의 관계

이 저장소는 [OpenHackathons AI-Powered-Physics-Bootcamp](https://github.com/openhackathons-org/AI-Powered-Physics-Bootcamp)의 원본 커밋 `9cae27f8303268cdaf7528fe963ce12ba439377f`와 이력을 보존한 AI4Sci 전용 독립 저장소입니다. 기존 KSC2026 포크는 별도로 유지합니다.

[원본 강의 개요](https://github.com/openhackathons-org/AI-Powered-Physics-Bootcamp/blob/9cae27f8303268cdaf7528fe963ce12ba439377f/README.md)와 행사 개편안을 구분합니다. 최신 PhysicsNeMo로의 API 이식은 별도 검증이 필요합니다.

## Attribution

This material originates from the OpenHackathons Github repository. Check out additional materials [here](https://github.com/openhackathons-org)

Don't forget to check out additional [Open Hackathons Resources](https://www.openhackathons.org/s/technical-resources) and join our [OpenACC and Hackathons Slack Channel](https://www.openacc.org/community#slack) to share your experience and get more help from the community.

## Licensing

Copyright © 2026 OpenACC-Standard.org. This material is released by OpenACC-Standard.org, in collaboration with NVIDIA Corporation, under the Creative Commons Attribution 4.0 International (CC BY 4.0). These materials may include references to hardware and software developed by other entities; all applicable licensing and copyrights apply.
