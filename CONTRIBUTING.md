# 기여 방법

이 저장소에서 작업할 때 지키는 브랜치 · 커밋 · PR 규칙입니다.

## 1. main에는 직접 push하지 않는다

main은 보호되어 있습니다. 모든 변경은 **브랜치 → PR → 리뷰 승인 → Squash merge** 순서로 들어갑니다.

## 2. 브랜치 이름: `역할/작업내용`

| 접두어 | 용도 | 예시 |
|---|---|---|
| `road/` | 도로 · 신호 · 경로 | `road/yeonsan-osm` |
| `traffic/` | 일반 차량 · 교통량 | `traffic/hv-routes` |
| `mdp/` | 관측 · 행동 · 보상 | `mdp/signal-obs` |
| `train/` | 학습 · 평가 | `train/ppo-1m` |
| `data/` | 데이터 수집 | `data/collector-v1` |
| `page/` | 프로젝트 페이지 | `page/method-section` |
| `docs/` | 문서 | `docs/data-format` |
| `fix/` | 버그 수정 | `fix/teleport` |

## 3. 작업 흐름

```bash
git switch main
git pull
git switch -c road/yeonsan-osm
# ... 작업 ...
git add <바꾼 파일>
git commit -m "road: 연산교차로 OSM 도로 생성"
git push -u origin road/yeonsan-osm
# GitHub에서 Pull Request 생성 → 리뷰 승인 → Squash merge
```

## 4. 커밋 메시지: `역할: 내용`

예) `mdp: obs에 신호 상태 3차원 추가`, `fix: 우회전 슬립 teleport 제거`

## 5. 리뷰

- PR은 작게, 하루 이틀 단위로 올립니다.
- PR을 만들 때 저장소 관리자를 리뷰어로 지정합니다. 다른 사람이 담당하는 파일을 고쳤으면 그 사람도 추가합니다.
- 저장소 관리자가 올린 PR은 다른 팀원 한 명이 승인합니다.
- `env/sumo_env.py`, `env/road_config.py`처럼 여러 명이 고치는 파일은 머지한 뒤 팀에 알립니다.
- 관측 · 행동 · 보상 · 데이터 컬럼이 바뀌면 [`DATA_FORMAT.md`](./DATA_FORMAT.md)도 같이 고칩니다.
- `env/sumo/`처럼 실행할 때마다 다시 만들어지는 파일은 커밋하지 않습니다.
