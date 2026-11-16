# 문헌 대조 중 발견한 구현 점검 사항

조사일:2026-11-10. 범위: 현재 작업 소스의 단계별 보상 참조. **문헌 결과와 별도인 로컬 코드 관찰**이다. 이번 조사에서 학습 소스·설정·체크포인트는 수정하지 않았다. CPU 함수 계산만 실행했으며 PPO/물리 rollout은0이다.

## 1. 단계별 phase profile 전달 누락

현재 [gallop_env.py](../../isaac_task/gallop_env.py)에서:

- `begin_step()`, line378: `phase_targets(..., profile=self.profile)`로 단계별 주기/접촉 참조를 만든다.
- `update_measurements()`, line456: `phase_targets(self.phase,self.command[:,0],self.cfg,self.strength)`로 호출하여 **profile을 전달하지 않는다**.
- 그 결과 `self.expected`, `qref`, `height_ref`, contact/joint/swing 점수와 `air_target`이 기본 `self.cfg.touchdown/duty`를 사용한다.
- 반면 flight tracker에는 `self.profile`이 연결되어 있고, phase 진행 주기와 flight timing은 단계 변경을 반영한다.

`phase_targets`는 [gallop_core.py](../../isaac_task/gallop_core.py) line558에서 profile이 있을 때만 touchdown/duty/주기 변경을 반영한다.

즉 **설계된 긴 공중기 시간표와 일부 스타일 보상의 시간표가 다를 수 있다.** 어느 것이 실제 학습 실패의 주원인인지는 이번 CPU 비교만으로 판정할 수 없다.

## 2. 비학습 CPU 재현

동일 phase1000개(0, .001, …, .999), 동일 해당 단계 속도, strength1에서 profile 유무를 비교했다. contact 불일치율의 분모는 **1000phase×4다리**이며 실제 episode 실패율이 아니다.

| 단계 index | 거리 목표 | profile 적용 주기 | 미적용 주기 | 다리×phase 접촉 기준 불일치 | 발 높이 참조 최대 차이 |
|---|---:|---:|---:|---:|---:|
|0|18cm|.390s|.390s|0%|0cm|
|1|18cm|.385s|.385s|0%|0cm|
|2|18cm|.380s|.380s|0%|0cm|
|3|20cm|.380s|.380s|0%|0cm|
|4|30cm|.435s|.380s|7.825%|3.65cm|
|5|40cm|.473s|.380s|13.275%|5.65cm|
|6|50cm|.508s|.380s|18.175%|7.09cm|
|7|65cm|.552s|.380s|24.600%|8.46cm|
|8|80cm|.600s|.380s|31.475%|9.40cm|

여기서 발 높이 차이는 **reference 함수 차이**이며 로봇이 실제 그만큼 높이 들었다는 측정이 아니다. period 차이도 두 함수의 반환값 비교다. 물리 상태를 재생한 결과로 읽으면 안 된다.

초기18cm와20cm까지는 template alpha0이므로 이 차이가 없다. 따라서 기존18cm 파일럿 결과를 이 발견으로 소급 무효화하지 않는다.

수치 원본: [implementation_audit.json](implementation_audit.json). 당시 소스 SHA256: [project_sources.json](project_sources.json).

재현 명령(프로젝트 루트에서 CPU만 사용):

```bash
/home/cvr/miniconda3/envs/kine2go/bin/python - <<'PY'
from pathlib import Path
import torch
from isaac_task.gallop_core import (
    GallopSettings, StagedEpisodeProfile, staged_gallop_plan, phase_targets,
)
s = GallopSettings.load(Path("configs/gallop_staged_80cm.json"))
phase = torch.arange(1000) / 1000
profile = StagedEpisodeProfile(phase, s, staged_gallop_plan())
for index, goal in enumerate(profile.plan):
    profile.latch(torch.arange(len(phase)), index)
    speed = phase.new_full(phase.shape, goal["speed"])
    intended = phase_targets(phase, speed, s, 1., profile=profile)
    current = phase_targets(phase, speed, s, 1.)
    mismatch = (intended["stance"] != current["stance"]).float().mean()
    height_difference = (intended["foot_height"] - current["foot_height"]).abs().max()
    print(index, goal["leap"], float(mismatch), float(height_difference))
PY
```

## 3. pair stance 허용 차이도 기본 설정 사용

`PairBalanceTracker`는 초기 base settings를 보관하고 `pair_cycle_penalties`는 base `settings.duty`에서 허용 차이를 계산한다.

현재 pair stance 비용 허용 차이는 기본 duty를 사용해 앞0.30/뒤0.25다. 최종 stage duty와 같은 공식을 사용하면 앞0.21/뒤0.15다. 이 항목도 단계별 duty와 어떤 관계여야 하는지 정의가 필요하다. 현재 값이 더 느슨하며, 실제 성능 영향은 검증하지 않았다.

관련 위치: `gallop_env.py` line294; `gallop_core.py` line939/957/974.

## 4. 문헌과 연결되는 의미

[P06](https://arxiv.org/html/2402.08662v2)은 phase 관측과 phase-dependent reward를 함께 맞추는 것의 중요성을 보여준다. [P01](https://arxiv.org/html/2403.10723v4)·[P04](https://arxiv.org/html/2212.03238)도 gait 기준과 접촉 보상 기준을 연결한다. 문헌의 효과가 현재 구현에 자동 재현된다고 볼 수 없으며, 우선 각 단계의 의도한 시간표가 보상에 실제 전달되는지 확인해야 한다.

후속 구현 작업에서는 profile 전달을 일치시키고, 단계별 접촉·joint/height reference·flight window가 맞는지 검증한 뒤 적용하는 것이 타당하다. 기존 실행은 실행별 source snapshot으로 별도 판정해야 하며 현재 작업 파일의 수정이 이미 학습 중인 프로세스에 소급 적용되지는 않는다.

