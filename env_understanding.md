# OBELIX environment understanding note

## 1) Observation structure (18 entries, exact order used in code)

Observation is `sensor_feedback` with indices `0..17` in `obelix.py` (`get_feedback`).

For each sensor mask `i`:
- bit `2*i` = **near** hit (mask value 50)
- bit `2*i+1` = **far** hit (mask value 100)

So the exact 18-entry order is:

1. `obs[0]`  = left-side sensor A (near)
2. `obs[1]`  = left-side sensor A (far)
3. `obs[2]`  = left-side sensor B (near)
4. `obs[3]`  = left-side sensor B (far)
5. `obs[4]`  = front-left outer (near)
6. `obs[5]`  = front-left outer (far)
7. `obs[6]`  = front-left inner (near)
8. `obs[7]`  = front-left inner (far)
9. `obs[8]`  = front-right inner (near)
10. `obs[9]`  = front-right inner (far)
11. `obs[10]` = front-right outer (near)
12. `obs[11]` = front-right outer (far)
13. `obs[12]` = right-side sensor A (near)
14. `obs[13]` = right-side sensor A (far)
15. `obs[14]` = right-side sensor B (near)
16. `obs[15]` = right-side sensor B (far)
17. `obs[16]` = IR front contact/proximity bit
18. `obs[17]` = stuck flag

Notes:
- The 8 directional sonar sectors come from `sonar_positions` / `sonar_facing_angles`.
- `obs[17]` is not a sonar sensor; it is set from `self.stuck_flag`.

## 2) Action meanings

Action set (`move_options`):
- `L45`: rotate left by 45°
- `L22`: rotate left by 22.5°
- `FW`: move forward
- `R22`: rotate right by 22.5°
- `R45`: rotate right by 45°

## 3) Reward summary

From `update_reward` + `check_done_state`:
- Per-step base cost: `-1`
- One-time sensor-bit discovery bonuses (only first time each bit turns on in episode):
  - left/right side bits: `+1` each
  - forward far bits: `+2` each
  - forward near bits: `+3` each
  - IR bit (`obs[16]`): `+5`
- Stuck (`obs[17]=1`): `-200` that step
- First box attachment: `+100` (then push mode starts)
- Push-state transition also applies per-step `-1` as coded
- Success (attached box touches boundary): `+2000` and episode ends

When FW is usually rewarded:
- FW is attractive when front sensors/IR are active and movement is feasible (you often gain sensor-discovery reward and progress to attach/push).
- FW is bad if it causes stuck (`obs[17]=1`), because `-200` dominates.

## 4) Situation notes (3-column: sensor pattern, likely best action, why)

| Sensor pattern (example) | Best likely action | Why |
|---|---|---|
| All zeros (`obs[0..16]=0`, `obs[17]=0`) | `FW` (default), occasional small turn | Explore quickly; forward coverage helps discover rewarded sensor bits. |
| Far-left active (e.g., `obs[1]=1` or `obs[3]=1`) | `L22` or `L45` | Turn toward detected object side to bring it into forward sectors. |
| Far-right active (e.g., `obs[13]=1` or `obs[15]=1`) | `R22` or `R45` | Symmetric to left case: align robot toward right-side detection. |
| Near-front-left active (`obs[6]=1`) | `FW` or slight `L22` then `FW` | Object is close and mostly ahead; forward can convert to IR/attach soon. |
| Near-front-right active (`obs[8]=1`) | `FW` or slight `R22` then `FW` | Same logic on right side; minimize unnecessary rotation. |
| Front far bits only (`obs[5]`/`obs[7]`/`obs[9]`/`obs[11]`) | `FW` | Forward approach is sensible; front sensing has stronger reward weights than side sensing. |
| IR active (`obs[16]=1`, not stuck) | `FW` | High-confidence close contact/proximity; forward tends to secure attachment/push transition. |
| Attached = 1 (push enabled in environment state) and not stuck | `FW` | Objective becomes pushing attached box to boundary for terminal `+2000`. |
| Stuck flag active (`obs[17]=1`) | turn away (`L45` or `R45`) | Immediate recovery is priority; repeating FW usually repeats heavy `-200`. |
| Side sensors active but front silent (e.g., left bits on, front bits off) | turn toward active side (`L22`/`L45`) | Re-center object into forward cone before committing to FW. |

## 5) Three high-penalty bad situations to avoid

1. **Stuck against wall/obstacle while trying FW**
   - Signature: `obs[17]=1` after FW.
   - Penalty: `-200` (plus base step cost).

2. **Pushing attached box into wall/obstacle so robot cannot advance**
   - In push mode, blocked movement sets stuck.
   - Penalty pattern: repeated `obs[17]=1` events can accumulate large negative return.

3. **Trying FW at boundary without box (or when forward cell blocked)**
   - Boundary/blocked forward attempt sets stuck.
   - Penalty: same heavy `-200` per bad step.

---
This note is for mental model building only (no RL policy code yet).