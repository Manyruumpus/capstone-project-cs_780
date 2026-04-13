import os
import math
import numpy as np
import torch
import torch.nn as nn


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class DQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


_model = None
def _load_once():
    global _model
    if _model is not None:
        return
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError("weights.pth not found")
    model = DQN(in_dim=18, n_actions=5, hidden_dim=256)
    state = torch.load(wpath, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    _model = model


# ── Sonar bit layout ──────────────────────────────────────────────────────────
# obs[0,1]   LEFT  sonar A   pos=-112  face=-90  (hard left)
# obs[2,3]   LEFT  sonar B   pos=-68   face=-90  (left)
# obs[4,5]   FWD-L sonar A   pos=-45   face=0    (forward-left outer)
# obs[6,7]   FWD-L sonar B   pos=-22   face=0    (forward-left inner)
# obs[8,9]   FWD-R sonar A   pos=+22   face=0    (forward-right inner)
# obs[10,11] FWD-R sonar B   pos=+45   face=0    (forward-right outer)
# obs[12,13] RIGHT sonar A   pos=+68   face=+90  (right)
# obs[14,15] RIGHT sonar B   pos=+112  face=+90  (hard right)
# obs[16]    IR sensor       direct forward contact
# obs[17]    stuck flag


# ══════════════════════════════════════════════════════════════════════════════
# LIVE ENV CONTEXT
# Call set_env_context(env) before policy(obs, rng)
#
# Expected env fields:
#   env.x, env.y          -> agent center
#   env.box_x, env.box_y  -> box center
#   env.angle             -> heading in degrees
#
# If these names differ in obelix.py, rename them in _get_geo().
# ══════════════════════════════════════════════════════════════════════════════
_env_ctx = None

def set_env_context(env):
    global _env_ctx
    _env_ctx = env


def _get_geo():
    if _env_ctx is None:
        return None
    try:
        ax = float(_env_ctx.x)
        ay = float(_env_ctx.y)
        bx = float(_env_ctx.box_x)
        by = float(_env_ctx.box_y)
        hd = float(_env_ctx.angle)
        return ax, ay, bx, by, hd
    except AttributeError:
        return None


# ── FSM states: SEARCH → ALIGN → APPROACH → PUSH ─────────────────────────────
_fsm_state       = "SEARCH"
_last_rng_id     = None
_align_dir       = None
_align_steps     = 0
_ALIGN_FW_EVERY  = 6
_drift_counter   = 0
_DRIFT_THRESHOLD = 2
_stuck_steps     = 0
_STUCK_LIMIT     = 4
_escape_budget   = 0
_last_obs        = None
_steps_no_signal = 0
_STALE_LIMIT     = 30
_last_action_idx = None
_repeat_count    = 0
_SPIN_LIMIT      = 17

# NEW: anti-oscillation lock
_align_lock       = 0
_ALIGN_LOCK_STEPS = 4
_last_seen_side   = None

# PID for geometric alignment
_pid_integral   = 0.0
_pid_prev_err   = 0.0
_PID_KP         = 1.2
_PID_KI         = 0.02
_PID_KD         = 0.35
_ALIGN_THRESH   = 5.0   # as requested: ±5 degrees


def _decode(obs):
    return dict(
        left_hard  = bool(obs[0])  or bool(obs[1]),
        left_soft  = bool(obs[2])  or bool(obs[3]),
        fwd_l_out  = bool(obs[4])  or bool(obs[5]),
        fwd_l_in   = bool(obs[6])  or bool(obs[7]),
        fwd_r_in   = bool(obs[8])  or bool(obs[9]),
        fwd_r_out  = bool(obs[10]) or bool(obs[11]),
        right_soft = bool(obs[12]) or bool(obs[13]),
        right_hard = bool(obs[14]) or bool(obs[15]),
        ir         = bool(obs[16]),
        stuck      = bool(obs[17]),
    )


def _reset_all():
    global _fsm_state, _align_dir, _align_steps, _drift_counter
    global _stuck_steps, _escape_budget, _last_obs, _steps_no_signal
    global _last_action_idx, _repeat_count, _pid_integral, _pid_prev_err
    global _align_lock, _last_seen_side

    _fsm_state       = "SEARCH"
    _align_dir       = None
    _align_steps     = 0
    _drift_counter   = 0
    _stuck_steps     = 0
    _escape_budget   = 0
    _last_obs        = None
    _steps_no_signal = 0
    _last_action_idx = None
    _repeat_count    = 0
    _pid_integral    = 0.0
    _pid_prev_err    = 0.0
    _align_lock      = 0
    _last_seen_side  = None


def _angle_diff(target_deg, current_deg):
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


def _pid_action(err_deg):
    global _pid_integral, _pid_prev_err

    _pid_integral = max(-60.0, min(60.0, _pid_integral + _PID_KI * err_deg))
    deriv = err_deg - _pid_prev_err
    output = _PID_KP * err_deg + _pid_integral + _PID_KD * deriv
    _pid_prev_err = err_deg

    abs_out = abs(output)
    if abs_out <= _ALIGN_THRESH:
        return "FW"

    if output > 0:
        return "L45" if abs_out > 35 else "L22"
    else:
        return "R45" if abs_out > 35 else "R22"


def _dqn_action(obs):
    global _last_obs, _steps_no_signal, _last_action_idx, _repeat_count

    has_signal = bool(np.any(obs[:17]))
    if has_signal:
        _last_obs = obs.copy()
        _steps_no_signal = 0
    else:
        _steps_no_signal += 1
        if _steps_no_signal >= _STALE_LIMIT:
            _last_obs = None

    obs_use = _last_obs if (_last_obs is not None and not has_signal) else obs
    x = torch.tensor(obs_use, dtype=torch.float32).unsqueeze(0)
    q = _model(x).squeeze(0).cpu().numpy()
    idx = int(np.argmax(q))

    if idx == _last_action_idx:
        _repeat_count += 1
    else:
        _repeat_count = 0
    _last_action_idx = idx

    if _repeat_count >= _SPIN_LIMIT and idx in {0, 1, 3, 4}:
        _repeat_count = 0
        return "FW"

    return ACTIONS[idx]


@torch.no_grad()
def policy(obs, rng=None):
    global _fsm_state, _align_dir, _align_steps, _drift_counter
    global _stuck_steps, _escape_budget, _last_rng_id
    global _steps_no_signal, _last_obs
    global _pid_integral, _pid_prev_err
    global _align_lock, _last_seen_side

    _load_once()

    rid = id(rng) if rng is not None else None
    if _last_rng_id != rid:
        _reset_all()
        _last_rng_id = rid

    obs = np.asarray(obs, dtype=np.float32)
    s = _decode(obs)

    any_left   = s["left_hard"] or s["left_soft"]
    any_right  = s["right_hard"] or s["right_soft"]
    any_fwd_l  = s["fwd_l_out"] or s["fwd_l_in"]
    any_fwd_r  = s["fwd_r_out"] or s["fwd_r_in"]
    any_fwd    = any_fwd_l or any_fwd_r
    has_signal = bool(np.any(obs[:17]))

    geo = _get_geo()

    # Stuck escape 
    if s["stuck"]:
        _stuck_steps += 1
    else:
        _stuck_steps = 0

    if _escape_budget > 0:
        _escape_budget -= 1
        return "R45" if _align_dir == "L" else "L45"

    if _stuck_steps >= _STUCK_LIMIT:
        _stuck_steps = 0
        _escape_budget = 4
        return "R45" if _align_dir == "L" else "L45"

    # Push on contact 
    if s["ir"]:
        _fsm_state = "PUSH"
        _drift_counter = 0
        _pid_integral = 0.0
        _pid_prev_err = 0.0
        return "FW"

    if _fsm_state == "PUSH":
        if any_fwd or s["ir"]:
            return "FW"
        _fsm_state = "SEARCH"

    # GEOMETRIC MODE: Detect → Align → Move using atan2 + heading error
    if geo is not None:
        ax, ay, bx, by, heading = geo

        theta_target = math.degrees(math.atan2(by - ay, bx - ax))
        err = _angle_diff(theta_target, heading)

        if not has_signal:
            _fsm_state = "SEARCH"
            _pid_integral = 0.0
            _pid_prev_err = 0.0
            return _dqn_action(obs)

        _fsm_state = "ALIGN"

        if err > 0:
            _align_dir = "L"
        elif err < 0:
            _align_dir = "R"

        if abs(err) <= _ALIGN_THRESH:
            _fsm_state = "APPROACH"
            _pid_integral = 0.0
            _pid_prev_err = 0.0
            return "FW"

        return _pid_action(err)

    # FALLBACK: sensor-only logic
    if _fsm_state == "SEARCH":
        if not has_signal:
            return _dqn_action(obs)

        _fsm_state = "ALIGN"
        _align_steps = 0

        if any_left and not any_fwd:
            _align_dir = "L"
            _align_lock = _ALIGN_LOCK_STEPS
            _last_seen_side = "L"
        elif any_right and not any_fwd:
            _align_dir = "R"
            _align_lock = _ALIGN_LOCK_STEPS
            _last_seen_side = "R"
        elif any_fwd_l and not any_fwd_r:
            _align_dir = "L"
            _last_seen_side = "L"
        elif any_fwd_r and not any_fwd_l:
            _align_dir = "R"
            _last_seen_side = "R"
        else:
            _align_dir = _last_seen_side if _last_seen_side is not None else "L"

    centred = s["fwd_l_in"] and s["fwd_r_in"]
    near_centre = (s["fwd_l_in"] or s["fwd_r_in"]) and any_fwd

    if centred or near_centre:
        _fsm_state = "APPROACH"
        _drift_counter = 0
        _align_steps = 0
        return "FW"

    if _fsm_state == "APPROACH":
        _fsm_state = "ALIGN"

    if _fsm_state == "ALIGN":
        _align_steps += 1

        if not has_signal:
            _steps_no_signal += 1
            if _steps_no_signal >= _STALE_LIMIT:
                _fsm_state = "SEARCH"
                return _dqn_action(obs)
            return "FW" if _align_dir is None else ("L22" if _align_dir == "L" else "R22")

        _steps_no_signal = 0

        side_only_left  = any_left and not any_fwd
        side_only_right = any_right and not any_fwd

        if side_only_left:
            _last_seen_side = "L"
            if _align_dir is None:
                _align_dir = "L"
            _align_lock = _ALIGN_LOCK_STEPS

        elif side_only_right:
            _last_seen_side = "R"
            if _align_dir is None:
                _align_dir = "R"
            _align_lock = _ALIGN_LOCK_STEPS

        if side_only_left or side_only_right:
            if _align_lock > 0:
                _align_lock -= 1

            if _align_dir == "L":
                return "L45" if s["left_hard"] else "L22"
            else:
                return "R45" if s["right_hard"] else "R22"

        # Fine correction when forward cones are active
        if s["fwd_l_in"] and not any_fwd_r:
            _align_dir = "L"
            _last_seen_side = "L"
        elif s["fwd_r_in"] and not any_fwd_l:
            _align_dir = "R"
            _last_seen_side = "R"
        elif any_fwd_l and not any_fwd_r:
            _align_dir = "L"
            _last_seen_side = "L"
        elif any_fwd_r and not any_fwd_l:
            _align_dir = "R"
            _last_seen_side = "R"

        if _align_dir is None:
            _align_dir = _last_seen_side if _last_seen_side is not None else "L"

        if _align_steps % _ALIGN_FW_EVERY == 0:
            return "FW"

        if _align_dir == "L":
            return "L22"
        else:
            return "R22"

    return _dqn_action(obs)