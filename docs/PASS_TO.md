# Hardcoded Pass Logic

This document describes the Stage 4 hybrid pass implementation. The design is intentionally narrow: PPO is only used as a solo finisher, while passing and receiving are deterministic.

## Goal

The Stage 3 PPO model already knows how to recover/control the ball, dribble around a defender, align, and shoot. It does not reliably learn multi-robot passing under the time constraint. The hybrid Stage 4 strategy therefore uses:

- PPO for whichever attacker has stable possession and should finish solo.
- A hardcoded supporter for off-ball receive positioning.
- A hardcoded pass lifecycle when the carrier's shot lane is blocked and the teammate has a better continuation lane.

## Tactical Gate

The gate lives in `JALTeamEnv._update_claimant_follow_mapping()`.

Each step, the env chooses one of two modes:

- `solo_finish`: PPO controls the current carrier. The other attacker is handled by the hardcoded supporter aux controller.
- `hardcoded_pass`: PPO is paused and the env-owned pass coordinator controls both attackers until the pass succeeds, misses, is intercepted, or times out.

The gate enters `hardcoded_pass` only when:

- the carrier has usable ball contact;
- the carrier-to-goal lane is blocked enough;
- the teammate is far enough away for a real pass;
- the pass lane to the teammate is clear enough;
- the teammate's continuation shot/lane score is better than the carrier's score by the configured margin.

## Pass Lifecycle

The lifecycle is implemented in `ai_interface/hybrid_pass.py` by `HardcodedPassCoordinator`.

### 1. Start

When the gate selects `hardcoded_pass`, the coordinator stores:

- `carrier_id`
- `receiver_id`
- pass `target`
- start simulator count
- phase
- fired/receive counters

The target is the receiver's current legal field position. The receiver is expected to already be near a useful support point because the normal hardcoded supporter keeps it there before the pass starts.

### 2. Prepare And Fire

The carrier executes pass-specific commands, not the generic supporter shoot/return routine.

If the carrier is not close enough to the ball, it runs `approach_ball(... obstacle_avoidance=True, avoid_ball=False, avoid_players=True)`. This avoids robots but does not treat the ball as an obstacle.

If the carrier has the ball but the ball is not in the front reception cone, it moves to a contact pose behind the ball relative to the pass target.

If the ball is in the physical front cone, it uses the geometric `kick(..., dribbling=True)` helper. This turns through the smaller normalized angle until aligned, then fires a real `kick`.

The emitted labels are:

- `hc_pass_recover`
- `hc_pass_settle`
- `hc_pass_align`
- `hc_pass_fired`

### 3. Receiver Holds The Line

Before the kick fires, the receiver moves to a receive pose and faces the incoming ball.

After the kick fires, the receiver does not go back to generic support positioning. It stays on the predicted ball line and moves to the earliest reachable intercept point. If the ball reaches the receiver's front cone, it issues `catch 0` to emulate trapping with the dribbler.

The emitted labels are:

- `hc_receive_hold`
- `hc_receive_line`
- `hc_receive_reposition`
- `hc_receive_settle`

### 4. Carrier Clears Out

After firing the pass, the original carrier moves laterally/back from the ball path using `hc_pass_clearout`. This prevents the common failure where both attackers crowd the receiver immediately after a pass.

### 5. Stable Handoff To PPO

PPO is not handed control immediately when a pass is fired. The receiver must show stable possession:

- ball within kickable distance plus margin;
- ball inside the receiver's front cone;
- condition held for the configured number of frames.

Once stable, the env maps the single PPO slot to the receiver robot and resumes `solo_finish`. The Stage 3 finisher can then dribble/align/shoot with the same skill it used as the original carrier.

If the pass times out, is intercepted, or cannot align, the coordinator clears the lifecycle and PPO resumes with the carrier or nearest valid claimant.

## Ball Speed

Pass power is chosen from a simple decaying-ball model:

- max ball speed is `4 m/s = 4 env units/step`;
- ball decay is `0.94` per simulator step;
- candidate initial speeds are simulated until they reach the target;
- the selected speed favors a catchable terminal speed near `1.25 env units/step`;
- kick power is `100 * initial_speed / 4`, clamped to the configured min/max power.

This avoids fixed-power passes that are too short at long range or too hot at short range.

## Obstacle Avoidance

`goto()` now supports separate ball and player obstacle flags:

- support movement can avoid both ball and robots;
- receive/acquire/pass-settle movement avoids robots but ignores the ball obstacle;
- this mirrors Sumatra-style receive/touch skills, where the ball must not be treated as something to detour around.

`approach_ball()` exposes the same behavior through `obstacle_avoidance`, `avoid_ball`, and `avoid_players`.

## Logging

Hardcoded pass events are recorded into the existing action distribution and debug info:

- `hc_pass_*`
- `hc_receive_*`
- `hc_pass_clearout`

Terminal debug info still includes `external_action_counts` and `external_action_events_tail`, so inference can show whether passes were requested, fired, received, missed, or crowded.

## Failure Handling

The lifecycle ends without PPO credit when:

- pass alignment exceeds the align timeout;
- fired pass exceeds the receive timeout;
- an opponent reaches the ball before the receiver;
- required carrier/receiver pose data is missing.

In all failed cases, the env returns to PPO recovery/solo-finish mode rather than leaving both attackers in hardcoded hold.
