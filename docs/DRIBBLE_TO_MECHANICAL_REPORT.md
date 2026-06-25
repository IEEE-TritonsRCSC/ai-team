# Dribble-To Mechanical Report

**Date:** 2026-06-23
**Audience:** mechanical team / robot-hardware review
**Scope:** current `dribble_to` functionality in the PPO JAL environment and the patched embedded simulator.

This report explains how the simulated dribble works, what it is trying to represent on the real robot, and what details are worth checking mechanically.

## Plain summary

`dribble_to` is the code's "move the ball while keeping control" behavior. The AI does not tell the robot to instantly move the ball to a point. Instead, it runs a short multi-step routine:

1. Get close to the ball using `approach_ball`.
2. Select `dribble_to` when already close enough.
3. Aim at the ball and send `catch 0`, which the embedded simulator treats as "grab/hold the ball".
4. Verify that the ball is actually moving with the robot.
5. Dash toward a chosen target while the ball is held in front of the robot.
6. Stop before the carry distance reaches about `0.85 m`.
7. Turn to a useful release direction.
8. Send `drop`, which releases the simulated hold.

The repeated pattern is: **grab, carry a short legal distance, release, then re-grab if needed**. In the current implementation, one committed `dribble_to` selection completes one carry segment; after release, the policy must choose the next approach/dribble segment.

The main idea is to approximate a real dribbler/kicker assembly: the robot gains control of the ball, moves while the ball stays constrained near its front face, then releases it so the robot is not treated as continuously carrying the ball too far.

## Why this exists

The robot soccer rules include an excessive-dribbling constraint. The notes in `DRIBBLE_TO.md` describe the practical limit as roughly `1 m` of continuous dribbling. The implementation therefore uses `segment_limit = 0.85 m`, leaving margin below the rule limit.

That matters mechanically because the behavior assumes the real robot can:

- Capture or stabilize the ball near the front of the robot.
- Move with the ball for a short segment without losing it.
- Release it cleanly.
- Reacquire it again after separation.

If any one of those physical behaviors is unreliable, the software will look good in sim but fail on the real robot.

## The current software flow

### 1. Target selection

The policy outputs a `dribble_to` target. For the current Stage 2g setup, the target is decoded relative to the ball and goal, not as a random absolute point on the field.

In simple terms:

- Forward means "toward the opponent goal".
- Lateral means "side-step the ball path to open the shooting angle around the goalie".
- The target y-position can be clipped so the robot does not aim far away from the goal mouth.

This happens in `JAL_env.py`, where the target is computed from the ball position, goal center, and the policy's raw parameters.

Mechanical interpretation: the software is not just trying to move the ball anywhere. It is trying to move it into a better shooting lane.

### 2. Dribble starts only when already close

`dribble_to` does not walk across the field to find the ball. If the robot is not already within possession/kickable range, the primitive returns `done` and the environment effectively idles that command.

That separation is deliberate:

- `approach_ball` means "go to the ball".
- `dribble_to` means "I am at the ball; start controlled transport".

Mechanical interpretation: the dribbler is only expected to work once the ball is physically close enough to the front of the robot.

### 3. The macro commits after one valid selection

Once the AI selects `dribble_to` at the ball, the environment keeps running the dribble routine across later simulator steps even if the policy would have sampled another non-kick action.

This is important because a real dribble is not a one-frame action. The robot may need several frames to align, catch, verify, carry, align for release, and drop. Only a `kick` is allowed to interrupt it.

Mechanical interpretation: once a dribble has begun, the software treats it as a short low-level maneuver, not as a decision that must be reselected perfectly every 100 ms.

## The phase machine

The current phase order is:

`GRAB -> SETTLE -> VERIFY -> CARRY -> ALIGN_RELEASE -> RELEASE -> DONE`

### GRAB

The robot turns to face the ball, then sends `catch 0`.

The simulator's catch area is body-relative, so facing the ball matters. If the robot is close but angled badly, the catch can fail.

Real-robot analogy: the intake/dribbler must be square enough to the ball for the ball to enter the capture zone.

### SETTLE

After `catch 0`, the code waits for a fresh simulator frame before trusting the ball and robot positions.

This exists because the simulator may update the ball-hold state one cycle later. Reading the same frame too early can make verification wrong.

Real-robot analogy: wait a moment after engaging the dribbler before declaring "we have possession".

### VERIFY

The robot sends a small probe dash away from the ball. The code checks whether:

- the robot moved,
- the ball moved,
- the relative robot-ball offset stayed nearly the same.

If all three are true, the catch is treated as verified and the phase opens `CARRY`.

If verification fails, the code sends `drop`, waits for the simulator to process it, and tries again up to a small retry limit.

Real-robot analogy: possession should be confirmed by the ball staying coupled to the robot during a tiny motion, not just by proximity. On the real robot, this could correspond to ball sensor confirmation, wheel-current feedback, camera-observed ball motion, or another possession signal.

### CARRY

During carry, the code uses a directional `dash` toward the latched target. It intentionally avoids turning the body during this phase.

This matters because the embedded simulator keeps the caught ball pinned in front of the robot body. If the robot turns in place while the ball is glued, the ball visibly orbits around the robot. That is not a good physical model for controlled dribbling. Directional dash translates the robot and ball together without creating large glued-ball arcs.

Real-robot analogy: while the dribbler is engaged, the robot should mainly translate the controlled ball. Large in-place rotations under possession can sweep the ball around unnaturally and may not match the mechanical dribbler's actual constraints.

### ALIGN_RELEASE

Before releasing, the robot may perform a bounded turn toward the remaining target direction. This puts the held ball in front of the robot in a useful orientation before dropping it.

The alignment is bounded so the robot cannot spin forever.

Real-robot analogy: just before releasing or preparing to kick, the robot should face a useful direction. This is a short pose correction, not the whole carry motion.

### RELEASE

The code sends `drop`. In the patched embedded simulator, `drop` clears the internal ball-holder pointer. The ball is then left at its current field position.

The software waits at least one new simulator count after `drop` before treating the release as complete.

Real-robot analogy: the dribbler/retention mechanism stops actively holding the ball, and the robot must allow the ball to separate enough that a future grab is a new possession segment.

## What the embedded simulator does

The embedded simulator has local patches that stock `rcssserver` does not have.

### Field-player catch is allowed

Normally, `catch` is a goalie concept. In this patched server, a non-goalie field player is allowed to use catch during `play_on`. This is how the code represents grabbing the ball for dribbling.

When catch succeeds, `Stadium::ballCaught()` stores that player as `M_ball_catcher`.

### The ball is pinned in front of the catcher

Every simulator cycle, if `M_ball_catcher` exists, the server moves the ball to:

`catcher position + catcher front direction * (player radius + ball radius)`

So the ball is not rolling freely during a verified catch. It is explicitly repositioned at the front of the robot every cycle.

Mechanical interpretation: this is a simplified model of a dribbler keeping the ball captured at the robot's front.

### Field-player catch no longer teleports the player

The current local server keeps stock goalkeeper catch correction for goalies, but skips that correction for field players. That means a field-player catch does not automatically move, rotate, or stop the robot. The robot has to do its own approach and alignment.

Mechanical interpretation: this is closer to the real robot, because the real robot will not magically snap into the ball's pose when possession starts.

### Drop releases the hold

The local server adds a `drop` command. If the command comes from the current catcher, it clears `M_ball_catcher`.

After that, the ball is no longer pinned to the robot and can be treated as released.

Mechanical interpretation: this represents turning off or relaxing the ball-retention behavior.

## Important sim-vs-real assumptions

These are the main points the mechanical team should cross-check.

### 1. The simulator assumes perfect ball retention after catch

Once catch is verified, the ball is pinned to the robot front every cycle. The ball does not slip, bounce, rotate out, or lag behind unless the software explicitly drops it or the sim loses ownership.

Mechanical question: can the real dribbler hold the ball reliably during short translations, especially under acceleration and lateral movement?

### 2. The simulator uses a hard release command

`drop` instantly clears the simulated hold. The real robot may not have an equally clean on/off release. The ball may stay trapped, roll unpredictably, or remain too close for a clean re-grab.

Mechanical question: when the dribbler is disabled, does the ball separate predictably enough to count as a new possession segment?

### 3. Directional carrying may not match drivetrain/dribbler limits

The code can command a directional dash while preserving body orientation. This lets the robot move toward the target without turning the ball around the body.

Mechanical question: can the drivetrain translate sideways or diagonally while the front dribbler keeps control, or is stable dribbling mostly forward-only?

If the real robot cannot carry laterally well, the software target decoder may need constraints that prefer forward arcs or smaller lateral offsets.

### 4. The 0.85 m segment assumes a reliable margin

The software releases before 1 m to avoid excessive dribbling. In the real world, measured ball travel may differ from simulator travel.

Mechanical question: what is the real measured continuous-control distance from first contact to release? Is `0.85 m` still safely below the rule threshold after sensor noise, ball lag, and robot overshoot?

### 5. Verification in sim is motion-based, not sensor-based

The code verifies catch by checking that the ball and robot move together after a small probe. Real robots need an equivalent possession signal.

Mechanical question: what signal should software trust on hardware: camera distance, ball sensor, dribbler motor current, velocity coupling, or a combination?

### 6. Turning while holding the ball is intentionally minimized

The sim produced unrealistic glued-ball orbiting when turning during carry, so the current carry phase avoids body turns until release alignment.

Mechanical question: how much in-place rotation can the real dribbler tolerate without losing control or dragging the ball around in a way that would violate the intended behavior?

## Failure modes to watch for

### In simulation

- `catch` fails because the robot is close but not facing the ball.
- Verification fails because the ball did not move with the robot.
- Carry aborts because the ball is too far from the robot.
- Carry stalls because the ball does not make progress toward target.
- Episode ends while the ball is still caught; the embedded backend rebuilds the engine on reset to clear that held-ball state.

### On the real robot

- Robot reaches the ball but intake geometry does not capture it.
- Ball is captured but slips during lateral/diagonal movement.
- Ball remains stuck after intended release.
- Robot turns while holding and loses the ball or sweeps it outside legal/controllable bounds.
- The software believes it has possession because the ball is nearby, but mechanically the ball is not actually controlled.

## Source references

- Current technical design note: `docs/DRIBBLE_TO.md`
- Dribble phase machine: `ai_interface/utils/basic_commands.py`
  - `DribbleState`
  - `dribble_to()`
  - phases `GRAB`, `SETTLE`, `VERIFY`, `CARRY`, `ALIGN_RELEASE`, `RELEASE`, `DONE`
- Environment action wiring: `ai_interface/envs/JAL_env.py`
  - goal-relative target decoding
  - macro continuation after commit
  - reward/session bookkeeping
- Reward definitions: `ai_interface/envs/reward.py`
  - dribble target quality
  - achieved gap reward
  - reachable one-segment endpoint
- Embedded simulator wrapper: `networking/socket_utils.py`
  - tracks catch/drop ownership
  - rebuilds embedded engine if reset happens while a ball is still caught
- Patched embedded simulator source:
  - `robocup_downloads/rcssserver/src/player.cpp`
    - field-player catch behavior
    - field-player catch avoids goalie pose correction
    - `Player::drop()`
  - `robocup_downloads/rcssserver/src/stadium.cpp`
    - per-cycle ball pinning to `M_ball_catcher`
    - `Stadium::ballCaught()`
  - `robocup_downloads/rcssserver/src/player_command_parser.ypp`
    - parser support for `(drop)`

## Bottom line for mechanical review

The current approach models dribbling as a controlled hold at the robot front, not as natural ball rolling. That is useful for training a high-level policy, but it assumes the real robot can reproduce three physical capabilities:

1. **Acquire:** reliably capture a nearby, front-facing ball.
2. **Carry:** keep the ball constrained during short translations, including any lateral component the policy asks for.
3. **Release:** cleanly let the ball go before the continuous-control distance exceeds the rule limit.

The biggest hardware risks are imperfect retention during lateral motion and imperfect release. If either is weak, the software should be constrained to more physically realistic carry directions, shorter segment lengths, stronger possession verification, or a slower release/reacquire routine.
