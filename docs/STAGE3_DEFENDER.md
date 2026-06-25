# Stage 3 Defender

**Date:** 2026-06-25
**Scope:** Hardcoded field defender used in Stage 3: one learned attacker vs one hardcoded defender plus one hardcoded goalie.

This document describes the current `ai_interface/defender.py` behavior, why it was built this way, and the rule/simulator assumptions around pressure, snatching, clearing, and future multi-defender stages.

Code:
- Defender controller: [`ai_interface/defender.py`](../ai_interface/defender.py)
- Defender tests: [`tests/test_defender.py`](../tests/test_defender.py)
- Shared movement and catch/dribble helpers: [`ai_interface/utils/basic_commands.py`](../ai_interface/utils/basic_commands.py)
- Patched simulator catch-glue notes: [`docs/PATCH_RCSSSERVER.md`](PATCH_RCSSSERVER.md)
- SSL rules reference: [`sslrules.md`](../sslrules.md)

---

## 1. Goal

The old Stage 3 defender was mostly passive: it tried to sit in the shot lane and block shots, but it did not actively pressure the ball carrier, disrupt passes, or clear the ball after winning it.

The new defender is a small state machine built around the idea of defending threats:

- protect the line from the active threat to our goal;
- pressure a controlled ball when it enters the dangerous half;
- intercept goalward shots or likely passes;
- clear immediately when the defender wins the ball;
- back off from prolonged shared contact to avoid pushing/holding behavior.

For Stage 3, only one field defender is active. The helper functions are intentionally reusable for later stages with two or three defenders.

---

## 2. Coordinate And Side Assumptions

The defender is initialized with:

```python
Defender(teamname="TeamB", unum=2, side="right")
```

For `side="right"`:

- the defended goal is the right goal;
- goalward ball velocity means positive x velocity;
- a safe clearance is generally upfield toward the left side.

For `side="left"`, the signs invert.

The implementation uses the repo's simulator field constants:

- `FIELD_X`, `FIELD_Y` from `ai_interface/constants/field_constants.py`;
- `GOAL_L`, `GOAL_R` from the same file;
- `PLAYER_SIZE`, `BALL_SIZE`, and `KICKABLE_MARGIN` from `ai_interface/constants/player_constants.py`.

The kickable distance used by the defender is:

```python
KICKABLE = KICKABLE_MARGIN + PLAYER_SIZE + BALL_SIZE
```

---

## 3. State Machine

The defender has five modes:

| Mode | Purpose | Main trigger | Main command shape |
|---|---|---|---|
| `PROTECT` | Default line defense | no immediate shot, pass, clear, or pressure condition | `dash` / `turn` to a threat-line point |
| `PRESS` | Pressure the attacker with controlled possession | opponent handler detected, ball in danger zone, defender close enough | `dash` / `turn` to a goal-side lateral pressure point |
| `INTERCEPT` | Cut off a goalward ball or likely pass | ball velocity projects toward goal or receiver | fast `dash` / `turn` to interception/receiver point |
| `CLEAR` | Remove the ball from danger | defender is kickable to ball, outside own defense area | `kick`, or short align/drop fallback |
| `RECOVER` | Break shared-contact/pushing loop | both defender and attacker are on ball for too many cycles | back-off `dash` / `turn` |

The priority order inside `Defender.action()` is:

1. Update ball velocity.
2. Detect opponent ball handler.
3. Update shared-contact guard.
4. If recovery is active, run `RECOVER`.
5. If defender can legally play the ball, run `CLEAR`.
6. If a goalward shot/pass can be intercepted, run `INTERCEPT`.
7. If the attacker is carrying in the danger zone, run `PRESS`.
8. Otherwise run `PROTECT`.

This order is intentional. Clearing a legal ball is more urgent than shape. Interception is more urgent than ordinary pressure. Recovery overrides all because it is the rule-safety escape.

---

## 4. Perception Helpers

### `detect_ball_handler()`

Detects the nearest opponent whose distance to the ball is within:

```python
KICKABLE + possession_margin
```

Default possession margin is `0.35`.

This is a simple Stage 3 possession heuristic. It does not know the simulator's internal `M_ball_catcher` owner directly. It infers possession from geometry.

### `predict_receiver()`

Predicts a likely opponent receiver from ball trajectory.

The helper:

- requires ball speed above `RECEIVER_MIN_BALL_SPEED`;
- projects the ball forward for `INTERCEPT_LOOKAHEAD_STEPS`;
- checks opponents near the ball path segment;
- chooses the candidate closest to the path, then closest to the ball.

Stage 3 has no teammate receiver for the attacker, but this helper is already present for later 2v2/3v3 stages.

### `threat_line()`

Chooses the current threat source and target:

1. predicted receiver to defended goal;
2. current ball handler/ball to defended goal;
3. goalward moving ball to projected goal-mouth point;
4. fallback ball to defended goal.

This is the simplified Sumatra-style idea: defend the active threat line, not a fixed static point.

### `protection_point()`

Places the defender on the line from threat source to defended goal, using a fixed standoff:

```python
PROTECT_STANDOFF = 4.0
```

The result is clamped outside the field margin and outside the own defense area.

### `pressure_point()`

Builds a legal pressure target when an attacker controls the ball.

The target is:

- goal-side of the ball, so the defender does not open a direct shot lane;
- laterally offset, so the defender does not drive straight through the attacker;
- separated from the attacker by at least `MIN_ATTACKER_SEPARATION`.

This creates a poke/contain shape, not a head-on ram.

### `safe_clear_target()`

Selects an in-field target away from the defended goal.

It biases toward the opposite half and shifts laterally if an opponent blocks the direct clearance lane. It intentionally avoids field boundaries because SSL rules forbid kicking the ball out over the boundary.

---

## 5. Mode Details

### 5.1 `PROTECT`

`PROTECT` is the default mode.

The defender:

1. builds a threat with `threat_line()`;
2. computes a legal `protection_point()`;
3. shifts that point by some lateral ball velocity using `_mimic_threat_lateral_velocity()`;
4. moves there while facing the ball.

This makes the defender slide across the shot lane instead of chasing the ball blindly.

Important constants:

```python
PROTECT_STANDOFF = 4.0
PROTECT_SPEED = 72.0
```

### 5.2 `PRESS`

`PRESS` activates when:

- an opponent ball handler is detected;
- the ball is in the danger zone;
- the defender is within `PRESS_RADIUS` of the ball.

For a right-side defender, the danger zone is:

```python
ball_x >= DANGER_X
```

where:

```python
DANGER_X = 14.0
PRESS_RADIUS = 7.0
```

The defender moves to `pressure_point()`, which is goal-side/lateral rather than directly through the attacker.

Speed is reduced near contact:

```python
PRESS_SPEED = 58.0
PRESS_CONTACT_SPEED = 42.0
```

This keeps pressure strong enough to disrupt the attacker, but less likely to create a pushing/crashing artifact.

### 5.3 `INTERCEPT`

`INTERCEPT` has two cases.

Goalward shot interception:

- ball velocity must point toward the defended goal;
- projected y at the goal line must be near the goal mouth;
- the defender chooses the earliest reachable point on the ball path.

Receiver/pass interception:

- `predict_receiver()` finds an opponent near the current ball trajectory;
- defender moves to the predicted receiver point and faces the ball.

Important constants:

```python
BALL_SPEED_INTERCEPT = 0.28
GOAL_SHOT_Y_MARGIN = 7.5
RECEIVER_MIN_BALL_SPEED = 0.55
INTERCEPT_LOOKAHEAD_STEPS = 16
DEFENDER_REACH_PER_STEP = 0.45
INTERCEPT_SPEED = 95.0
```

`INTERCEPT` uses `full_speed=True`, which allows faster emergency movement. This is deliberately reserved for moving balls, not ordinary attacker contact.

### 5.4 `CLEAR`

`CLEAR` activates when:

- defender is within `KICKABLE` distance to the ball;
- defender is not inside its own defense area;
- the ball is not inside the defender's own defense area.

The defender chooses `safe_clear_target()` and calls `self.kick()` toward it with high power.

The intended behavior is immediate clearance. The defender should not become a dribbling possession player in Stage 3.

There is a short emergency hold guard:

```python
CLEAR_HOLD_CYCLE_LIMIT = 8
CLEAR_CARRY_LIMIT = 0.75
```

If the defender somehow enters a dribbling/catch state while clearing, it must kick quickly if aligned or drop the ball. This keeps it under the SSL excessive-dribbling limit and prevents glued-ball stalls.

### 5.5 `RECOVER`

`RECOVER` activates when both defender and opponent handler are simultaneously near the ball for too many cycles:

```python
SHARED_CONTACT_LIMIT = 5
RECOVER_CYCLES = 7
```

The defender:

- clears any internal dribbling state;
- stops trying to clear;
- moves a few meters goal-side of the ball;
- faces the ball.

This is not a full SSL pushing detector. It is a controller-level safety rule that breaks contact loops before they become persistent pushing/holding behavior.

---

## 6. Rule Compliance

The defender is designed around the practical parts of `sslrules.md` that matter for Stage 3.

### 6.1 Pushing

SSL pushing is a stopping foul when robots keep contact with the ball or each other while one robot exerts force such that both travel toward the opponent.

The current defender reduces pushing risk by:

- pressuring from a goal-side/lateral point instead of directly through the attacker;
- reducing speed near attacker contact;
- entering `RECOVER` after repeated shared ball contact.

The current implementation does not compute a full referee-grade pushing foul. It does not measure force directly. It uses geometry and duration as a practical approximation.

### 6.2 Crashing

SSL crashing is based on projected relative speed at collision:

- relative speed is projected onto the robot-to-robot line;
- if the projection is greater than `1.5 m/s`, the faster robot is at fault;
- if absolute robot speed difference is below `0.3 m/s`, both conduct a foul.

The current defender does not yet measure robot velocities for a formal crash penalty. It reduces crash risk through speed caps in `PRESS` and by reserving emergency full speed for ball-path interception.

Future reward/termination code should implement this explicitly if we want training to learn crash avoidance.

### 6.3 Ball Holding

SSL says robots must not surround the ball to prevent access by others.

Stage 3 has only one defender, so multi-defender surround behavior cannot occur yet. Future 2/3 defender logic must allow only one active ball challenger and keep other defenders on receiver/line roles.

### 6.4 Defense Area

Non-goalie defenders must make best effort to stay outside their own defense area, and if a non-goalie fully inside its own defense area touches the ball, the opponent gets a penalty kick.

The defender:

- clamps targets outside its own defense area;
- refuses `CLEAR` if the defender or the ball is inside its own defense area.

The goalie owns deep goal-mouth defense.

### 6.5 Boundary Crossing

SSL forbids kicking the ball over the field boundary.

The defender's `safe_clear_target()` chooses an in-field target with margin from the touch lines and goal lines. This does not guarantee a simulated kick can never leave the field after bounces, but it avoids intentional boundary clearances.

### 6.6 Excessive Dribbling

SSL limits continuous dribbling to 1 m from first contact before observable separation.

The defender is not intended to dribble. If it enters a dribble/catch state while clearing, it uses:

- max carry distance `0.75`;
- max hold cycles `8`;
- then immediate kick or drop.

This is stricter than the 1 m rule.

---

## 7. Catch-Glue Policy

The patched `rcssserver` allows non-goalie `catch 0` to act as a dribble grab. Internally, this sets `M_ball_catcher` and pins the ball to the robot front each cycle until kick/drop/contest clears it.

This is useful as a training abstraction for attacker `dribble_to`, but it is stronger than real SSL dribbling. A real dribbler can keep contact with the ball, but the rules require:

- another robot must be able to remove the ball;
- the robot must not fully control all degrees of freedom;
- most of the ball must remain outside the robot convex hull.

### Defender should not use catch-glue as the steal

The defender should not win contested balls by driving into the attacker and issuing `catch 0`. That would train against a simulator artifact, not realistic SSL defense.

Instead, the defender should:

1. move to a goal-side/lateral pressure point;
2. face or move toward the exposed carried ball;
3. let the simulator's contest logic clear the attacker's `M_ball_catcher`;
4. kick or briefly settle/clear the now-loose ball.

The simulator contest logic clears catch ownership when an opponent reaches the carried ball position and is facing or moving toward it. That means the defender can snatch without using its own catch-glue as the primary steal mechanism.

### Defender may briefly settle after winning

After the ball is loose or defender-only kickable, a short catch/dribble settle is acceptable if it immediately leads to a clear and stays under the strict carry guard. This approximates a real defender using its dribbler to control a loose ball before clearing.

The current `CLEAR` path calls `Player.kick()`, and `Player.kick()` can fall back to `dribble()`/`catch 0` while aligning if not already facing the target. The guard in `_clear()` limits that behavior and forces kick/drop quickly.

If later testing shows defenders winning too much through catch-glue, the next change should be:

- disable `catch 0` fallback in `CLEAR` when an opponent handler is also within kickable/contact range;
- use turn/positioning until the ball is loose, then kick.

---

## 8. Simulator Collision Reality

The embedded/sim-only environment is based on `rcssserver`, not a full SSL auto-referee.

Robot and ball overlaps are physically resolved by separating objects and damping velocity. The simulator does not enforce SSL pushing, ball holding, defense-area touch, or crash fouls by itself.

Therefore:

- controller code must avoid illegal behavior;
- reward/termination code must penalize illegal behavior if we want the learned attacker to respect it;
- visual sim may show legal-looking or illegal-looking contacts without referee consequences.

This is especially important for Stage 3 because an active defender can expose unrealistic possession and contact artifacts more than a passive shot blocker.

---

## 9. Suggested Training Penalties

These are not implemented in `defender.py`; they are recommended environment/reward additions.

### Pushing penalty

Penalize the responsible robot when all are true:

1. defender and attacker are in contact, or both are simultaneously on the ball;
2. the robot's velocity/dash direction is toward the opponent or through the ball into the opponent;
3. the contacted pair moves in the pushed direction;
4. the condition persists for several consecutive cycles, e.g. 3 to 5 frames.

Do not penalize ordinary incidental bumps. If both robots push with similar force, SSL says no team is at fault.

### Crashing penalty

At robot-robot collision:

1. compute relative velocity;
2. project it onto the line connecting the robots;
3. if projected speed is above `1.5 m/s`, penalize the faster robot;
4. if absolute speed difference is below `0.3 m/s`, both can be considered at fault.

### Catch-glue contest penalty

Penalize glued-ball no-progress loops when:

- ball remains near one robot for many cycles;
- an opponent is contesting;
- ball displacement/progress stays near zero.

This approximates SSL no-progress and prevents policies from farming possession without creating a shot or clearance.

---

## 10. Future Multi-Defender Extension

Stage 3 uses one defender only. Later stages should reuse the same helper functions with role assignment.

### Two defenders

- Defender 1: `BALL/PROTECT`
  - handles active ball threat;
  - presses only if it is the assigned active challenger;
  - clears if it wins the ball.
- Defender 2: `MARKER`
  - marks the most dangerous receiver;
  - sits on receiver-to-goal or pass protection line;
  - disrupts passes in flight.

### Three defenders

- Defender 1: ball pressure/protection;
- Defender 2: receiver marker/pass disruptor;
- Defender 3: center-back line defender near the goal-mouth protection line.

### Assignment priority

Use this priority:

1. pass disruption if a pass is already in flight;
2. ball threat protection/pressure;
3. receiver marking;
4. center-back line cover.

Only one non-goalie defender should actively challenge the ball at a time. Other defenders should spread along protection lines or mark receivers. This avoids ball surrounding and makes the defense stronger than multiple robots chasing the same point.

---

## 11. Tests

Current focused tests live in `tests/test_defender.py` and cover:

- protection point avoids own defense area;
- pressure point is goal-side/lateral and separated from attacker;
- ball handler detection;
- receiver prediction;
- safe clearance target;
- `PROTECT`, `PRESS`, `INTERCEPT`, `CLEAR`, and `RECOVER` modes;
- no clear when ball is inside own defense area.

Run:

```bash
python -m pytest tests/test_defender.py
```

Also run compile checks after editing defender code:

```bash
python -m py_compile ai_interface/defender.py tests/test_defender.py
```

---

## 12. Current Limitations

- No full SSL referee model is implemented in the simulator or environment.
- `detect_ball_handler()` uses geometry, not the simulator's internal catch owner.
- Crash/pushing penalties are not yet implemented in reward/termination.
- `CLEAR` can still briefly use `catch 0` via `Player.kick()` alignment fallback.
- Receiver prediction exists but is mostly future-facing in Stage 3 because the attacker has no teammate receiver.
- Multi-defender role assignment is not implemented yet.

The defender is therefore best understood as a strong, rule-conscious hardcoded opponent for Stage 3 training, not a complete SSL team defense stack.
