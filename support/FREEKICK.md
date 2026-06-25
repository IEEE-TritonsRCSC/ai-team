# Division B FreeKick State Implementation Guide

This file is scoped to the code that implements only the `FreeKick` state for RoboCup SSL Division B. Other states (`Stop`, `Halt`, `BallPlacement`, `Kickoff`, `Penalty`, `ForceStart`, `Running`) and transitions between states are assumed to be handled by other modules.

The `FreeKick` state starts when an external referee/game-state program has already selected this state and provided the attacking team. It ends by emitting events such as `BALL_IN_PLAY`, `DOUBLE_TOUCH`, `DEFENDER_TOO_CLOSE`, or `ATTACKER_TOO_CLOSE_TO_DEFENSE_AREA`; another program decides the next state.

## Division B Constants

| Constant | Value | Meaning |
| --- | ---: | --- |
| `FREEKICK_TIMEOUT` | `10 s` | In Division B, the ball is considered in play once 10 seconds have passed after the free-kick command, even if it did not move |
| `BALL_IN_PLAY_DISTANCE` | `0.05 m` | Ball movement required for the free kick to enter play before timeout |
| `DEFENDER_MIN_BALL_DISTANCE` | `0.5 m` | Defending robots must stay this far from the ball before it is in play |
| `OPPONENT_DEFENSE_AREA_MIN_DISTANCE` | `0.2 m` | All robots must keep this far from the opponent defense area before the ball is in play |
| `FOUL_GRACE_PERIOD` | `2 s` | Grace period for repeated ball-out-of-play fouls |

Distances from a robot are measured from the nearest side of the robot, not from the center. The rules assume a robot radius of `0.09 m` when needed.

## Inputs Required By This State

The transition/game-controller layer should provide:

- `attacking_team`: team awarded the free kick.
- `defending_team`: opponent of `attacking_team`.
- `ball_position_at_command`: ball position when the free-kick command becomes active.
- Current ball pose/velocity from vision/tracking.
- Current robot poses/velocities from vision/tracking.
- Ball-touch/manipulation detection, if available.
- Field geometry, especially both defense areas.
- Current time or monotonic frame timestamp.

This state does not decide why the free kick was awarded and does not compute the restart position. Those are external responsibilities.

## Out Of Scope

Do not implement these inside the Division B `FreeKick` state:

- Choosing free-kick recipients.
- Computing ball placement positions.
- Automatic or manual ball placement.
- Stop/halt behavior before the free kick.
- State transitions after the free kick ends.
- Yellow/red-card bookkeeping.
- Foul-counter bookkeeping, except emitting foul events.
- Full goal validation.
- Other-division timing or placement requirements.

The free-kick state may still emit events that other modules use for those responsibilities.

## Rule Summary

When the free-kick command is issued:

- Attacking robots may approach the ball immediately.
- One attacking robot may shoot/take the kick. This robot is the `kicker`.
- Defending robots must stay at least `0.5 m` from the ball until the ball is in play.
- All robots must stay at least `0.2 m` from the opponent defense area until the ball is in play.
- A goal may be scored directly from a free kick.
- The ball remains out of play until it has moved at least `0.05 m` or `10 s` have elapsed.
- Once the ball is in play, free-kick positioning restrictions are lifted.
- Once the ball is in play, the kicker may not touch the ball again until another robot touches it or the game is stopped.

## Ball-In-Play Logic

The free-kick state should emit `BALL_IN_PLAY` when either condition is true:

```text
distance(ball.position, ball_position_at_command) >= 0.05 m
```

or:

```text
now - command_start_time >= 10 s
```

The `0.05 m` threshold is important for double-touch handling. The rules allow the kicker to bump the ball multiple times over a short distance while taking the kick. The kick is considered complete only after the ball has moved at least `0.05 m`.

## Kicker Detection

The `kicker` is the attacking robot that first shoots, dribbles, or otherwise deliberately manipulates the ball during this free kick.

Ball manipulation means shooting or dribbling. The ball accidentally bouncing off a robot hull is not considered manipulation.

Recommended behavior:

```text
if kicker_robot_id is null and attacking_robot_manipulates_ball:
  kicker_robot_id = robot_id
```

If your system only has touch detection, use the first attacking touch as a practical approximation, but keep the distinction in mind: accidental hull bounces are not rule-level manipulation.

## Double Touch

After the ball is in play, the kicker may not touch the ball again until:

- Another robot has touched the ball, or
- The game has stopped externally.

If the kicker touches/manipulates the ball again before another robot touches it, emit:

```text
DOUBLE_TOUCH
```

The external state/transition layer should handle the consequence: stop followed by a free kick for the opposing team from the same ball position.

Important edge case:

- Remaining in contact with the ball for more than `0.05 m` also counts as double touch, even if it was one continuous contact.

Suggested tracking:

```text
if !ball_in_play:
  if ball moved >= 0.05 m:
    ball_in_play = true
    emit BALL_IN_PLAY

if ball_in_play:
  if any_robot_other_than(kicker_robot_id) touches ball:
    other_robot_touched_after_kick = true

  if kicker_robot_id touches ball and !other_robot_touched_after_kick:
    emit DOUBLE_TOUCH
```

## Defender Too Close To Ball

Before the ball is in play, each defending robot must stay at least `0.5 m` from the ball.

If a defending robot is too close during the free kick, emit:

```text
DEFENDER_TOO_CLOSE_TO_KICK_POINT
```

Rule consequences handled externally:

- The defending team's foul counter is incremented.
- The attacking team's timer for bringing the ball into play is reset.
- The referee may decide to repeat the free kick after significant disturbance.

Implementation notes:

- This foul only applies while the ball is out of play.
- Each team has a `2 s` grace period before the same foul is raised again.
- If multiple defending robots violate the distance within the same `2 s` window, emit only one foul event.
- If a robot remains too close, emit again after the grace period expires.

Suggested tracking:

```text
if !ball_in_play:
  if any_defender_distance_to_ball < 0.5 m:
    if now - last_defender_too_close_event_time >= 2 s:
      emit DEFENDER_TOO_CLOSE_TO_KICK_POINT
      last_defender_too_close_event_time = now
      reset local free_kick_timer to 0 if your module owns the timer
```

If the timer is owned by another program, emit the event and let that program reset it.

## Too Close To Opponent Defense Area

During a free kick, before the ball is in play, all robots must keep at least `0.2 m` from the opponent defense area.

This applies to both teams. Interpret "opponent defense area" from each robot's team perspective:

- Our robots must stay at least `0.2 m` away from their defense area.
- Their robots must stay at least `0.2 m` away from our defense area.

If a team violates this for more than the grace period, emit:

```text
ATTACKER_TOO_CLOSE_TO_DEFENSE_AREA
```

The event name in the SSL rules/game event table is attacker-oriented, but the rule text says all robots must keep distance to the opponent defense area during stop and free kicks before the ball is in play.

Rule consequences handled externally:

- The foul counts toward the violating team's foul counter.
- The first such foul during a free kick stops the game normally.
- The second such foul by the same team while the game is stopped or during a free kick before the ball enters play causes an immediate halt.

Suggested tracking:

```text
if !ball_in_play:
  for each team:
    if any robot of team is within 0.2 m of opponent defense area:
      if violation has lasted >= 2 s:
        emit ATTACKER_TOO_CLOSE_TO_DEFENSE_AREA(team)
```

If your free-kick module does not own stop/halt transition decisions, do not halt directly. Emit enough information for the transition program to decide.

## Direct Goal From Free Kick

A goal may be scored directly from a free kick. The free-kick state should not reject or suppress a direct shot.

Full goal validation is outside this module, but the related rule constraints are:

- The scoring team must not have more than the allowed number of robots on the field when the ball enters the goal.
- The ball height must not exceed `0.15 m` after the last touch by the scoring team's robots.
- The scoring team must not have committed a non-stopping foul in the last `2 s` before the ball entered the goal.

If another module owns scoring, this state only needs to expose that the free kick permits direct goals.

## Events This State Should Emit

Recommended event interface:

| Event | When To Emit | External Consequence |
| --- | --- | --- |
| `BALL_IN_PLAY` | Ball moved `0.05 m` or `10 s` elapsed | Transition layer can leave `FreeKick` and enter normal play |
| `KICKER_IDENTIFIED(robot_id)` | First attacking robot manipulates/touches the ball | Useful for double-touch tracking/debugging |
| `DOUBLE_TOUCH(team, robot_id)` | Kicker touches again after ball is in play before another robot touches it | External layer stops and awards free kick to opponent |
| `DEFENDER_TOO_CLOSE_TO_KICK_POINT(team, robot_id)` | Defender is within `0.5 m` of ball before ball is in play, respecting grace period | External layer increments foul counter and resets/repeats kick as needed |
| `ATTACKER_TOO_CLOSE_TO_DEFENSE_AREA(team, robot_id)` | Robot is within `0.2 m` of opponent defense area before ball is in play, respecting grace period | External layer handles foul/stop/halt consequence |

## Suggested State Data

```text
FreeKickStateDivisionB {
  attacking_team
  defending_team
  command_start_time
  ball_position_at_command
  ball_in_play = false
  kicker_robot_id = null
  other_robot_touched_after_kick = false
  last_defender_too_close_event_time_by_team
  defense_area_violation_started_at_by_team
}
```

## Suggested Update Loop

```text
on_enter_free_kick:
  ball_in_play = false
  kicker_robot_id = null
  other_robot_touched_after_kick = false
  command_start_time = now
  ball_position_at_command = current_ball_position

each_frame:
  if !ball_in_play:
    check_defenders_are_at_least_0_5m_from_ball()
    check_all_robots_are_at_least_0_2m_from_opponent_defense_area()

    if kicker_robot_id is null and attacking_robot_manipulates_ball:
      kicker_robot_id = robot_id
      emit KICKER_IDENTIFIED(robot_id)

    if distance(ball.position, ball_position_at_command) >= 0.05m:
      ball_in_play = true
      emit BALL_IN_PLAY(reason = "ball_moved")

    else if now - command_start_time >= 10s:
      ball_in_play = true
      emit BALL_IN_PLAY(reason = "division_b_timeout")

  else:
    if any robot other than kicker_robot_id touches ball:
      other_robot_touched_after_kick = true

    if kicker_robot_id touches/manipulates ball and !other_robot_touched_after_kick:
      emit DOUBLE_TOUCH(attacking_team, kicker_robot_id)
```

## Division B Test Cases

- `BALL_IN_PLAY` is emitted when the ball moves at least `0.05 m`.
- `BALL_IN_PLAY` is emitted after `10 s` even if the ball does not move.
- The timeout used by this state is `10 s`.
- A direct shot on goal is allowed; the state does not block it.
- Kicker is recorded on the first attacking manipulation/touch.
- Repeated short contact before `0.05 m` ball movement does not trigger double touch.
- Kicker touching the ball again after `BALL_IN_PLAY` and before any other robot touch emits `DOUBLE_TOUCH`.
- Continuous kicker contact over more than `0.05 m` emits `DOUBLE_TOUCH`.
- Defender within `0.5 m` before `BALL_IN_PLAY` emits `DEFENDER_TOO_CLOSE_TO_KICK_POINT`.
- Multiple defenders too close inside one `2 s` grace window emit only one defender-too-close event.
- Defender remaining too close emits again after the `2 s` grace period.
- Robot within `0.2 m` of opponent defense area before `BALL_IN_PLAY` emits `ATTACKER_TOO_CLOSE_TO_DEFENSE_AREA` after grace handling.
- After `BALL_IN_PLAY`, free-kick-specific defender distance and opponent-defense-area restrictions are no longer enforced by this state.
