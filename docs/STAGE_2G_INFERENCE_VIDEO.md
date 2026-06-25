# Stage 2g inference video review

Source: `/Users/Adnan/Desktop/Stage 2g inference.mov`

## Method and interpretation

The silent 6:08 recording was reviewed as a full-video scan followed by a 1 FPS episode-level pass. Timestamps are therefore approximate to about one second.

- A score increase on the `TritonBots` scoreboard is recorded as a goal.
- A field reset without a score increase is recorded as a non-goal reset. The precise environment termination reason cannot be recovered from the video alone.
- The orange controlled player briefly changes colour near the ball in some episodes. This is described as a visible catch/hold-state transition, but the exact internal action cannot be proven from colour alone.
- “Direct approach and finish” means the player runs to the stationary ball and promptly sends it toward the goal; no meaningful pre-shot ball transport is visible.

## Episode log

| Episode | Approx. time | Result | Observed model behaviour |
| ---: | --- | --- | --- |
| 1 | 00:00–00:13 | Goal (0→1) | Approaches the stationary ball directly and finishes quickly. No sustained dribble is visible. |
| 2 | 00:13–00:26 | Goal (1→2) | Direct approach and prompt finish. |
| 3 | 00:26–00:32 | Goal (2→3) | Fast direct approach and finish. |
| 4 | 00:32–00:39 | Non-goal reset | Reaches the ball on the attacking half, but the episode resets before a useful transport or successful finish is visible. |
| 5 | 00:39–00:59 | Non-goal reset | Approaches the ball, then stalls near the right side. A brief catch/hold-state transition is visible, but there is no meaningful transport or shot before reset. |
| 6 | 00:59–01:06 | Goal (3→4) | Direct approach and finish. |
| 7 | 01:06–01:12 | Goal (4→5) | Direct approach and finish. |
| 8 | 01:12–01:18 | Goal (5→6) | Direct approach and finish. |
| 9 | 01:18–01:25 | Goal (6→7) | Direct approach and finish. |
| 10 | 01:25–01:32 | Goal (7→8) | Approaches the ball, briefly enters the visible catch/hold state, then finishes. Any transport before release is very short. |
| 11 | 01:32–01:39 | Goal (8→9) | Direct approach and finish. |
| 12 | 01:39–01:45 | Goal (9→10) | Direct approach and finish. |
| 13 | 01:45–01:53 | Goal (10→11) | Direct approach and finish. |
| 14 | 01:53–02:00 | Goal (11→12) | Direct approach and finish. |
| 15 | 02:00–02:07 | Goal (12→13) | Direct approach and finish. |
| 16 | 02:07–02:14 | Goal (13→14) | Direct approach and finish. |
| 17 | 02:14–02:22 | Goal (14→15) | Direct approach and finish. |
| 18 | 02:22–02:58 | Non-goal reset | Reaches the ball, then remains almost stationary beside it for roughly 28 seconds. Repeated colour/state cycling is visible, with no meaningful ball transport or shot. |
| 19 | 02:58–03:34 | Non-goal reset | Repeats the same close-ball failure: approach, prolonged stationary catch/hold-state cycling, no transport, and no finish. |
| 20 | 03:34–03:48 | Goal (15→16) | Returns to a direct approach and successful finish. |
| 21 | 03:48–04:00 | Goal (16→17) | Direct approach and finish. |
| 22 | 04:00–04:14 | Non-goal reset | Approaches and settles next to the ball near the right side, briefly changes state, then resets without transporting or shooting successfully. |
| 23 | 04:14–04:21 | Goal (17→18) | Direct approach and finish. |
| 24 | 04:21–04:29 | Non-goal reset | Reaches the ball but resets with the score unchanged; no useful transport or successful shot is visible. |
| 25 | 04:29–04:39 | Goal (18→19) | Direct approach and finish. |
| 26 | 04:39–04:46 | Goal (19→20) | Direct approach and finish. |
| 27 | 04:46–04:53 | Goal (20→21) | Direct approach; a visible catch/hold-state transition occurs immediately before the successful finish. |
| 28 | 04:53–05:01 | Non-goal reset | Approaches the ball and stops close to it; the field resets with no score increase. |
| 29 | 05:01–05:09 | Goal (21→22) | Direct approach and finish. |
| 30 | 05:09–05:21 | Goal (22→23) | Approaches, pauses near the ball, visibly enters the catch/hold state, and then completes a successful finish. Little displacement occurs before release. |
| 31 | 05:21–05:29 | Goal (23→24) | Direct approach and finish. |
| 32 | 05:29–05:35 | Non-goal reset | Fast approach followed by a reset before a successful finish; no meaningful transport is visible. |
| 33 | 05:35–05:46 | Goal (24→25) | Approaches, spends several seconds close to the ball with a catch/hold-state transition, then finishes successfully. Ball transport remains minimal. |
| 34 | 05:46–05:53 | Goal (25→26) | Direct approach and finish. |
| 35 | 05:53–06:05 | Incomplete | Approaches the ball and becomes nearly stationary close to it. No shot or reset occurs before the simulator is closed. |

## Overall behaviour

- Completed episodes: **34** — **26 goals** and **8 non-goal resets** (76.5% observed goal rate).
- The policy overwhelmingly uses a direct run-to-ball-and-finish pattern. It does not visibly perform meaningful goal-directed ball transport before most shots.
- Some successful episodes include a short catch/hold-state transition, but the player releases from approximately the same area; these do not demonstrate the sustained `GRAB → CARRY → RELEASE` behaviour described for Stage 2g.
- The clearest failure mode is getting stuck beside the ball while cycling state, especially episodes 18 and 19. Episodes 5, 22, 28, and the incomplete final episode show shorter versions of the same pattern.
- No obvious long-distance wrong-way carry is visible. The main issue is failure to carry at all: either the policy shoots directly or stalls at the ball.
