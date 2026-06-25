"""Tests for MarkerDefender (Stage 4+ receiver-marking / cover defender)."""

from __future__ import annotations

import math
from collections import namedtuple
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

try:
    import pytest
except ImportError:
    pytest = None

from ai_interface.defender import OpponentInfo, KICKABLE
from ai_interface.marker_defender import (
    MarkerDefender,
    cover_position,
    mark_position,
    pass_intercept_point,
    select_mark_target,
    threat_rating,
)

GameState = namedtuple("GameState", ["count", "timestamp", "ball_pos", "robot_poses", "playmode"])


def _make_game_state(
    ball_pos: Tuple[float, float] = (0.0, 0.0),
    team_robots: Optional[Dict[str, List[dict]]] = None,
    count: int = 100,
) -> GameState:
    robot_poses = team_robots or {}
    return GameState(
        count=count,
        timestamp=0.0,
        ball_pos=ball_pos,
        robot_poses=robot_poses,
        playmode="play_on",
    )


def _opp(unum: int, x: float, y: float, theta: float = 0.0) -> OpponentInfo:
    return OpponentInfo(team="TritonBots", unum=unum, pose=(x, y, theta))


class TestThreatRating:
    def test_central_near_goal_scores_high(self):
        opp = _opp(1, 40.0, 0.0)
        goal = (45.0, 0.0)
        ball = (0.0, 0.0)
        score = threat_rating(opp, ball, goal)
        assert score > 1.5

    def test_far_wide_scores_low(self):
        opp = _opp(1, -30.0, 25.0)
        goal = (45.0, 0.0)
        ball = (0.0, 0.0)
        score = threat_rating(opp, ball, goal)
        assert score < 1.0

    def test_closer_to_goal_scores_higher(self):
        goal = (45.0, 0.0)
        ball = (20.0, 0.0)
        near = threat_rating(_opp(1, 40.0, 0.0), ball, goal)
        far = threat_rating(_opp(2, 5.0, 15.0), ball, goal)
        assert near > far


class TestSelectMarkTarget:
    def test_excludes_ball_handler(self):
        handler = _opp(1, 10.0, 0.0)
        gs = _make_game_state(
            ball_pos=(10.0, 0.0),
            team_robots={"TritonBots": [{1: (10.0, 0.0, 0.0)}, {2: (20.0, 5.0, 0.0)}]},
        )
        target = select_mark_target("TeamB", (10.0, 0.0), handler, (45.0, 0.0), gs)
        assert target is not None
        assert target.unum == 2

    def test_returns_none_when_only_handler(self):
        handler = _opp(1, 10.0, 0.0)
        gs = _make_game_state(
            ball_pos=(10.0, 0.0),
            team_robots={"TritonBots": [{1: (10.0, 0.0, 0.0)}]},
        )
        target = select_mark_target("TeamB", (10.0, 0.0), handler, (45.0, 0.0), gs)
        assert target is None

    def test_returns_none_no_opponents(self):
        gs = _make_game_state(ball_pos=(10.0, 0.0), team_robots={})
        target = select_mark_target("TeamB", (10.0, 0.0), None, (45.0, 0.0), gs)
        assert target is None


class TestMarkPosition:
    def test_goal_side_of_target(self):
        target = _opp(2, 20.0, 5.0)
        pos = mark_position(target, (45.0, 0.0), "right")
        # Should be closer to goal than the target
        dist_target_to_goal = math.hypot(45.0 - 20.0, 0.0 - 5.0)
        dist_mark_to_goal = math.hypot(45.0 - pos[0], 0.0 - pos[1])
        assert dist_mark_to_goal < dist_target_to_goal

    def test_stays_outside_defense_area(self):
        target = _opp(2, 38.0, 0.0)
        pos = mark_position(target, (45.0, 0.0), "right")
        from ai_interface.defender import is_inside_own_defense_area
        assert not is_inside_own_defense_area(pos, "right")


class TestCoverPosition:
    def test_in_front_of_goal_right(self):
        pos = cover_position((10.0, 5.0), (45.0, 0.0), "right")
        assert pos[0] < 45.0
        assert pos[0] > 30.0

    def test_shifts_toward_ball(self):
        pos_left = cover_position((10.0, -10.0), (45.0, 0.0), "right")
        pos_right = cover_position((10.0, 10.0), (45.0, 0.0), "right")
        assert pos_left[1] < pos_right[1]

    def test_stays_outside_defense_area(self):
        pos = cover_position((30.0, 0.0), (45.0, 0.0), "right")
        from ai_interface.defender import is_inside_own_defense_area
        assert not is_inside_own_defense_area(pos, "right")


class TestPassInterceptPoint:
    def test_returns_none_for_slow_ball(self):
        result = pass_intercept_point((30.0, 0.0), (10.0, 0.0), (0.1, 0.0), "right")
        assert result is None

    def test_returns_point_for_fast_ball(self):
        result = pass_intercept_point((20.0, 0.0), (10.0, 0.0), (1.0, 0.0), "right")
        assert result is not None
        assert result[0] > 10.0


class TestMarkerDefenderAction:
    def _make_defender(self) -> MarkerDefender:
        return MarkerDefender(
            teamname="TeamB", unum=3, side="right", ball_defender_robot_id=2
        )

    def test_cover_mode_no_opponents(self):
        d = self._make_defender()
        gs = _make_game_state(ball_pos=(0.0, 0.0), team_robots={
            "TeamB": [{1: (42.0, 0.0, 0.0)}, {2: (30.0, 0.0, 0.0)}, {3: (35.0, 5.0, 0.0)}],
        })
        cmd = d.action((0.0, 0.0), (35.0, 5.0, 0.0), gs)
        assert d.mode == "COVER"
        assert cmd.startswith("dash") or cmd.startswith("turn")

    def test_mark_mode_with_opponent(self):
        d = self._make_defender()
        gs = _make_game_state(
            ball_pos=(10.0, 0.0),
            team_robots={
                "TeamB": [{1: (42.0, 0.0, 0.0)}, {2: (25.0, 0.0, 0.0)}, {3: (35.0, 5.0, 0.0)}],
                "TritonBots": [{1: (10.0, 0.0, 0.0)}, {2: (20.0, 8.0, 0.0)}],
            },
        )
        cmd = d.action((10.0, 0.0), (35.0, 5.0, 0.0), gs)
        assert d.mode in ("MARK", "INTERCEPT_PASS")

    def test_clear_when_ball_at_feet(self):
        d = self._make_defender()
        ball_pos = (30.0, 5.0)
        def_pose = (30.0, 5.0, 0.0)
        gs = _make_game_state(
            ball_pos=ball_pos,
            team_robots={
                "TeamB": [{1: (42.0, 0.0, 0.0)}, {2: (25.0, 0.0, 0.0)}, {3: def_pose}],
            },
        )
        cmd = d.action(ball_pos, def_pose, gs)
        assert d.mode == "CLEAR"

    def test_no_clear_in_defense_area(self):
        d = self._make_defender()
        ball_pos = (38.0, 0.0)
        def_pose = (38.0, 0.0, 0.0)
        gs = _make_game_state(
            ball_pos=ball_pos,
            team_robots={
                "TeamB": [{1: (42.0, 0.0, 0.0)}, {2: (25.0, 0.0, 0.0)}, {3: def_pose}],
            },
        )
        cmd = d.action(ball_pos, def_pose, gs)
        assert d.mode != "CLEAR"
