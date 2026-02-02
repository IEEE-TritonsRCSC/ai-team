# AI Team - RoboCup Soccer AI System

Python-based framework for running RoboCup Soccer Simulation League agents (and optionally bridging to real-world robots/cameras). The core AI lives in `ai_interface/` and is designed to be modular, testable, and easy to extend.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Command Line](#command-line)
- [Usage](#usage)
  - [Environment Modes](#environment-modes)
  - [Demos](#demos)
  - [Parameter Estimation](#parameter-estimation)
- [Development](#development)
  - [Game State Structure](#game-state-structure)
  - [AI Interface](#ai-interface)
    - [SoccerAI Contract](#soccerai-contract)
  - [Utilities](#utilities)
- [License](#license)
- [Contact](#contact)

## Overview

This project provides a complete stack for running soccer agents in the RoboCup Soccer Simulation Server and (optionally) camera-based field setups. The AI decision layer lives in `ai_interface/` and produces low-level command strings like `dash`, `turn`, `kick`, `catch`, and `skick` that are executed by the networking layer.

## Quick Start

**Prerequisites**: conda must be installed on your system.

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-team
```

2. Create the conda environment with all dependencies:
```bash
conda env create
```

This creates a new `rcai` Python 3.11 environment with required dependencies (e.g., `sslclient` for camera integration).

3. Activate the environment:
```bash
conda activate rcai
```

4. Run the default AI:
```bash
python .
```

## Command Line

```
python . [--team_config PATH] [--env MODE] [--demo NAME] [--estimate ball|player]
```

- `--team_config`: Path to the JSON team config (default: `team_config.json`).
- `--env`: Environment mode (default: `sim-only`).
  - `sim-only`: Simulator only
  - `sim-mixed`: Sim + physical robots
  - `field-practice`: Camera + physical robots
  - `field-tournament`: Camera + physical robots (control only your team)
- `--demo`: Run a demo AI from `ai_interface/demos/` (file name without `.py`).
- `--estimate`: Run the parameter estimator (`ball` or `player`).

## Usage

### Environment Modes

- **Simulation Only (`sim-only`)**: simulator-only setup, best for development.
- **Mixed Mode (`sim-mixed`)**: simulator + physical robots.
- **Field Practice (`field-practice`)**: camera + physical robots.
- **Tournament Mode (`field-tournament`)**: camera + physical robots; only your team is controlled.

### Demos

Demo AIs live in `ai_interface/demos/` and can be run via `--demo`:

```bash
python . --demo simple_intercept_demo
python . --demo intercept_demo
```

Rules for demo loading (see `__main__.py`):
- The file name is converted to CamelCase + `AI` to find the class.
  - Example: `simple_intercept_demo.py` -> `SimpleInterceptDemoAI`

### Parameter Estimation

`ai_interface/utils/param_estimator.py::ParamEstimatorAI` collects trajectories in sim and estimates:

- `ball_decay`
- `player_decay`
- `dash_power_rate`

Run with:
```bash
python . --estimate ball
python . --estimate player
```

The estimator uses `ai_interface/utils/estimator_config.json` for its team configuration. At the end of a run, it prints estimates plus fit metrics (RMSE and R^2). Update `ai_interface/constants/player_constants.py` and `ai_interface/constants/field_constants.py` with the results as needed.

## Development

### Game State Structure

`networking.data_utils.GameState` is a named tuple:

- `count`: server cycle number
- `timestamp`: wall time for the state
- `ball_pos`: `(x, y)`
- `robot_poses`: dict of `{team_name: [{unum: (x, y, heading_deg)}, ...]}`
- `playmode`: simulator play mode (if available)

**Angles**: simulator/camera headings are degrees; AI helpers generally expect radians.

### AI Interface

The AI layer is in `ai_interface/` and centers on a simple contract: read a `GameState`, return a list of command strings for your team. Helpers are provided for movement, shooting, interception, and goalie behavior.

#### SoccerAI Contract

The default AI is `ai_interface/naive.py::SoccerAI`. Any custom AI should match this interface:

```python
from networking.data_utils import GameState

class SoccerAI:
    def __init__(self, team_info):
        ...

    def decide_action(self, game_state: GameState, teamname: str):
        # return list of command strings, one per robot
        return ["dash 0 0", ...]

    def translate_ai_output(self, ai_output):
        return ai_output
```

### Utilities

`ai_interface/utils/` provides reusable building blocks:

- `basic_commands.py`: low-level command helpers (goto/turn/kick/shoot/dribble) with obstacle detours built from `GameState`.
- `algo_utils.py`: math helpers (angle normalization, bisector target, ball velocity estimation).
- `intercept.py`: discrete-time interception solver (`earliest_intercept_control`) and a constant-friction variant for analysis.
- `test.py`: a visualization harness for interception experiments (uses `matplotlib`).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 UCSD RoboCup TritonBots

## Contact

**Team Leader** Lukas Cao \<lucao@ucsd.edu>, 

**Branch Owner** Travis Wu \<miw039@ucsd.edu>

IEEE@UCSD https://ieeeatucsd.org/