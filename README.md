# AI Team - RoboCup Soccer AI System

A Python-based AI framework for controlling robotic soccer teams in both simulation and real-world environments. The system supports the RoboCup Soccer Simulation Server and can be extended for physical robot control.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
  - [Core Components](#core-components)
  - [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Command Line Options](#command-line-options)
- [Development](#development)
  - [Implementing Custom AI](#implementing-custom-ai)
  - [Game State Structure](#game-state-structure)
- [Environment Modes](#environment-modes)
  - [Simulation Only](#simulation-only-sim-only)
  - [Mixed Mode](#mixed-mode-sim-mixed)
  - [Field Practice](#field-practice-field-practice)
  - [Tournament Mode](#tournament-mode-field-tournament)
- [Technical Details](#technical-details)
  - [Networking Protocol](#networking-protocol)
  - [Data Processing](#data-processing)
  - [Threading Model](#threading-model)
- [License](#license)
- [Contact](#contact)

## Overview

This project provides a complete infrastructure for developing AI strategies for soccer-playing robots. It includes networking components for communication with simulators and robots, data processing utilities, and a modular AI interface for implementing custom strategies.

## Features

- **Multi-environment Support**: Works with simulation, mixed simulation/physical, and field environments
- **SSL Vision Integration**: Camera-based game state processing using SSL vision protocol
- **Concurrent Team Control**: Can control multiple teams simultaneously using threading
- **Modular AI Interface**: Easy-to-extend AI system for implementing custom strategies
- **Real-time Game State Processing**: Efficient parsing and processing of game state data from simulators and cameras
- **Flexible Network Communication**: Support for simulator, robot, and camera communication protocols

## Architecture

### Core Components

- **Main Entry Point** (`__main__.py`): Application entry point with command-line interface
- **AI Interface** (`ai_interface/`): Pluggable AI system for decision making
- **Networking** (`networking/`): Communication layer for simulators and robots
  - `networker.py`: High-level networking coordinator
  - `socket_utils.py`: Socket management and communication protocols
  - `data_utils.py`: Data serialization/deserialization utilities

### Directory Structure

```
ai-team/
├── __main__.py           # Main application entry point
├── ai_interface/         # AI strategy implementations
│   └── naive.py         # Basic AI implementation
├── networking/          # Network communication layer
│   ├── networker.py     # Main networking coordinator
│   ├── socket_utils.py  # Socket utilities and protocols
│   └── data_utils.py    # Data processing utilities
├── game_logs/           # Game state logs (generated)
└── text_logs/           # Debug/info logs (generated)
```

## Installation

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

This creates a new "rcai" Python 3.11 environment with all required dependencies including `sslclient` for camera integration.

3. Activate the environment:
```bash
conda activate rcai
```

**Note for developers**: When adding dependencies, regenerate the environment file using:
```bash
conda env export --no-build | grep -vE "prefix: |_libgcc_mutex|_openmp_mutex|ld_impl_linux-64|libgcc-ng|libgomp|libstdcxx-ng" > environment.yml
```

## Usage

### Basic Usage

Run the AI system with default settings:
```bash
python .
```

### Command Line Options

- `--teamname`: Set team name (default: "TritonBots")
- `--env`: Set environment mode
  - `sim-only`: Simulator only (default)
  - `sim-mixed`: Mixed simulator and physical robots
  - `field-practice`: Physical robots with camera
  - `field-tournament`: Tournament mode (own team only)

## Development

### Implementing Custom AI

1. Create a new AI class in `my_custom_ai.py` under `ai_interface/`:

```python
from networking.data_utils import GameState

class SoccerAI:
    def __init__(self):
        # Initialize your AI
        pass
    
    def decide_action(self, game_state: GameState, teamname: str):
        # Implement your decision logic
        output = None
        # ... your logic here
        return output

    def translate_ai_output(self, ai_output):
        # Translate AI decisions to robot commands
        return ai_output
```

2. Update `__main__.py` to use your custom AI:

```python
from ai_interface.my_custom_ai import SoccerAI
```

### Game State Structure

The `GameState` object contains:
- `count`: Server cycle number
- `timestamp`: Current timestamp
- `ball_pos`: Ball position (x, y)
- `robot_poses`: Dictionary of team robot positions

## Environment Modes

### Simulation Only (`sim-only`)
- One or both teams controlled by AI in simulation
- Uses RoboCup Soccer Server
- Ideal for development and testing

### Mixed Mode (`sim-mixed`)
- Combination of simulated and physical robots
- Requires additional hardware setup

### Field Practice (`field-practice`)
- Physical robots with SSL vision camera system
- Real-time ball and robot detection via camera
- One or both teams can be controlled

### Tournament Mode (`field-tournament`)
- Only controls your own team
- Other team controlled by opponents
- SSL vision-based positioning and game state

## Technical Details

### Networking Protocol

**Simulation Mode**: Communicates with the Simulation Server using UDP sockets on localhost:
- **Client Port**: 6000 (robot commands)
- **Trainer Port**: 6001 (game state monitoring)

**Camera Mode**: Uses SSL vision protocol via `sslclient`:
- Receives detection data with ball and robot positions
- Processes protobuf-formatted vision messages
- Real-time field coordinate system

### Data Processing

The system processes game state data from two sources:

**Simulator Data**: Parsed using regex patterns to extract:
- Ball position from simulation messages
- Robot poses and orientations
- Game timing information

**Camera Data**: Processed from SSL vision protocol to extract:
- Ball position with confidence filtering
- Robot detection with pattern IDs
- Real-time field coordinates and orientations

### Threading Model

The system uses threading to:
- Process multiple teams concurrently
- Send commands to multiple robots simultaneously
- Maintain responsive network communication

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 UCSD RoboCup TritonBots

## Contact

[Add contact information here]