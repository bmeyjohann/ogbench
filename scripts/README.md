# OGBench Teleoperation Scripts

This directory contains example scripts demonstrating human teleoperation and intervention capabilities for OGBench point maze environments.

## Overview

We've implemented a teleoperation system similar to gymnasium-maze but adapted for OGBench environments. The system includes:

- **Gamepad Support**: Uses pygame to interface with external controllers (Xbox, PlayStation, etc.)
- **Keyboard Fallback**: Automatically falls back to keyboard controls if no gamepad is detected
- **Two Control Modes**: Direct teleoperation and human intervention

## Scripts

### 1. `teleoperation_example.py`

**Direct teleoperation** - Complete human control of the agent.

```bash
cd ogbench/scripts
python teleoperation_example.py
```

**Features:**
- Choose from available point maze environments
- Direct control using gamepad or keyboard
- Automatic goal detection and episode reset
- Real-time feedback

### 2. `intervention_example.py`

**Human intervention** - Policy runs autonomously with human override capability.

```bash
cd ogbench/scripts
python intervention_example.py
```

**Features:**
- Choose environment and policy type (heuristic/random/zero)
- Policy controls agent by default
- Human can intervene when needed
- Tracks intervention statistics
- Configurable override thresholds and hold times

## Controls

### Gamepad Controls
- **Movement**: Left analog stick
- **Reset**: X/Cross button (PlayStation) or A button (Xbox)
- **Quit**: Circle button (PlayStation) or B button (Xbox)

### Keyboard Controls (Fallback)
- **Movement**: Arrow keys (← → ↑ ↓)
- **Reset**: Space bar
- **Quit**: Escape key

## Installation Requirements

Make sure you have the following dependencies installed:

```bash
pip install pygame numpy gymnasium
```

The OGBench package should be installed in development mode:

```bash
cd ogbench
pip install -e .
```

## Usage Examples

### Quick Start - Direct Control
```python
import gymnasium as gym
import ogbench
from ogbench.ui import TeleopPoint2D
from ogbench.wrappers import DirectTeleopWrapper

# Create environment
env = gym.make('pointmaze-medium-v0', render_mode="human")

# Setup teleoperation
teleop = TeleopPoint2D(deadzone=0.1, use_keyboard_fallback=True)
env = DirectTeleopWrapper(env, teleop)

# Run
obs, info = env.reset()
while True:
    obs, reward, terminated, truncated, info = env.step()
    if terminated or truncated:
        obs, info = env.reset()
```

### Quick Start - Human Intervention
```python
import gymnasium as gym
import ogbench
from ogbench.ui import TeleopPoint2D
from ogbench.wrappers import HumanInterventionWrapper

# Create environment
env = gym.make('pointmaze-medium-v0', render_mode="human")

# Setup intervention
teleop = TeleopPoint2D(deadzone=0.15, use_keyboard_fallback=True)
env = HumanInterventionWrapper(env, teleop, threshold=0.2, hold_time=0.5)

# Run with policy
obs, info = env.reset()
while True:
    policy_action = env.action_space.sample()  # Your policy here
    obs, reward, terminated, truncated, info = env.step(policy_action)
    
    # Check if human intervened
    if info.get("human_override", False):
        print(f"Human intervention: {info['intervene_action']}")
    
    if terminated or truncated:
        obs, info = env.reset()
```

## Architecture

### Components

1. **TeleopPoint2D** (`ogbench.ui.TeleopPoint2D`):
   - Main interface for human input
   - Handles gamepad detection and keyboard fallback
   - Converts input to action vectors

2. **HumanInterventionWrapper** (`ogbench.wrappers.HumanInterventionWrapper`):
   - Wraps environments to allow policy override
   - Configurable intervention thresholds
   - Tracks intervention statistics

3. **DirectTeleopWrapper** (`ogbench.wrappers.DirectTeleopWrapper`):
   - Provides complete human control
   - Useful for demonstrations and exploration

### Key Features

- **Automatic Fallback**: Seamlessly switches to keyboard if no gamepad is detected
- **Configurable Thresholds**: Adjust sensitivity to prevent accidental interventions
- **Hold Time**: Maintains intervention for a period after input stops
- **Statistics Tracking**: Monitor intervention frequency and patterns
- **Multiple Environments**: Works with all OGBench point maze variants

## Troubleshooting

### No Gamepad Detected
The system automatically falls back to keyboard controls. Make sure your gamepad is:
- Properly connected
- Recognized by your system
- Initialized before running the script

### Performance Issues
If you experience lag or performance issues:
- Reduce the rendering frame rate
- Increase the sleep time in the main loop
- Use a simpler environment (e.g., medium instead of giant)

### Import Errors
Make sure OGBench is installed in development mode:
```bash
cd ogbench
pip install -e .
```

## Extending the System

### Adding New Input Devices
Extend the `TeleopPoint2D` class to support additional input methods:

```python
class CustomTeleopPoint2D(TeleopPoint2D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize custom device
    
    def get_action(self):
        # Implement custom action logic
        return custom_action
```

### Custom Policies
Create custom policies for intervention demos:

```python
class MyPolicy:
    def get_action(self, obs, env):
        # Implement your policy logic
        return action
```

### Environment-Specific Adaptations
The system is designed to work with any 2D action space environment. For other OGBench environments, you may need to:
- Adjust action space mappings
- Modify observation processing
- Add environment-specific features

## Notes

- The system is primarily designed for point maze environments with 2D action spaces
- Gamepad mappings are based on standard PlayStation/Xbox controllers
- Keyboard controls use standard arrow keys for maximum compatibility
- All scripts include proper cleanup for pygame and environment resources 