# Obstacle Runner — 3D RL Agent

## Concept
A cube agent learns to navigate 3D obstacle courses using reinforcement learning. Start with simple gaps and platforms, progress to moving obstacles, timed jumps, and multi-stage courses. Watch the agent fail, learn, and eventually conquer levels it couldn't initially complete.

---

## Architecture

### Tech Stack
- **Rendering**: Ursina (Python 3D engine, simple primitives, cross-platform)
- **Physics**: Custom AABB collision + gravity (no external physics engine needed for a cube)
- **RL Brain**: DQN neural net (PyTorch) for discrete action selection
- **Observation space**: Proprioceptive — agent velocity, position relative to goal, ground contact, nearby obstacle proximity (12 dims)
- **Action space**: Discrete 6 — forward/back/left/right, jump, jump+forward

### Agent Actions
| ID | Action |
|----|--------|
| 0 | Move forward |
| 1 | Move backward |
| 2 | Strafe left |
| 3 | Strafe right |
| 4 | Jump |
| 5 | Jump + forward |

### Reward Signals
- **Reaching goal**: +100
- **Moving toward goal**: +distance_reduction × 10
- **Moving away**: -5
- **Falling off**: -50
- **Step penalty**: -0.1 (encourages speed)
- **Time penalty**: -0.5/s (encourages efficiency)

---

## Level Progression

### Level 1 — "First Steps"
Flat platform with one gap (2 units wide). Agent must jump across.
- Mostly solvable by random exploration
- Teaches: jump timing

### Level 2 — "The Hop"
Three platforms at increasing heights, gaps between each.
- Requires precise jump + forward timing
- Teaches: jump arc, height

### Level 3 — "Narrow Path"
Platform narrows to 1 unit wide, then widens again.
- Tests precision movement
- Teaches: strafe control

### Level 4 — "The Climb"
Staircase of 5 platforms, each slightly higher.
- Must chain multiple jumps
- Teaches: chained actions

### Level 5 — "Moving Target"
Platform with a moving block that sweeps across the path.
- Agent must time movement through gaps
- Teaches: timing, prediction

### Level 6+ — "Gauntlet"
Combines all previous challenges + new elements (timed doors, smaller platforms, longer gaps).

---

## Neural Network Architecture

```
Input: 12 dims
  - agent velocity (3)
  - agent to goal vector normalized (3)
  - ground contact (1)
  - distance to nearest obstacle (1)
  - obstacle relative position if nearest (3)
  - time alive normalized (1)

Hidden: 128 ReLU

Output: 6 (Q-values per action)
```

Training: DQN with experience replay, target network, epsilon-greedy exploration.

---

## Rendering

- **Ursina** window 800×600
- Agent: white cube with colored trail showing recent path
- Platforms: dark gray with subtle grid texture (via color)
- Goal: glowing green platform
- Obstacles: red/orange blocks
- UI overlay: current level, episode count, success rate, epsilon

---

## Controls
- `SPACE` — restart current episode
- `N` — next level (skip)
- `R` — reset to level 1
- `,/.` — decrease/increase epsilon (exploration)

---

## Success Criteria
An agent that:
1. Completes Level 1 within 20 episodes
2. Completes Level 3 within 100 episodes
3. Shows visibly improving performance on graph over time
4. Can complete all 5 levels in under 1000 episodes

---

## File Structure
```
obstacle_runner.py   — main game + training loop
level.py             — level definitions
agent.py             — DQN brain, experience replay
design.md            — this file
requirements.txt     — ursina, torch, numpy
```
