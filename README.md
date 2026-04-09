# Obstacle Runner

A 3D cube agent learns to navigate obstacle courses using DQN reinforcement learning. Watch it fail, learn, and eventually conquer levels it couldn't initially complete.

---

## Run It

```bash
pip install -r requirements.txt
python obstacle_runner.py
```

**Headless test** (no window, fast training):
```bash
python obstacle_runner.py --headless
```

---

## Controls

| Key | Action |
|-----|--------|
| `R` | Reset current episode |
| `N` | Next level |
| `H` | Switch to headless mode |
| `,` / `.` | Decrease / increase epsilon (exploration) |
| `ESC` | Quit |

---

## What It Does

**Agent**: A white cube with 6 actions — forward, backward, strafe left/right, jump, jump+forward.

**5 Levels**:
1. **First Steps** — flat platform with one gap
2. **The Hop** — platforms at increasing heights
3. **Narrow Path** — platforms that narrow to 1.5 units wide
4. **The Climb** — 5-step staircase
5. **Moving Target** — platform with a sweeping obstacle

**Learning**: DQN neural net (12 inputs → 128 hidden → 6 outputs) trained by TD error. Epsilon-greedy exploration decays over time. Experiences stored in replay buffer and trained in batches.

**Reward signals**: Reaching goal = +100, moving toward goal = positive, falling = -50, step penalty = -0.1.

---

## Headless Results (200 episodes)

```
Episode 0: steps=169, reward=-0.3, wins=0/1 (0%)
Episode 100: steps=104, reward=-0.7, wins=4/101 (4%)
Episode 200: steps=87, reward=-4.2, wins=14/201 (7%)
```

Win rate climbs over time — the agent is learning.

---

## Requirements

- Python 3.10+
- pygame
- torch
- ursina
- numpy
