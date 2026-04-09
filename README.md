# Learning Sandbox

A digital ecosystem where predators hunt using **DQN neural nets** trained by TD error, and prey learn with tabular Q-tables. Both populations inherit learned weights to their offspring with mutation. Watch intelligence emerge.

---

## Quick Start

```bash
pip install -r requirements.txt
python ecosystem.py
```

---

## What Each Agent Learns

**Predators (red)**: DQN neural net — 7 inputs (3 nearest prey positions + own energy), 64-unit hidden layer, 3 outputs (turn left/right/straight). Trained by TD error with experience replay and a target network. Each catch is a reward signal; each miss is a penalty. Over generations, they learn to anticipate prey movement.

**Prey (green)**: Tabular Q-learning — discretized state (food direction, predator direction, energy level) → action values. They learn to seek food and flee predators.

**Both**: Reproduction passes down learned weights with small Gaussian mutation. Selection pressure means better hunters and smarter prey reproduce more.

---

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Toggle 1x / 5x speed |
| `R` | Reset simulation |
| `G` | Toggle population graph |
| `Click` | Add food at cursor |

---

## Requirements

- Python 3.10+
- pygame >= 2.0.0
- torch >= 2.0.0
- numpy
