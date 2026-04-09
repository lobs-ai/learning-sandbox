# Learning Sandbox

A digital ecosystem where prey and predators both have neural net brains, both learn in real time, and both pass down trained weights to their offspring with mutation.

Watch evolution and learning happen simultaneously.

---

## Quick Start

```bash
cd learning-sandbox
pip install -r requirements.txt
python ecosystem.py
```

---

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Toggle 1x / 5x speed |
| `R` | Reset simulation |
| `G` | Toggle population graph |
| `Click` | Add food at cursor |

---

## What You're Watching

**Prey** (green dots): Stand still to gather food. Have MLP brains that learn via policy gradient — expand or shrink detection radius based on reward. Reproduce with inherited weights + mutation.

**Predators** (red dots): Hunt prey. Have MLP brains that learn targeting strategy. Reproduce with inherited weights + mutation.

**Population graph**: Green = prey count, Red = predator count. Classic Lotka-Volterra dynamics — boom and bust cycles.

---

## Requirements

- Python 3.10+
- pygame
- numpy

---

## Project Structure

```
ecosystem.py      — Main simulation
test_ecosystem.py — Unit tests
requirements.txt  — Dependencies
.gitignore       — Git ignores
DESIGN.md        — Full design doc
README.md        — This file
```
